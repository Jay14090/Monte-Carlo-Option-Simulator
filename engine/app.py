"""
app.py — FastAPI server for the NIFTY Monte Carlo Pricing & Risk Engine.

Endpoints:
- POST /api/price         — Price a single option
- POST /api/greeks        — Compute all Greeks
- POST /api/stress        — Run stress test ladder
- POST /api/regime        — Detect current market regime
- POST /api/calibrate     — Run two-stage calibration
- POST /api/hedge         — Run hedging backtest
- GET  /api/health        — Health check
"""

import os
import time
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from engine.models import SVJParams
from engine.monte_carlo import MonteCarloEngine
from engine.greeks import GreeksEngine
from engine.risk import StressTestEngine, HedgingBacktest, compute_risk_metrics, LiquidityStress
from engine.regime import RegimeDetector, MarketRegime, compute_realized_vol, compute_iv_percentile, compute_skew_slope
from engine.guards import PricingGuard
from engine.config import RISK_FREE_RATE, DIVIDEND_YIELD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="NIFTY Monte Carlo Engine",
    description="Trading-desk grade SVJ pricing & risk engine for NIFTY index options",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────
class SVJParamsRequest(BaseModel):
    kappa: float = Field(3.0, description="Mean reversion speed")
    theta: float = Field(0.04, description="Long-run variance")
    xi: float = Field(0.5, description="Vol-of-vol")
    rho: float = Field(-0.7, description="Spot-vol correlation")
    v0: float = Field(0.04, description="Initial variance")
    lambda_j: float = Field(1.0, description="Jump intensity")
    mu_j: float = Field(-0.05, description="Mean jump size (log)")
    sigma_j: float = Field(0.10, description="Jump size volatility")
    r: float = Field(RISK_FREE_RATE, description="Risk-free rate")
    q: float = Field(DIVIDEND_YIELD, description="Dividend yield")

    def to_params(self) -> SVJParams:
        return SVJParams(**self.dict())


class PriceRequest(BaseModel):
    spot: float = Field(..., description="Current spot price")
    strike: float = Field(..., description="Strike price")
    T: float = Field(..., description="Time to maturity (years)")
    is_call: bool = Field(True, description="True for call, False for put")
    params: SVJParamsRequest = SVJParamsRequest()
    num_paths: int = Field(500_000, description="Number of MC paths")
    use_sobol: bool = Field(True, description="Use Sobol quasi-random")
    use_antithetic: bool = Field(True, description="Use antithetic variates")
    use_control_variate: bool = Field(True, description="Use BS control variate")


class GreeksRequest(BaseModel):
    spot: float
    strike: float
    T: float
    is_call: bool = True
    params: SVJParamsRequest = SVJParamsRequest()
    num_paths: int = 200_000


class StressRequest(BaseModel):
    spot: float
    strike: float
    T: float
    is_call: bool = True
    params: SVJParamsRequest = SVJParamsRequest()
    num_paths: int = 100_000


class RegimeRequest(BaseModel):
    realized_vol: float = Field(..., description="Annualized realized vol")
    iv_percentile: float = Field(..., description="IV percentile rank (0-100)")
    skew_slope: float = Field(..., description="25Δ put IV − 25Δ call IV")


class HedgeRequest(BaseModel):
    spot: float
    strike: float
    T: float
    is_call: bool = True
    params: SVJParamsRequest = SVJParamsRequest()
    num_scenarios: int = 500
    txn_cost_bps: float = 5.0
    slippage_bps: float = 2.0


class SmileRequest(BaseModel):
    spot: float
    T: float
    params: SVJParamsRequest = SVJParamsRequest()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "healthy", "engine": "SVJ Monte Carlo", "version": "1.0.0"}


@app.post("/api/price")
async def price_option(req: PriceRequest):
    """Price a European option using SVJ Monte Carlo."""
    start = time.time()
    svj = req.params.to_params()

    # Pre-price guard
    guard = PricingGuard(svj)
    pre = guard.check_pre_price(req.spot, req.strike, req.T)
    if not pre["pass"]:
        raise HTTPException(400, detail={"failures": pre["failures"], "alerts": pre["alerts"]})

    engine = MonteCarloEngine(
        svj, num_paths=req.num_paths, use_sobol=req.use_sobol,
        use_antithetic=req.use_antithetic, use_control_variate=req.use_control_variate
    )
    result = engine.price(req.spot, req.strike, req.T, req.is_call)

    # Add sample paths for visualization
    sample_paths = engine.get_sample_paths(req.spot, req.T, num_samples=50)
    result["sample_paths"] = sample_paths.tolist()

    # Post-price guard
    post = guard.check_post_price(result, req.spot, req.strike, req.T, req.is_call)

    elapsed = time.time() - start
    result["elapsed_ms"] = round(elapsed * 1000, 1)
    result["pre_checks"] = pre
    result["post_checks"] = post
    result["params_used"] = req.params.dict()

    logger.info("Priced %s K=%.0f T=%.4f → %.4f (%.0fms)",
                "Call" if req.is_call else "Put",
                req.strike, req.T, result["price"], elapsed * 1000)
    return result


@app.post("/api/greeks")
async def compute_greeks(req: GreeksRequest):
    """Compute all Greeks."""
    start = time.time()
    svj = req.params.to_params()
    engine = GreeksEngine(svj, num_paths=req.num_paths)
    greeks = engine.all_greeks(req.spot, req.strike, req.T, req.is_call)
    elapsed = time.time() - start
    greeks["elapsed_ms"] = round(elapsed * 1000, 1)
    return greeks


@app.post("/api/stress")
async def run_stress(req: StressRequest):
    """Run full stress test suite."""
    start = time.time()
    svj = req.params.to_params()
    engine = StressTestEngine(svj, num_paths=req.num_paths)
    report = engine.full_stress_report(req.spot, req.strike, req.T, req.is_call)
    elapsed = time.time() - start
    report["elapsed_ms"] = round(elapsed * 1000, 1)
    return report


@app.post("/api/regime")
async def detect_regime(req: RegimeRequest):
    """Detect current market regime."""
    detector = RegimeDetector()
    result = detector.classify(req.realized_vol, req.iv_percentile, req.skew_slope)
    return result


@app.post("/api/hedge")
async def run_hedge_backtest(req: HedgeRequest):
    """Run hedging backtest."""
    start = time.time()
    svj = req.params.to_params()
    bt = HedgingBacktest(svj)
    result = bt.run_backtest(
        req.spot, req.strike, req.T, req.is_call,
        txn_cost_bps=req.txn_cost_bps, slippage_bps=req.slippage_bps,
        num_scenarios=req.num_scenarios
    )
    elapsed = time.time() - start
    result["elapsed_ms"] = round(elapsed * 1000, 1)
    return result


@app.post("/api/smile")
async def generate_smile(req: SmileRequest):
    """Generate volatility smile data for SVJ."""
    svj = req.params.to_params()
    engine = MonteCarloEngine(svj, num_paths=50000)
    
    # Range of strikes around spot
    strikes = np.linspace(req.spot * 0.7, req.spot * 1.3, 21)
    results = engine.price_batch(req.spot, strikes, req.T, is_call=True)
    
    # Compute Implied Vol for each
    from engine.surface import implied_vol
    smile = []
    for r in results:
        iv = implied_vol(r["price"], req.spot, r["strike"], req.T, svj.r, svj.q, True)
        smile.append({
            "strike": r["strike"],
            "price": r["price"],
            "iv": iv if iv is not None else 0.0
        })
    
    return {"smile": smile}


# Serve files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(base_dir, "index.html"))

@app.get("/advanced")
async def serve_advanced():
    return FileResponse(os.path.join(base_dir, "dashboard.html"))

# Mount static asset folders
app.mount("/js", StaticFiles(directory=os.path.join(base_dir, "js")), name="js")
app.mount("/css", StaticFiles(directory=os.path.join(base_dir, "css")), name="css")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
