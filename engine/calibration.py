"""
calibration.py — Two-Stage SVJ Calibration Engine.

Stage 1: Fit Heston core (κ, θ, ξ, ρ, v₀) to ATM + near-money strikes.
Stage 2: Add jump parameters (λ, μ_J, σ_J) to fit wings.

Features:
- Vega-weighted objective: w_i = Vega_i / BidAskSpread_i
- Tikhonov regularization on ξ, ρ, λ
- Forward-space calibration
- Parameter logging
"""

import numpy as np
import logging
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Optional, Tuple, List
from engine.models import SVJParams, forward_price
from engine.monte_carlo import MonteCarloEngine, bs_price
from engine.surface import bs_vega
from engine.config import (
    PARAM_BOUNDS, REGULARIZATION, CALIBRATION_CONFIG,
    clamp_params, check_feller
)

logger = logging.getLogger("calibration")


def compute_vega_weights(
    spot: float, strikes: np.ndarray, T: float,
    r: float, q: float, atm_vol: float,
    bid_ask_spreads: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute vega-weighted calibration weights.
    w_i = Vega_i / BidAskSpread_i

    Falls back to pure vega weighting if no spread data.
    """
    vegas = np.array([
        max(bs_vega(spot, K, T, r, q, atm_vol), 1e-10) for K in strikes
    ])

    if bid_ask_spreads is not None:
        spreads = np.maximum(bid_ask_spreads, 1e-4)
        weights = vegas / spreads
    else:
        weights = vegas

    return weights / np.sum(weights)


def _heston_objective(
    x: np.ndarray,
    spot: float, strikes: np.ndarray, T: float,
    market_prices: np.ndarray, weights: np.ndarray,
    r: float, q: float, is_call: bool,
    num_paths: int = 100_000, num_steps: int = 100
) -> float:
    """
    Stage 1 objective: Heston core (no jumps).
    Params: [kappa, theta, xi, rho, v0]
    """
    kappa, theta, xi, rho, v0 = x

    # Enforce Feller soft penalty
    feller_penalty = 0.0
    if not check_feller(kappa, theta, xi):
        feller_violation = xi**2 - 2 * kappa * theta
        feller_penalty = 10.0 * feller_violation**2

    params = SVJParams(
        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
        lambda_j=0.0, mu_j=0.0, sigma_j=0.01,  # No jumps in stage 1
        r=r, q=q
    )

    engine = MonteCarloEngine(params, num_paths=num_paths, num_steps=num_steps,
                              use_sobol=True, use_antithetic=True,
                              use_control_variate=True)

    total_err = 0.0
    for i, K in enumerate(strikes):
        try:
            result = engine.price(spot, K, T, is_call=is_call)
            model_price = result["price"]
            total_err += weights[i] * (model_price - market_prices[i])**2
        except Exception:
            total_err += 1.0

    # Tikhonov regularization
    reg = (REGULARIZATION["xi"] * xi**2 +
           REGULARIZATION["rho"] * rho**2)

    return total_err + reg + feller_penalty


def _svj_objective(
    x_jump: np.ndarray,
    heston_params: np.ndarray,
    spot: float, strikes: np.ndarray, T: float,
    market_prices: np.ndarray, weights: np.ndarray,
    r: float, q: float, is_call: bool,
    num_paths: int = 100_000, num_steps: int = 100
) -> float:
    """
    Stage 2 objective: Full SVJ with fixed Heston core.
    Params: [lambda_j, mu_j, sigma_j]
    """
    lambda_j, mu_j, sigma_j = x_jump
    kappa, theta, xi, rho, v0 = heston_params

    params = SVJParams(
        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
        lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j,
        r=r, q=q
    )

    engine = MonteCarloEngine(params, num_paths=num_paths, num_steps=num_steps,
                              use_sobol=True, use_antithetic=True,
                              use_control_variate=True)

    total_err = 0.0
    for i, K in enumerate(strikes):
        try:
            result = engine.price(spot, K, T, is_call=is_call)
            model_price = result["price"]
            total_err += weights[i] * (model_price - market_prices[i])**2
        except Exception:
            total_err += 1.0

    # Regularization on jump intensity
    reg = REGULARIZATION["lambda_j"] * lambda_j**2

    return total_err + reg


class CalibrationEngine:
    """
    Two-stage SVJ calibration with vega-weighted objective and regularization.
    """

    def __init__(self, config=None):
        self.config = config or CALIBRATION_CONFIG
        self.history = []  # Log parameter evolution

    def calibrate(
        self,
        spot: float, strikes: np.ndarray, T: float,
        market_prices: np.ndarray, is_call: bool = True,
        r: float = 0.065, q: float = 0.012,
        bid_ask_spreads: Optional[np.ndarray] = None,
        atm_vol: float = 0.15,
        num_paths: int = 100_000
    ) -> Dict:
        """
        Run two-stage calibration.

        Returns:
            Dict with calibrated SVJParams, errors, convergence info.
        """
        # Compute forward price for forward-space calibration
        F = forward_price(spot, r, q, T)
        moneyness = strikes / F

        # Filter strikes by stage
        cfg = self.config
        stage1_mask = (moneyness >= cfg.stage1_moneyness_range[0]) & \
                      (moneyness <= cfg.stage1_moneyness_range[1])
        stage2_mask = (moneyness >= cfg.stage2_moneyness_range[0]) & \
                      (moneyness <= cfg.stage2_moneyness_range[1])

        stage1_strikes = strikes[stage1_mask]
        stage1_prices = market_prices[stage1_mask]
        stage2_strikes = strikes[stage2_mask]
        stage2_prices = market_prices[stage2_mask]

        # Weights
        stage1_bas = bid_ask_spreads[stage1_mask] if bid_ask_spreads is not None else None
        stage2_bas = bid_ask_spreads[stage2_mask] if bid_ask_spreads is not None else None
        w1 = compute_vega_weights(spot, stage1_strikes, T, r, q, atm_vol, stage1_bas)
        w2 = compute_vega_weights(spot, stage2_strikes, T, r, q, atm_vol, stage2_bas)

        # ─── Stage 1: Heston Core ───
        logger.info("Stage 1: Fitting Heston core to %d ATM-near strikes", len(stage1_strikes))

        heston_bounds = [
            PARAM_BOUNDS["kappa"],
            PARAM_BOUNDS["theta"],
            PARAM_BOUNDS["xi"],
            PARAM_BOUNDS["rho"],
            PARAM_BOUNDS["v0"],
        ]

        result1 = differential_evolution(
            _heston_objective,
            bounds=heston_bounds,
            args=(spot, stage1_strikes, T, stage1_prices, w1, r, q, is_call, num_paths, 50),
            maxiter=cfg.stage1_max_iter,
            tol=cfg.ftol,
            seed=42,
            workers=1,
        )

        heston_params = result1.x
        logger.info("Stage 1 complete: κ=%.3f θ=%.4f ξ=%.3f ρ=%.3f v₀=%.4f error=%.6f",
                     *heston_params, result1.fun)

        # ─── Stage 2: Add Jumps ───
        logger.info("Stage 2: Fitting jump parameters to %d strikes", len(stage2_strikes))

        jump_bounds = [
            PARAM_BOUNDS["lambda_j"],
            PARAM_BOUNDS["mu_j"],
            PARAM_BOUNDS["sigma_j"],
        ]

        result2 = differential_evolution(
            _svj_objective,
            bounds=jump_bounds,
            args=(heston_params, spot, stage2_strikes, T, stage2_prices, w2,
                  r, q, is_call, num_paths, 50),
            maxiter=cfg.stage2_max_iter,
            tol=cfg.ftol,
            seed=42,
            workers=1,
        )

        jump_params = result2.x
        logger.info("Stage 2 complete: λ=%.3f μ_J=%.4f σ_J=%.4f error=%.6f",
                     *jump_params, result2.fun)

        # Build final params
        final_params = SVJParams(
            kappa=heston_params[0], theta=heston_params[1],
            xi=heston_params[2], rho=heston_params[3], v0=heston_params[4],
            lambda_j=jump_params[0], mu_j=jump_params[1], sigma_j=jump_params[2],
            r=r, q=q
        )

        # Validate
        warnings = final_params.validate()

        # Log history
        entry = {
            "params": final_params.to_array().tolist(),
            "stage1_error": float(result1.fun),
            "stage2_error": float(result2.fun),
            "warnings": warnings,
        }
        self.history.append(entry)

        return {
            "params": final_params,
            "stage1_result": {
                "error": float(result1.fun),
                "nit": result1.nit,
                "success": result1.success,
            },
            "stage2_result": {
                "error": float(result2.fun),
                "nit": result2.nit,
                "success": result2.success,
            },
            "warnings": warnings,
            "feller_satisfied": final_params.feller_satisfied,
        }

    def get_history(self) -> List[Dict]:
        """Return calibration parameter history for stability analysis."""
        return self.history
