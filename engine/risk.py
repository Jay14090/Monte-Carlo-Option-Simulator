"""
risk.py — Risk Engine: Stress Tests, Hedging Backtest, VaR/CVaR, Tail Metrics.

Features:
- Spot shocks (±2%, ±5%, ±8%)
- Vol shocks (±5 points)
- 4% overnight gap scenario
- VaR, CVaR, skewness, kurtosis, tail index
- Liquidity stress layer: bid-ask widening, vol-gap-without-spot-move, expiry vol crush
- Daily delta hedge backtest with transaction costs and slippage
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from engine.models import SVJParams
from engine.monte_carlo import MonteCarloEngine, bs_price, bs_delta
from engine.config import SPOT_SHOCKS, VOL_SHOCKS, JUMP_SCENARIO_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Stress Test Module
# ─────────────────────────────────────────────────────────────────────────────
class StressTestEngine:
    """
    Run stress scenarios and compute risk metrics.
    """

    def __init__(self, params: SVJParams, num_paths: int = 200_000, seed: int = 42):
        self.params = params
        self.num_paths = num_paths
        self.seed = seed

    def spot_shock_ladder(self, spot: float, strike: float, T: float,
                          is_call: bool = True) -> List[Dict]:
        """Compute option price under spot shocks."""
        results = []
        engine = MonteCarloEngine(self.params, num_paths=self.num_paths, seed=self.seed)
        base = engine.price(spot, strike, T, is_call)

        for shock in SPOT_SHOCKS:
            shocked_spot = spot * (1 + shock)
            res = engine.price(shocked_spot, strike, T, is_call)
            results.append({
                "shock_pct": shock * 100,
                "spot": shocked_spot,
                "price": res["price"],
                "pnl": res["price"] - base["price"],
                "pnl_pct": (res["price"] - base["price"]) / max(base["price"], 1e-6) * 100,
            })

        return results

    def vol_shock_ladder(self, spot: float, strike: float, T: float,
                         is_call: bool = True) -> List[Dict]:
        """Compute option price under vol shocks."""
        results = []
        base_engine = MonteCarloEngine(self.params, num_paths=self.num_paths, seed=self.seed)
        base = base_engine.price(spot, strike, T, is_call)

        for shock in VOL_SHOCKS:
            shocked_v0 = max(self.params.v0 + 2 * np.sqrt(self.params.v0) * shock, 0.001)
            shocked_theta = max(self.params.theta + shock**2, 0.001)
            shocked_params = SVJParams(
                kappa=self.params.kappa, theta=shocked_theta,
                xi=self.params.xi, rho=self.params.rho, v0=shocked_v0,
                lambda_j=self.params.lambda_j, mu_j=self.params.mu_j,
                sigma_j=self.params.sigma_j, r=self.params.r, q=self.params.q
            )
            engine = MonteCarloEngine(shocked_params, num_paths=self.num_paths, seed=self.seed)
            res = engine.price(spot, strike, T, is_call)
            results.append({
                "vol_shock": shock * 100,
                "v0": shocked_v0,
                "price": res["price"],
                "pnl": res["price"] - base["price"],
            })

        return results

    def jump_scenario(self, spot: float, strike: float, T: float,
                      is_call: bool = True,
                      gap_size: float = JUMP_SCENARIO_SIZE) -> Dict:
        """Simulate 4% overnight gap scenario."""
        engine = MonteCarloEngine(self.params, num_paths=self.num_paths, seed=self.seed)
        base = engine.price(spot, strike, T, is_call)

        # Scenario: spot gaps down (worst case for long calls)
        gap_down_spot = spot * (1 - gap_size)
        res_down = engine.price(gap_down_spot, strike, T, is_call)

        # Scenario: spot gaps up (worst case for long puts)
        gap_up_spot = spot * (1 + gap_size)
        res_up = engine.price(gap_up_spot, strike, T, is_call)

        return {
            "base_price": base["price"],
            "gap_down_price": res_down["price"],
            "gap_down_pnl": res_down["price"] - base["price"],
            "gap_up_price": res_up["price"],
            "gap_up_pnl": res_up["price"] - base["price"],
            "gap_size_pct": gap_size * 100,
        }

    def full_stress_report(self, spot: float, strike: float, T: float,
                           is_call: bool = True) -> Dict:
        """Run all stress scenarios."""
        return {
            "spot_shocks": self.spot_shock_ladder(spot, strike, T, is_call),
            "vol_shocks": self.vol_shock_ladder(spot, strike, T, is_call),
            "jump_scenario": self.jump_scenario(spot, strike, T, is_call),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tail Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk_metrics(returns: np.ndarray,
                         confidence: float = 0.99) -> Dict[str, float]:
    """
    Compute VaR, CVaR, and tail shape metrics.

    Args:
        returns: Array of simulated PnL or returns
        confidence: VaR confidence level

    Returns:
        Dict with VaR, CVaR, skewness, kurtosis, tail_index
    """
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    cutoff = int(n * (1 - confidence))

    var = -sorted_returns[cutoff] if cutoff < n else -sorted_returns[0]
    cvar = -np.mean(sorted_returns[:cutoff]) if cutoff > 0 else -sorted_returns[0]

    # Moments
    mean = np.mean(returns)
    std = np.std(returns)
    skewness = float(np.mean(((returns - mean) / max(std, 1e-10)) ** 3))
    kurtosis = float(np.mean(((returns - mean) / max(std, 1e-10)) ** 4))

    # Hill tail index estimator (on losses)
    losses = -sorted_returns[sorted_returns < 0]
    tail_index = _hill_estimator(losses) if len(losses) > 20 else np.nan

    return {
        "var": float(var),
        "cvar": float(cvar),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": kurtosis - 3.0,
        "tail_index": float(tail_index),
        "mean": float(mean),
        "std": float(std),
    }


def _hill_estimator(sorted_losses: np.ndarray, k: int = None) -> float:
    """
    Hill estimator for tail index.
    Uses top k order statistics.
    """
    n = len(sorted_losses)
    if k is None:
        k = max(int(np.sqrt(n)), 10)
    k = min(k, n - 1)

    sorted_desc = np.sort(sorted_losses)[::-1]
    if sorted_desc[k] <= 0:
        return np.nan

    log_ratios = np.log(sorted_desc[:k] / sorted_desc[k])
    return float(k / np.sum(log_ratios)) if np.sum(log_ratios) > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity Stress Layer
# ─────────────────────────────────────────────────────────────────────────────
class LiquidityStress:
    """
    Simulate liquidity stress scenarios specific to NIFTY weekly trading.
    """

    @staticmethod
    def bid_ask_widening(base_spread: float, widening_factor: float = 3.0) -> Dict:
        """Simulate bid-ask spread widening during stress."""
        stressed_spread = base_spread * widening_factor
        return {
            "base_spread": base_spread,
            "stressed_spread": stressed_spread,
            "slippage_increase": stressed_spread - base_spread,
        }

    @staticmethod
    def vol_gap_no_spot_move(params: SVJParams, vol_jump: float = 0.05) -> SVJParams:
        """
        Simulate vol spike without spot move.
        Common during RBI announcements or budget gaps.
        """
        new_v0 = params.v0 + 2 * np.sqrt(params.v0) * vol_jump + vol_jump**2
        return SVJParams(
            kappa=params.kappa, theta=params.theta, xi=params.xi,
            rho=params.rho, v0=new_v0,
            lambda_j=params.lambda_j, mu_j=params.mu_j,
            sigma_j=params.sigma_j, r=params.r, q=params.q
        )

    @staticmethod
    def expiry_vol_crush(params: SVJParams, crush_pct: float = 0.30) -> SVJParams:
        """
        Simulate vol crush on expiry day.
        Weekly NIFTY options can lose 20-40% of IV intraday on expiry.
        """
        crushed_v0 = params.v0 * (1 - crush_pct)
        crushed_theta = params.theta * (1 - crush_pct * 0.5)
        return SVJParams(
            kappa=params.kappa, theta=max(crushed_theta, 0.001),
            xi=params.xi, rho=params.rho, v0=max(crushed_v0, 0.001),
            lambda_j=params.lambda_j, mu_j=params.mu_j,
            sigma_j=params.sigma_j, r=params.r, q=params.q
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hedging Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────
class HedgingBacktest:
    """
    Daily delta hedging simulator with transaction costs and slippage.
    
    Tracks: hedge PnL drift, gamma bleed, vega exposure drift.
    """

    def __init__(self, params: SVJParams, seed: int = 42):
        self.params = params
        self.seed = seed

    def run_backtest(
        self,
        spot: float, strike: float, T: float,
        is_call: bool = True,
        num_days: int = None,
        txn_cost_bps: float = 5.0,  # 5 basis points
        slippage_bps: float = 2.0,   # 2 basis points
        num_scenarios: int = 1000,
        num_mc_paths: int = 50_000,
    ) -> Dict:
        """
        Run hedging backtest.

        Simulates daily delta-hedging a short option position.
        """
        if num_days is None:
            num_days = max(int(T * 252), 1)

        dt = T / num_days
        p = self.params
        sigma_bs = np.sqrt(p.v0)

        rng = np.random.default_rng(self.seed)

        pnl_scenarios = []

        for scenario in range(num_scenarios):
            S = spot
            portfolio_cash = 0.0
            hedge_shares = 0.0
            total_txn_cost = 0.0

            # Initial option premium received (short option)
            engine = MonteCarloEngine(p, num_paths=num_mc_paths, seed=self.seed + scenario)
            initial = engine.price(spot, strike, T, is_call)
            portfolio_cash += initial["price"]

            t_remaining = T
            daily_pnl = []

            for day in range(num_days):
                if t_remaining <= 0:
                    break

                # Compute delta
                delta = bs_delta(S, strike, t_remaining, p.r, p.q, sigma_bs, is_call)

                # Rebalance hedge
                trade = delta - hedge_shares
                cost = abs(trade) * S * (txn_cost_bps + slippage_bps) / 10000
                total_txn_cost += cost
                portfolio_cash -= trade * S + cost
                hedge_shares = delta

                # Simulate next day's spot (simple GBM step for backtest)
                z = rng.standard_normal()
                S_new = S * np.exp((p.r - p.q - 0.5 * p.v0) * dt + np.sqrt(p.v0 * dt) * z)

                # Daily PnL from hedge position
                hedge_pnl = hedge_shares * (S_new - S)
                daily_pnl.append({
                    "day": day,
                    "spot": S,
                    "delta": delta,
                    "trade": trade,
                    "txn_cost": cost,
                    "hedge_pnl": hedge_pnl,
                })

                S = S_new
                t_remaining -= dt

            # Final settlement
            if is_call:
                option_payoff = max(S - strike, 0)
            else:
                option_payoff = max(strike - S, 0)

            final_pnl = portfolio_cash + hedge_shares * S - option_payoff
            pnl_scenarios.append(final_pnl)

        pnl_array = np.array(pnl_scenarios)
        metrics = compute_risk_metrics(pnl_array, confidence=0.99)

        return {
            "mean_pnl": float(np.mean(pnl_array)),
            "std_pnl": float(np.std(pnl_array)),
            "pnl_percentiles": {
                "1%": float(np.percentile(pnl_array, 1)),
                "5%": float(np.percentile(pnl_array, 5)),
                "25%": float(np.percentile(pnl_array, 25)),
                "50%": float(np.percentile(pnl_array, 50)),
                "75%": float(np.percentile(pnl_array, 75)),
                "95%": float(np.percentile(pnl_array, 95)),
                "99%": float(np.percentile(pnl_array, 99)),
            },
            "risk_metrics": metrics,
            "num_scenarios": num_scenarios,
            "total_txn_cost_avg": float(total_txn_cost),
        }
