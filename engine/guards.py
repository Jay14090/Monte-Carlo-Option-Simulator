"""
guards.py — Production Stability Guards.

Pre-price and post-price validation to reject unreliable runs.

Checks:
- Variance explosion
- Correlation bounds
- Jump compensation alignment
- Surface extrapolation beyond safe strikes
- Vol-of-vol spike alert
- No arbitrage violations
- No negative variance in simulation
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from engine.models import SVJParams
from engine.config import (
    MAX_VARIANCE, MAX_CORRELATION, VOL_OF_VOL_ALERT_THRESHOLD,
    SAFE_STRIKE_RANGE, JUMP_COMPENSATION_TOL
)

logger = logging.getLogger("guards")


class PricingGuard:
    """
    Production stability guard for Monte Carlo pricing.

    Call check_pre_price() before pricing.
    Call check_post_price() after pricing.
    If either fails, reject the run.
    """

    def __init__(self, params: SVJParams):
        self.params = params
        self.alerts = []

    def check_pre_price(self, spot: float, strike: float, T: float) -> Dict:
        """
        Pre-pricing validation.

        Returns:
            Dict with 'pass': bool, 'failures': list, 'alerts': list
        """
        failures = []
        alerts = []

        p = self.params

        # 1. Variance checks
        if p.v0 > MAX_VARIANCE:
            failures.append(f"v0={p.v0:.4f} exceeds MAX_VARIANCE={MAX_VARIANCE}")
        if p.v0 <= 0:
            failures.append(f"v0={p.v0:.6f} is non-positive")
        if p.theta > MAX_VARIANCE:
            failures.append(f"θ={p.theta:.4f} exceeds MAX_VARIANCE={MAX_VARIANCE}")
        if p.theta <= 0:
            failures.append(f"θ={p.theta:.6f} is non-positive")

        # 2. Correlation bound
        if abs(p.rho) > MAX_CORRELATION:
            failures.append(f"|ρ|={abs(p.rho):.4f} exceeds {MAX_CORRELATION}")

        # 3. Jump compensation check
        k = p.jump_compensation
        expected_k = np.exp(p.mu_j + 0.5 * p.sigma_j**2) - 1.0
        if abs(k - expected_k) > JUMP_COMPENSATION_TOL:
            failures.append(
                f"Jump compensation misaligned: k={k:.6f} vs expected={expected_k:.6f}"
            )

        # 4. Surface extrapolation check
        if spot > 0:
            moneyness = strike / spot
            lo, hi = SAFE_STRIKE_RANGE
            if moneyness < lo or moneyness > hi:
                alerts.append(
                    f"Moneyness={moneyness:.3f} outside safe range [{lo}, {hi}]. "
                    "Surface extrapolation may be unreliable."
                )

        # 5. Vol-of-vol alert
        if p.xi > VOL_OF_VOL_ALERT_THRESHOLD:
            alerts.append(
                f"ξ={p.xi:.3f} exceeds alert threshold={VOL_OF_VOL_ALERT_THRESHOLD}. "
                "Model may be unstable."
            )

        # 6. Feller condition
        if not p.feller_satisfied:
            alerts.append(
                f"Feller condition violated: 2κθ={2*p.kappa*p.theta:.4f} ≤ ξ²={p.xi**2:.4f}. "
                "Variance may hit zero frequently."
            )

        # 7. Maturity check
        if T <= 0:
            failures.append(f"T={T} is non-positive")
        if T > 5:
            alerts.append(f"T={T:.2f} years — very long maturity, model may be less reliable")

        self.alerts.extend(alerts)
        for f in failures:
            logger.error("PRE-PRICE FAILURE: %s", f)
        for a in alerts:
            logger.warning("PRE-PRICE ALERT: %s", a)

        return {
            "pass": len(failures) == 0,
            "failures": failures,
            "alerts": alerts,
        }

    def check_post_price(self, result: Dict, spot: float, strike: float,
                         T: float, is_call: bool = True) -> Dict:
        """
        Post-pricing validation.

        Args:
            result: Output from MonteCarloEngine.price()

        Returns:
            Dict with 'pass': bool, 'failures': list, 'alerts': list
        """
        failures = []
        alerts = []

        price = result.get("price", 0)
        std_error = result.get("std_error", 0)

        # 1. Price must be non-negative
        if price < -1e-6:
            failures.append(f"Negative price={price:.6f}")

        # 2. Std error tolerance check
        if price > 0 and std_error / price > 0.001:
            alerts.append(
                f"Std error ratio={std_error/price:.4f} exceeds 0.1% tolerance"
            )

        # 3. Price should not exceed spot (for calls) or strike (for puts)
        if is_call and price > spot * 1.01:
            failures.append(f"Call price={price:.2f} exceeds spot={spot:.2f}")
        if not is_call and price > strike * np.exp(-self.params.r * T) * 1.01:
            failures.append(f"Put price={price:.2f} exceeds discounted strike")

        # 4. Intrinsic value check — option should be worth at least intrinsic
        if is_call:
            intrinsic = max(spot * np.exp(-self.params.q * T) - strike * np.exp(-self.params.r * T), 0)
        else:
            intrinsic = max(strike * np.exp(-self.params.r * T) - spot * np.exp(-self.params.q * T), 0)

        if price < intrinsic - std_error * 3:
            failures.append(
                f"Price={price:.4f} below intrinsic={intrinsic:.4f} by more than 3σ"
            )

        for f in failures:
            logger.error("POST-PRICE FAILURE: %s", f)
        for a in alerts:
            logger.warning("POST-PRICE ALERT: %s", a)

        return {
            "pass": len(failures) == 0,
            "failures": failures,
            "alerts": alerts,
        }


def validate_simulation_output(S_final: np.ndarray, v_final: np.ndarray) -> Dict:
    """
    Validate simulation output arrays.
    """
    issues = []

    # Check for NaN/Inf
    nan_S = np.sum(np.isnan(S_final))
    nan_v = np.sum(np.isnan(v_final))
    inf_S = np.sum(np.isinf(S_final))
    inf_v = np.sum(np.isinf(v_final))

    if nan_S > 0:
        issues.append(f"{nan_S} NaN values in S_final")
    if nan_v > 0:
        issues.append(f"{nan_v} NaN values in v_final")
    if inf_S > 0:
        issues.append(f"{inf_S} Inf values in S_final")
    if inf_v > 0:
        issues.append(f"{inf_v} Inf values in v_final")

    # Check for negative spots
    neg_S = np.sum(S_final < 0)
    if neg_S > 0:
        issues.append(f"{neg_S} negative S values")

    # Check for variance explosion
    max_v = np.max(v_final) if len(v_final) > 0 else 0
    if max_v > MAX_VARIANCE:
        issues.append(f"Max variance={max_v:.4f} exceeds limit={MAX_VARIANCE}")

    # Check for negative variance (should be caught by full truncation, but verify)
    neg_v = np.sum(v_final < -1e-10)
    if neg_v > 0:
        issues.append(f"{neg_v} negative variance values (truncation failed)")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "stats": {
            "S_mean": float(np.nanmean(S_final)),
            "S_std": float(np.nanstd(S_final)),
            "v_mean": float(np.nanmean(v_final)),
            "v_max": float(max_v),
        }
    }
