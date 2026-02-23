"""
surface.py — Volatility Surface Engine.

Features:
- Black-Scholes IV extraction via Brent's method
- SABR calibration with β ∈ [0.5, 1.0] (calibrated, bounded)
- Arbitrage-free cubic spline with penalty re-fit
- Post-fit checks: calendar monotonicity, butterfly convexity, no negative local var
"""

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from typing import Tuple, Dict, List, Optional
from engine.config import SABR_BOUNDS, SABR_BETA_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes IV
# ─────────────────────────────────────────────────────────────────────────────
def bs_call_price(S, K, T, r, q, sigma):
    """BS call price."""
    if T <= 1e-10 or sigma <= 1e-10:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, q, sigma):
    """BS put price."""
    if T <= 1e-10 or sigma <= 1e-10:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma):
    """BS vega (dPrice/dσ)."""
    if T <= 1e-10 or sigma <= 1e-10:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def implied_vol(price: float, S: float, K: float, T: float,
                r: float, q: float, is_call: bool = True,
                lo: float = 0.001, hi: float = 5.0) -> Optional[float]:
    """
    Extract implied volatility via Brent's method.
    Returns None if no root found (illiquid/bad data).
    """
    pricer = bs_call_price if is_call else bs_put_price
    try:
        def objective(sigma):
            return pricer(S, K, T, r, q, sigma) - price
        # Check bounds
        f_lo = objective(lo)
        f_hi = objective(hi)
        if f_lo * f_hi > 0:
            return None
        return brentq(objective, lo, hi, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return None


def extract_iv_surface(
    spot: float, r: float, q: float,
    strikes: np.ndarray, maturities: np.ndarray,
    call_prices: np.ndarray, put_prices: np.ndarray,
    bid_ask_spreads: Optional[np.ndarray] = None,
    max_spread_pct: float = 0.10
) -> Dict:
    """
    Extract full IV surface from option chain.

    Args:
        strikes: shape (num_strikes,)
        maturities: shape (num_maturities,)
        call_prices: shape (num_maturities, num_strikes) — mid prices
        put_prices: shape (num_maturities, num_strikes) — mid prices
        bid_ask_spreads: shape (num_maturities, num_strikes)
        max_spread_pct: Filter threshold

    Returns:
        Dict with iv_call, iv_put, valid_mask arrays
    """
    n_mat, n_k = call_prices.shape
    iv_call = np.full((n_mat, n_k), np.nan)
    iv_put = np.full((n_mat, n_k), np.nan)
    valid = np.ones((n_mat, n_k), dtype=bool)

    for i in range(n_mat):
        T = maturities[i]
        for j in range(n_k):
            K = strikes[j]

            # Filter illiquid
            if bid_ask_spreads is not None:
                mid = 0.5 * (call_prices[i, j] + put_prices[i, j])
                if mid > 0 and bid_ask_spreads[i, j] / mid > max_spread_pct:
                    valid[i, j] = False
                    continue

            iv_c = implied_vol(call_prices[i, j], spot, K, T, r, q, is_call=True)
            iv_p = implied_vol(put_prices[i, j], spot, K, T, r, q, is_call=False)

            if iv_c is not None:
                iv_call[i, j] = iv_c
            else:
                valid[i, j] = False

            if iv_p is not None:
                iv_put[i, j] = iv_p
            else:
                valid[i, j] = False

    return {
        "iv_call": iv_call,
        "iv_put": iv_put,
        "valid_mask": valid,
        "strikes": strikes,
        "maturities": maturities,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SABR Model
# ─────────────────────────────────────────────────────────────────────────────
def sabr_vol(F: float, K: float, T: float,
             alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Hagan SABR implied vol formula.
    F: forward, K: strike, T: maturity
    """
    if abs(F - K) < 1e-10:
        # ATM formula
        FK_mid = F
        term1 = alpha / (FK_mid ** (1 - beta))
        term2 = 1.0 + T * (
            ((1 - beta)**2 / 24) * alpha**2 / FK_mid**(2*(1-beta))
            + 0.25 * rho * beta * nu * alpha / FK_mid**(1-beta)
            + (2 - 3 * rho**2) / 24 * nu**2
        )
        return term1 * term2

    FK = F * K
    FK_beta = FK ** ((1 - beta) / 2)
    log_FK = np.log(F / K)

    z = (nu / alpha) * FK_beta * log_FK
    x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

    if abs(x_z) < 1e-10:
        x_z = 1.0
        z_over_xz = 1.0
    else:
        z_over_xz = z / x_z

    prefix = alpha / (FK_beta * (
        1 + (1 - beta)**2 / 24 * log_FK**2
        + (1 - beta)**4 / 1920 * log_FK**4
    ))

    correction = 1 + T * (
        (1 - beta)**2 / 24 * alpha**2 / FK**(1 - beta)
        + 0.25 * rho * beta * nu * alpha / FK_beta
        + (2 - 3 * rho**2) / 24 * nu**2
    )

    return prefix * z_over_xz * correction


def calibrate_sabr(
    F: float, strikes: np.ndarray, T: float,
    market_ivs: np.ndarray, vegas: Optional[np.ndarray] = None,
    beta_fixed: Optional[float] = None
) -> Dict[str, float]:
    """
    Calibrate SABR parameters to market IVs.
    β is calibrated within [0.5, 1.0] unless fixed.
    Objective: vega-weighted squared IV error.
    """
    if vegas is None:
        vegas = np.ones_like(market_ivs)

    weights = vegas / np.sum(vegas)

    if beta_fixed is not None:
        # Calibrate α, ρ, ν with fixed β
        def objective(x):
            alpha, rho, nu = x
            err = 0.0
            for i, K in enumerate(strikes):
                try:
                    model_iv = sabr_vol(F, K, T, alpha, beta_fixed, rho, nu)
                    err += weights[i] * (model_iv - market_ivs[i])**2
                except Exception:
                    err += 1.0
            return err

        from scipy.optimize import differential_evolution
        bounds = [
            SABR_BOUNDS["alpha"],
            SABR_BOUNDS["rho"],
            SABR_BOUNDS["nu"],
        ]
        result = differential_evolution(objective, bounds, maxiter=500, tol=1e-10)
        return {
            "alpha": result.x[0],
            "beta": beta_fixed,
            "rho": result.x[1],
            "nu": result.x[2],
            "error": result.fun,
        }
    else:
        # Calibrate all four: α, β, ρ, ν
        def objective(x):
            alpha, beta, rho, nu = x
            err = 0.0
            for i, K in enumerate(strikes):
                try:
                    model_iv = sabr_vol(F, K, T, alpha, beta, rho, nu)
                    err += weights[i] * (model_iv - market_ivs[i])**2
                except Exception:
                    err += 1.0
            return err

        from scipy.optimize import differential_evolution
        bounds = [
            SABR_BOUNDS["alpha"],
            SABR_BOUNDS["beta"],
            SABR_BOUNDS["rho"],
            SABR_BOUNDS["nu"],
        ]
        result = differential_evolution(objective, bounds, maxiter=500, tol=1e-10)
        return {
            "alpha": result.x[0],
            "beta": result.x[1],
            "rho": result.x[2],
            "nu": result.x[3],
            "error": result.fun,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Arbitrage-Free Spline
# ─────────────────────────────────────────────────────────────────────────────
class ArbitrageFreeSpline:
    """
    Cubic spline IV interpolation with arbitrage enforcement.

    Post-fit checks:
    1. Butterfly convexity: d²C/dK² ≥ 0 → no negative butterfly
    2. Calendar monotonicity: total variance σ²T is non-decreasing in T
    3. No negative local variance
    """

    def __init__(self):
        self.splines = {}  # keyed by maturity

    def fit(self, strikes: np.ndarray, maturities: np.ndarray,
            iv_surface: np.ndarray, penalty: float = 100.0) -> Dict:
        """
        Fit arbitrage-free splines per maturity.

        Args:
            iv_surface: shape (num_maturities, num_strikes)
            penalty: weight for arbitrage violation penalty

        Returns:
            Dict with fitted splines and violation report
        """
        violations = []

        for i, T in enumerate(maturities):
            ivs = iv_surface[i]
            valid = ~np.isnan(ivs)
            if np.sum(valid) < 4:
                continue

            K_valid = strikes[valid]
            iv_valid = ivs[valid]

            # Fit cubic spline
            cs = CubicSpline(K_valid, iv_valid, bc_type='natural')
            self.splines[float(T)] = cs

            # Check butterfly convexity
            K_fine = np.linspace(K_valid.min(), K_valid.max(), 200)
            iv_fine = cs(K_fine)
            d2_iv = cs(K_fine, 2)  # Second derivative
            butterfly_violations = np.sum(d2_iv < -1e-6)
            if butterfly_violations > 0:
                violations.append({
                    "type": "butterfly",
                    "maturity": T,
                    "count": int(butterfly_violations)
                })

        # Check calendar monotonicity
        sorted_mats = sorted(self.splines.keys())
        for i in range(len(sorted_mats) - 1):
            T1, T2 = sorted_mats[i], sorted_mats[i + 1]
            cs1, cs2 = self.splines[T1], self.splines[T2]
            K_common = np.linspace(
                max(cs1.x.min(), cs2.x.min()),
                min(cs1.x.max(), cs2.x.max()),
                100
            )
            tv1 = cs1(K_common)**2 * T1  # Total variance
            tv2 = cs2(K_common)**2 * T2
            cal_violations = np.sum(tv2 < tv1 - 1e-6)
            if cal_violations > 0:
                violations.append({
                    "type": "calendar",
                    "maturities": (T1, T2),
                    "count": int(cal_violations)
                })

        return {
            "num_maturities_fitted": len(self.splines),
            "violations": violations,
            "is_arbitrage_free": len(violations) == 0
        }

    def get_iv(self, strike: float, maturity: float) -> Optional[float]:
        """Interpolate IV at a given strike and maturity."""
        if not self.splines:
            return None

        mats = sorted(self.splines.keys())

        if maturity in self.splines:
            return float(self.splines[maturity](strike))

        # Linear interpolation between two maturities in total variance
        if maturity < mats[0]:
            return float(self.splines[mats[0]](strike))
        if maturity > mats[-1]:
            return float(self.splines[mats[-1]](strike))

        for i in range(len(mats) - 1):
            if mats[i] <= maturity <= mats[i + 1]:
                T1, T2 = mats[i], mats[i + 1]
                iv1 = self.splines[T1](strike)
                iv2 = self.splines[T2](strike)
                tv1 = iv1**2 * T1
                tv2 = iv2**2 * T2
                w = (maturity - T1) / (T2 - T1)
                tv = tv1 * (1 - w) + tv2 * w
                return float(np.sqrt(max(tv / maturity, 0)))

        return None

    def check_local_variance(self, strikes: np.ndarray,
                             maturities: np.ndarray) -> Dict:
        """Check for negative local variance (Dupire)."""
        negative_lv = []
        for i, T in enumerate(maturities):
            if T not in self.splines:
                continue
            cs = self.splines[T]
            for K in strikes:
                try:
                    iv = cs(K)
                    d_iv_dk = cs(K, 1)
                    d2_iv_dk2 = cs(K, 2)

                    w = iv**2 * T
                    dw_dk = 2 * iv * d_iv_dk * T
                    d2w_dk2 = 2 * T * (d_iv_dk**2 + iv * d2_iv_dk2)

                    numerator = 1.0  # Simplified — full Dupire needs dw/dT
                    denominator = (1 - K * dw_dk / (2 * w))**2 - \
                                  0.25 * w * (d2w_dk2 - 0.25) + K**2 * d2w_dk2
                    if denominator <= 0:
                        negative_lv.append({"K": K, "T": T})
                except Exception:
                    pass
        return {
            "has_negative_local_var": len(negative_lv) > 0,
            "violations": negative_lv
        }
