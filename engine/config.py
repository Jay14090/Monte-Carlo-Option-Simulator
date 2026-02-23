"""
config.py — Constants, parameter bounds, Feller condition checks, regime thresholds.

This module defines the entire configuration space for the NIFTY SVJ Monte Carlo engine.
All bounds are designed for the Indian index options market.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Market Constants (NIFTY / Indian Market)
# ─────────────────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.065          # ~6.5% RBI repo-linked
DIVIDEND_YIELD = 0.012          # ~1.2% NIFTY dividend yield
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_DAY = 375           # NSE trading session length

# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_NUM_PATHS = 500_000
DEFAULT_NUM_STEPS = 252
DEFAULT_TOLERANCE = 0.001       # 0.1% of premium
MAX_PATHS = 2_000_000

# ─────────────────────────────────────────────────────────────────────────────
# SVJ Parameter Bounds (for L-BFGS-B)
# Format: (lower, upper)
# ─────────────────────────────────────────────────────────────────────────────
PARAM_BOUNDS = {
    # Heston core
    "kappa":   (0.1,   15.0),    # Mean reversion speed
    "theta":   (0.005, 1.50),    # Long-run variance
    "xi":      (0.05,  3.00),    # Vol-of-vol
    "rho":     (-0.999, 0.0),    # Spot-vol correlation (negative for equity)
    "v0":      (0.005, 1.50),    # Initial variance

    # Jump parameters
    "lambda_j": (0.0,  10.0),   # Jump intensity
    "mu_j":     (-0.20, 0.05),  # Mean jump size (log)
    "sigma_j":  (0.01, 0.50),   # Jump size volatility
}

# Term-structure specific bounds (for maturity-dependent params)
TERM_STRUCTURE_BOUNDS = {
    "theta_T":   (0.005, 2.00),
    "xi_T":      (0.05,  5.00),  # Higher ceiling for weekly expiry vol-of-vol
    "lambda_T":  (0.0,   20.0),  # Higher jumps allowed for event weeks
}

# ─────────────────────────────────────────────────────────────────────────────
# Regularization Weights (Tikhonov)
# ─────────────────────────────────────────────────────────────────────────────
REGULARIZATION = {
    "xi":       0.01,   # Penalize extreme vol-of-vol
    "rho":      0.005,  # Penalize extreme correlation
    "lambda_j": 0.01,   # Penalize extreme jump intensity
}

# ─────────────────────────────────────────────────────────────────────────────
# SABR Bounds
# ─────────────────────────────────────────────────────────────────────────────
SABR_BOUNDS = {
    "alpha": (0.01, 5.0),
    "beta":  (0.5,  1.0),   # Calibrated, not fixed — range [0.5, 1.0]
    "rho":   (-0.999, 0.999),
    "nu":    (0.01, 5.0),
}
SABR_BETA_DEFAULT = 0.8   # Better skew control for NIFTY than β=1

# ─────────────────────────────────────────────────────────────────────────────
# Stability Guards
# ─────────────────────────────────────────────────────────────────────────────
MAX_VARIANCE = 10.0              # Reject if variance exceeds this
MAX_CORRELATION = 0.999          # |ρ| ceiling
VOL_OF_VOL_ALERT_THRESHOLD = 4.0 # Alert if ξ exceeds this
SAFE_STRIKE_RANGE = (0.70, 1.30) # Moneyness range for safe surface use
JUMP_COMPENSATION_TOL = 1e-6     # Drift compensation tolerance

# ─────────────────────────────────────────────────────────────────────────────
# Regime Detection Thresholds
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RegimeThresholds:
    """Thresholds for Calm / Event / Crisis classification."""
    # Realized vol (annualized)
    calm_rvol_upper: float = 0.15
    event_rvol_upper: float = 0.30
    # crisis = above event_rvol_upper

    # IV percentile rank (0–100)
    calm_iv_pctile_upper: float = 30.0
    event_iv_pctile_upper: float = 70.0

    # Put-call skew slope (25d put IV - 25d call IV)
    calm_skew_upper: float = 0.03
    event_skew_upper: float = 0.08

REGIME_THRESHOLDS = RegimeThresholds()

# ─────────────────────────────────────────────────────────────────────────────
# Calibration Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CalibrationConfig:
    """Two-stage calibration settings."""
    # Stage 1: Heston core (ATM + near-money)
    stage1_moneyness_range: Tuple[float, float] = (0.95, 1.05)
    stage1_max_iter: int = 200

    # Stage 2: Add jumps (full strike range)
    stage2_moneyness_range: Tuple[float, float] = (0.80, 1.20)
    stage2_max_iter: int = 300

    # Optimizer
    optimizer: str = "L-BFGS-B"
    ftol: float = 1e-12
    gtol: float = 1e-8

    # Liquidity filtering
    min_open_interest: int = 100
    max_bid_ask_spread_pct: float = 0.10  # 10% of mid

    # Recalibration interval (seconds)
    recalib_interval: int = 300  # 5 minutes

CALIBRATION_CONFIG = CalibrationConfig()

# ─────────────────────────────────────────────────────────────────────────────
# Stress Test Scenarios
# ─────────────────────────────────────────────────────────────────────────────
SPOT_SHOCKS = [-0.08, -0.05, -0.02, 0.02, 0.05, 0.08]
VOL_SHOCKS  = [-0.05, 0.05]   # ±5 vol points
JUMP_SCENARIO_SIZE = 0.04     # 4% overnight gap

# ─────────────────────────────────────────────────────────────────────────────
# Validation Helpers
# ─────────────────────────────────────────────────────────────────────────────
def check_feller(kappa: float, theta: float, xi: float) -> bool:
    """Check Feller condition: 2κθ > ξ² ensures variance stays positive."""
    return 2.0 * kappa * theta > xi * xi


def check_params_in_bounds(params: Dict[str, float]) -> Dict[str, bool]:
    """Verify all parameters are within their defined bounds."""
    results = {}
    for name, value in params.items():
        if name in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[name]
            results[name] = lo <= value <= hi
    return results


def clamp_params(params: Dict[str, float]) -> Dict[str, float]:
    """Clamp parameters to their bounds."""
    clamped = {}
    for name, value in params.items():
        if name in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[name]
            clamped[name] = np.clip(value, lo, hi)
        else:
            clamped[name] = value
    return clamped
