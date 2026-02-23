"""
regime.py — Market Regime Detection for NIFTY.

Classifies the market into three regimes:
- CALM: Low vol, flat skew, normal conditions
- EVENT: Elevated vol, steeper skew (RBI, budget, earnings)
- CRISIS: High vol, extreme skew, tail risk active

Switches calibration constraints per regime.
"""

import numpy as np
from enum import Enum
from typing import Dict, Optional
from engine.config import REGIME_THRESHOLDS, PARAM_BOUNDS
from engine.models import SVJParams


class MarketRegime(Enum):
    CALM = "calm"
    EVENT = "event"
    CRISIS = "crisis"


class RegimeDetector:
    """
    Detect market regime from observable market data.

    Inputs:
    - Realized volatility (annualized, rolling window)
    - IV percentile rank (0–100, relative to historical distribution)
    - Put-call skew slope (25Δ put IV − 25Δ call IV)
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or REGIME_THRESHOLDS
        self.history = []

    def classify(self, realized_vol: float, iv_percentile: float,
                 skew_slope: float) -> Dict:
        """
        Classify current regime.

        Returns:
            Dict with regime, scores, and calibration constraint adjustments.
        """
        th = self.thresholds

        # Score each indicator (0 = calm, 1 = event, 2 = crisis)
        vol_score = 0
        if realized_vol > th.event_rvol_upper:
            vol_score = 2
        elif realized_vol > th.calm_rvol_upper:
            vol_score = 1

        iv_score = 0
        if iv_percentile > th.event_iv_pctile_upper:
            iv_score = 2
        elif iv_percentile > th.calm_iv_pctile_upper:
            iv_score = 1

        skew_score = 0
        if abs(skew_slope) > th.event_skew_upper:
            skew_score = 2
        elif abs(skew_slope) > th.calm_skew_upper:
            skew_score = 1

        # Weighted average score
        total_score = 0.40 * vol_score + 0.35 * iv_score + 0.25 * skew_score

        if total_score >= 1.5:
            regime = MarketRegime.CRISIS
        elif total_score >= 0.7:
            regime = MarketRegime.EVENT
        else:
            regime = MarketRegime.CALM

        result = {
            "regime": regime.value,
            "score": float(total_score),
            "vol_score": vol_score,
            "iv_score": iv_score,
            "skew_score": skew_score,
            "inputs": {
                "realized_vol": realized_vol,
                "iv_percentile": iv_percentile,
                "skew_slope": skew_slope,
            },
            "calibration_adjustments": self._get_adjustments(regime),
        }

        self.history.append(result)
        return result

    def _get_adjustments(self, regime: MarketRegime) -> Dict:
        """
        Return calibration constraint adjustments per regime.

        CALM: Tighter bounds, less jump intensity allowed
        EVENT: Wider vol-of-vol, higher jump intensity
        CRISIS: Widest bounds, emergency constraints
        """
        if regime == MarketRegime.CALM:
            return {
                "xi_bounds": (0.05, 1.5),
                "lambda_bounds": (0.0, 3.0),
                "rho_bounds": (-0.95, -0.1),
                "regularization_scale": 1.5,  # Heavier reg in calm
                "description": "Tight constraints — low vol environment",
            }
        elif regime == MarketRegime.EVENT:
            return {
                "xi_bounds": (0.1, 3.0),
                "lambda_bounds": (0.5, 10.0),
                "rho_bounds": (-0.999, 0.0),
                "regularization_scale": 1.0,
                "description": "Relaxed constraints — event-driven vol",
            }
        else:  # CRISIS
            return {
                "xi_bounds": (0.2, 5.0),
                "lambda_bounds": (1.0, 20.0),
                "rho_bounds": (-0.999, 0.0),
                "regularization_scale": 0.5,  # Lighter reg — let model fit extremes
                "description": "Emergency constraints — crisis regime",
            }

    def get_regime_history(self) -> list:
        """Return timeline of regime classifications."""
        return self.history


def compute_realized_vol(prices: np.ndarray, window: int = 20,
                         annualize: int = 252) -> float:
    """
    Compute annualized realized volatility from price series.

    Args:
        prices: Array of daily closing prices
        window: Rolling window (default 20 days = ~1 month)
        annualize: Trading days per year
    """
    if len(prices) < window + 1:
        returns = np.diff(np.log(prices))
    else:
        returns = np.diff(np.log(prices[-window-1:]))

    return float(np.std(returns) * np.sqrt(annualize))


def compute_iv_percentile(current_iv: float,
                          historical_ivs: np.ndarray) -> float:
    """
    Compute IV percentile rank (0–100).

    Where does current IV sit relative to its historical distribution?
    """
    if len(historical_ivs) == 0:
        return 50.0
    return float(np.sum(historical_ivs <= current_iv) / len(historical_ivs) * 100)


def compute_skew_slope(put_25d_iv: float, call_25d_iv: float) -> float:
    """Compute 25-delta put-call skew slope."""
    return put_25d_iv - call_25d_iv
