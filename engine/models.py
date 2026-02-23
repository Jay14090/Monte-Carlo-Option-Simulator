"""
models.py — SVJ Model with Term Structure and Forward-Space Formulation.

Implements:
- Maturity-dependent parameters: θ(T), ξ(T), λ(T)
- Forward-space calibration: F = S₀·e^{(r-q)T}
- Forward variance initialization from ATM IV at shortest maturity
- Full truncation scheme for variance positivity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from engine.config import (
    RISK_FREE_RATE, DIVIDEND_YIELD, check_feller,
    PARAM_BOUNDS, TERM_STRUCTURE_BOUNDS, MAX_VARIANCE
)


@dataclass
class SVJParams:
    """
    SVJ model parameters — single maturity slice.

    Spot:  dS = (r - q - λk)S dt + √v S dW₁ + S(e^J - 1)dN
    Var:   dv = κ(θ - v)dt + ξ√v dW₂
    Corr:  dW₁·dW₂ = ρ dt
    Jump:  J ~ N(μ_J, σ_J²),  k = E[e^J - 1]
    """
    # Heston core
    kappa: float = 3.0       # Mean reversion speed
    theta: float = 0.04      # Long-run variance
    xi: float = 0.5          # Vol-of-vol
    rho: float = -0.7        # Spot-vol correlation
    v0: float = 0.04         # Initial variance

    # Jump component
    lambda_j: float = 1.0    # Jump intensity (events/year)
    mu_j: float = -0.05      # Mean jump size (log)
    sigma_j: float = 0.10    # Jump size std

    # Market
    r: float = RISK_FREE_RATE
    q: float = DIVIDEND_YIELD

    @property
    def jump_compensation(self) -> float:
        """k = E[e^J - 1] for drift compensation."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1.0

    @property
    def feller_satisfied(self) -> bool:
        return check_feller(self.kappa, self.theta, self.xi)

    def to_array(self) -> np.ndarray:
        """Flatten to optimizer-friendly array."""
        return np.array([
            self.kappa, self.theta, self.xi, self.rho, self.v0,
            self.lambda_j, self.mu_j, self.sigma_j
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray, r: float = RISK_FREE_RATE,
                   q: float = DIVIDEND_YIELD) -> 'SVJParams':
        """Reconstruct from optimizer array."""
        return cls(
            kappa=arr[0], theta=arr[1], xi=arr[2], rho=arr[3], v0=arr[4],
            lambda_j=arr[5], mu_j=arr[6], sigma_j=arr[7], r=r, q=q
        )

    def validate(self) -> List[str]:
        """Return list of validation warnings."""
        warnings = []
        if not self.feller_satisfied:
            warnings.append(
                f"Feller violated: 2κθ={2*self.kappa*self.theta:.4f} "
                f"≤ ξ²={self.xi**2:.4f}"
            )
        if abs(self.rho) > 0.999:
            warnings.append(f"|ρ|={abs(self.rho):.4f} exceeds 0.999")
        if self.v0 > MAX_VARIANCE:
            warnings.append(f"v0={self.v0:.4f} exceeds MAX_VARIANCE={MAX_VARIANCE}")
        if self.theta > MAX_VARIANCE:
            warnings.append(f"θ={self.theta:.4f} exceeds MAX_VARIANCE={MAX_VARIANCE}")
        return warnings


@dataclass
class TermStructureSVJ:
    """
    Maturity-dependent SVJ parameters.

    Some parameters vary across the term structure:
    - θ(T): long-run variance (higher for weeklies)
    - ξ(T): vol-of-vol (accelerated near expiry)
    - λ(T): jump intensity (higher around events)

    Fixed across maturities: κ, ρ, μ_J, σ_J
    """
    # Fixed parameters
    kappa: float = 3.0
    rho: float = -0.7
    mu_j: float = -0.05
    sigma_j: float = 0.10
    v0: float = 0.04
    r: float = RISK_FREE_RATE
    q: float = DIVIDEND_YIELD

    # Maturity-dependent: keyed by T (years)
    theta_curve: Dict[float, float] = field(default_factory=dict)
    xi_curve: Dict[float, float] = field(default_factory=dict)
    lambda_curve: Dict[float, float] = field(default_factory=dict)

    def get_params_at_maturity(self, T: float) -> SVJParams:
        """Interpolate parameters to a specific maturity."""
        theta = self._interp(self.theta_curve, T, default=0.04)
        xi = self._interp(self.xi_curve, T, default=0.5)
        lambda_j = self._interp(self.lambda_curve, T, default=1.0)

        return SVJParams(
            kappa=self.kappa, theta=theta, xi=xi, rho=self.rho,
            v0=self.v0, lambda_j=lambda_j, mu_j=self.mu_j,
            sigma_j=self.sigma_j, r=self.r, q=self.q
        )

    @staticmethod
    def _interp(curve: Dict[float, float], T: float, default: float) -> float:
        """Piecewise-linear interpolation on the term structure curve."""
        if not curve:
            return default
        mats = sorted(curve.keys())
        vals = [curve[m] for m in mats]

        if T <= mats[0]:
            return vals[0]
        if T >= mats[-1]:
            return vals[-1]

        # Find bracketing maturities
        for i in range(len(mats) - 1):
            if mats[i] <= T <= mats[i + 1]:
                w = (T - mats[i]) / (mats[i + 1] - mats[i])
                return vals[i] * (1 - w) + vals[i + 1] * w
        return default


def forward_price(spot: float, r: float, q: float, T: float) -> float:
    """Compute forward price: F = S₀·e^{(r-q)T}."""
    return spot * np.exp((r - q) * T)


def extract_forward_variance(atm_iv: float, T_shortest: float) -> float:
    """
    Extract initial variance from ATM implied volatility at shortest maturity.
    v₀ ≈ σ²_ATM(T_min)
    This ensures immediate surface consistency.
    """
    return atm_iv ** 2


def build_term_structure_from_surface(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
    skew_slopes: np.ndarray,
    base_params: SVJParams
) -> TermStructureSVJ:
    """
    Bootstrap a TermStructureSVJ from observed surface data.

    Uses heuristics:
    - θ(T) ≈ ATM_IV(T)² (variance target follows ATM term structure)
    - ξ(T) scaled by 1/√T (vol-of-vol accelerates near expiry)
    - λ(T) scaled by |skew_slope| (jumpier when skew is steeper)
    """
    ts = TermStructureSVJ(
        kappa=base_params.kappa, rho=base_params.rho,
        mu_j=base_params.mu_j, sigma_j=base_params.sigma_j,
        v0=extract_forward_variance(atm_ivs[0], maturities[0]),
        r=base_params.r, q=base_params.q
    )

    for i, T in enumerate(maturities):
        # θ(T): long-run variance follows ATM IV term structure
        ts.theta_curve[float(T)] = float(atm_ivs[i] ** 2)

        # ξ(T): vol-of-vol — accelerated for short maturities
        xi_scale = min(3.0, 1.0 / np.sqrt(max(T, 1 / 252)))
        ts.xi_curve[float(T)] = float(base_params.xi * xi_scale)

        # λ(T): jump intensity — higher when skew is steep
        skew_scale = max(1.0, abs(skew_slopes[i]) / 0.03)
        ts.lambda_curve[float(T)] = float(base_params.lambda_j * skew_scale)

    return ts
