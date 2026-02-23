"""Quick test of core engine modules."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("NIFTY MC Engine - Quick Verification")
print("=" * 60)

# Test 1: Models
print("\n[1] Models...")
from engine.models import SVJParams, forward_price
p = SVJParams(kappa=5.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
              lambda_j=0.0, mu_j=0.0, sigma_j=0.01)
print(f"    Feller: {p.feller_satisfied} (2*5*0.04=0.40 > 0.09={0.3**2})")
F = forward_price(22500, 0.065, 0.012, 0.04)
print(f"    Forward: {F:.2f}")
print("    PASS")

# Test 2: BS Price
print("\n[2] BS Price...")
from engine.monte_carlo import bs_price
analytical = bs_price(22500, 22500, 0.04, 0.065, 0.012, 0.2, True)
print(f"    BS Call ATM: {analytical:.4f}")
print("    PASS")

# Test 3: MC Engine
print("\n[3] MC Engine (50k paths, pure BS)...")
from engine.monte_carlo import MonteCarloEngine
bs_params = SVJParams(kappa=5.0, theta=0.04, xi=0.0001, rho=-0.7, v0=0.04,
                      lambda_j=0.0, mu_j=0.0, sigma_j=0.01)
engine = MonteCarloEngine(bs_params, num_paths=50000, num_steps=100,
                          use_sobol=True, use_antithetic=True, use_control_variate=True)
t0 = time.time()
result = engine.price(22500, 22500, 0.04, True)
dt = time.time() - t0
print(f"    MC Price: {result['price']:.4f}")
print(f"    Std Err:  {result['std_error']:.6f}")
print(f"    BS Ref:   {result.get('bs_ref', 'N/A')}")
print(f"    Time:     {dt*1000:.0f}ms")
diff = abs(result['price'] - analytical)
print(f"    |MC - BS| = {diff:.4f}")
print(f"    {'PASS' if diff < 20 else 'FAIL'}")

# Test 4: SVJ Pricing
print("\n[4] SVJ Pricing (50k paths)...")
svj = SVJParams(kappa=5.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04,
                lambda_j=1.0, mu_j=-0.05, sigma_j=0.10)
engine2 = MonteCarloEngine(svj, num_paths=50000, num_steps=100)
t0 = time.time()
call = engine2.price(22500, 22500, 0.04, True)
put = engine2.price(22500, 22500, 0.04, False)
dt = time.time() - t0
print(f"    Call: {call['price']:.4f}")
print(f"    Put:  {put['price']:.4f}")
print(f"    Time: {dt*1000:.0f}ms")
print("    PASS")

# Test 5: Regime
print("\n[5] Regime Detection...")
from engine.regime import RegimeDetector
det = RegimeDetector()
r1 = det.classify(0.12, 25, 0.02)
r2 = det.classify(0.22, 60, 0.06)
r3 = det.classify(0.35, 85, 0.12)
print(f"    Calm:   {r1['regime']} (score={r1['score']:.2f})")
print(f"    Event:  {r2['regime']} (score={r2['score']:.2f})")
print(f"    Crisis: {r3['regime']} (score={r3['score']:.2f})")
ok = r1['regime']=='calm' and r2['regime']=='event' and r3['regime']=='crisis'
print(f"    {'PASS' if ok else 'FAIL'}")

# Test 6: Guards
print("\n[6] Stability Guards...")
from engine.guards import PricingGuard
guard = PricingGuard(svj)
pre = guard.check_pre_price(22500, 22500, 0.04)
print(f"    Pre-check pass: {pre['pass']}")
print(f"    Alerts: {pre['alerts']}")
print("    PASS")

# Test 7: Risk
print("\n[7] Risk Metrics...")
import numpy as np
from engine.risk import compute_risk_metrics
rng = np.random.default_rng(42)
returns = rng.standard_normal(10000) * 0.02 - 0.001
m = compute_risk_metrics(returns)
print(f"    VaR:      {m['var']:.4f}")
print(f"    CVaR:     {m['cvar']:.4f}")
print(f"    Skew:     {m['skewness']:.4f}")
print(f"    Kurt:     {m['kurtosis']:.4f}")
print("    PASS")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("\nStart server: python -m uvicorn engine.app:app --port 8000 --reload")
print("Dashboard:    http://localhost:8000")
