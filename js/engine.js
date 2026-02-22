/**
 * Monte Carlo Option Pricing Engine — India Edition
 * All math runs in the browser using typed arrays for performance.
 * Uses Box-Muller transform for normal random numbers.
 */

'use strict';

// ─── Random Number Generation ─────────────────────────────────────────────────
/**
 * Box-Muller transform: generates standard normal random number.
 */
function randn() {
    let u, v;
    do { u = Math.random(); } while (u === 0);
    v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Generate a Float64Array of n standard normal samples.
 * Pre-generating all values is faster than calling randn() inline.
 */
function randnArray(n) {
    const arr = new Float64Array(n);
    for (let i = 0; i < n; i++) arr[i] = randn();
    return arr;
}

// ─── Geometric Brownian Motion ────────────────────────────────────────────────
/**
 * Simulate N stock price paths using Geometric Brownian Motion.
 *
 * @param {number} S0      - Initial stock price (₹)
 * @param {number} r       - Risk-free rate (RBI Repo, decimal e.g. 0.0625)
 * @param {number} sigma   - Annualised volatility (decimal e.g. 0.25)
 * @param {number} T       - Time to maturity (years, e.g. 0.25 = 3 months)
 * @param {number} steps   - Number of time steps (245 for NSE trading days)
 * @param {number} nPaths  - Number of simulation paths
 * @returns {{ t: Float64Array, paths: Float64Array[] }}
 */
function simulateGBM(S0, r, sigma, T, steps, nPaths) {
    const dt = T / steps;
    const drift = (r - 0.5 * sigma * sigma) * dt;
    const vol = sigma * Math.sqrt(dt);

    // Time axis
    const t = new Float64Array(steps + 1);
    for (let i = 0; i <= steps; i++) t[i] = (i / steps) * T;

    // Paths matrix: flat array of nPaths × (steps+1)
    const Z = randnArray(nPaths * steps);
    const paths = [];

    for (let p = 0; p < nPaths; p++) {
        const path = new Float64Array(steps + 1);
        path[0] = S0;
        for (let i = 1; i <= steps; i++) {
            path[i] = path[i - 1] * Math.exp(drift + vol * Z[p * steps + (i - 1)]);
        }
        paths.push(path);
    }

    return { t, paths };
}

// ─── Monte Carlo Pricing ──────────────────────────────────────────────────────
/**
 * Price a European option via Monte Carlo.
 *
 * @param {Float64Array[]} paths     - Array of price paths
 * @param {number} K                 - Strike price (₹)
 * @param {number} r                 - Risk-free rate
 * @param {number} T                 - Time to maturity
 * @param {'call'|'put'} optionType
 * @returns {number} Option price in ₹
 */
function monteCarloPricing(paths, K, r, T, optionType = 'call') {
    const n = paths.length;
    let payoffSum = 0;

    if (optionType === 'call') {
        for (let i = 0; i < n; i++) {
            const ST = paths[i][paths[i].length - 1];
            payoffSum += Math.max(ST - K, 0);
        }
    } else {
        for (let i = 0; i < n; i++) {
            const ST = paths[i][paths[i].length - 1];
            payoffSum += Math.max(K - ST, 0);
        }
    }

    return Math.exp(-r * T) * (payoffSum / n);
}

// ─── Black-Scholes Analytical Pricer ─────────────────────────────────────────
/**
 * Standard normal CDF using Horner's method (accurate to 7 decimal places).
 */
function normCDF(x) {
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
    const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.SQRT2;
    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return 0.5 * (1 + sign * y);
}

/**
 * Black-Scholes closed-form price for European options.
 *
 * @param {number} S0    - Current stock price
 * @param {number} K     - Strike price
 * @param {number} r     - Risk-free rate (RBI Repo)
 * @param {number} sigma - Volatility
 * @param {number} T     - Time to maturity (years)
 * @param {'call'|'put'} optionType
 * @returns {number} Option price in ₹
 */
function blackScholes(S0, K, r, sigma, T, optionType = 'call') {
    if (T <= 0) return Math.max(optionType === 'call' ? S0 - K : K - S0, 0);
    const sqrtT = Math.sqrt(T);
    const d1 = (Math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const d2 = d1 - sigma * sqrtT;

    if (optionType === 'call') {
        return S0 * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2);
    } else {
        return K * Math.exp(-r * T) * normCDF(-d2) - S0 * normCDF(-d1);
    }
}

/**
 * Compute d1 and d2 for Greeks calculation.
 */
function getD1D2(S0, K, r, sigma, T) {
    const sqrtT = Math.sqrt(T);
    const d1 = (Math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const d2 = d1 - sigma * sqrtT;
    return { d1, d2 };
}

// ─── Option Greeks ────────────────────────────────────────────────────────────
/**
 * Standard normal PDF.
 */
function normPDF(x) {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Calculate all five Greeks analytically.
 *
 * @param {number} S0
 * @param {number} K
 * @param {number} r
 * @param {number} sigma
 * @param {number} T
 * @param {'call'|'put'} optionType
 * @returns {{ delta, gamma, vega, theta, rho }}
 */
function calculateGreeks(S0, K, r, sigma, T, optionType = 'call') {
    if (T <= 0.001) return { delta: 0, gamma: 0, vega: 0, theta: 0, rho: 0 };

    const { d1, d2 } = getD1D2(S0, K, r, sigma, T);
    const sqrtT = Math.sqrt(T);
    const expRT = Math.exp(-r * T);
    const pdf1 = normPDF(d1);

    let delta, theta, rho;

    if (optionType === 'call') {
        delta = normCDF(d1);
        theta = ((-(S0 * pdf1 * sigma) / (2 * sqrtT) - r * K * expRT * normCDF(d2)) / (state.yearBasis || 365));
        rho = K * T * expRT * normCDF(d2) / 100;
    } else {
        delta = normCDF(d1) - 1;
        theta = ((-(S0 * pdf1 * sigma) / (2 * sqrtT) + r * K * expRT * normCDF(-d2)) / (state.yearBasis || 365));
        rho = -K * T * expRT * normCDF(-d2) / 100;
    }

    const gamma = pdf1 / (S0 * sigma * sqrtT);
    const vega = S0 * pdf1 * sqrtT / 100;

    return { delta, gamma, vega, theta, rho };
}

// ─── Convergence Series ───────────────────────────────────────────────────────
/**
 * Compute MC price at increasing path counts to show convergence.
 * Returns arrays of [pathCounts, prices] for the chart.
 */
function computeConvergence(S0, K, r, sigma, T, optionType, targetPaths = 10000, yearBasis = 250) {
    const steps = yearBasis;
    const checkpoints = [50, 100, 200, 500, 1000, 2000, 5000, targetPaths];
    const validCheckpoints = checkpoints.filter(n => n <= targetPaths);

    const bsPrice = blackScholes(S0, K, r, sigma, T, optionType);
    const { t, paths } = simulateGBM(S0, r, sigma, T, steps, targetPaths);

    const pathCounts = [];
    const mcPrices = [];

    for (const n of validCheckpoints) {
        const slice = paths.slice(0, n);
        const price = monteCarloPricing(slice, K, r, T, optionType);
        pathCounts.push(n);
        mcPrices.push(price);
    }

    return { pathCounts, mcPrices, bsPrice };
}

// ─── Sensitivity Analysis ─────────────────────────────────────────────────────
/**
 * Sweep across volatility range and compute BS prices.
 * Returns data for a sensitivity chart.
 */
function sensitivityVolatility(S0, K, r, T, optionType, steps = 30) {
    const vols = [], callPrices = [], putPrices = [];
    for (let i = 0; i <= steps; i++) {
        const sigma = 0.05 + (i / steps) * 0.75;  // 5% to 80%
        vols.push(+(sigma * 100).toFixed(1));
        callPrices.push(+blackScholes(S0, K, r, sigma, T, 'call').toFixed(2));
        putPrices.push(+blackScholes(S0, K, r, sigma, T, 'put').toFixed(2));
    }
    return { vols, callPrices, putPrices };
}

/**
 * Sweep across strike price range.
 */
function sensitivityStrike(S0, r, sigma, T, optionType, steps = 30) {
    const strikes = [], callPrices = [], putPrices = [];
    const kMin = S0 * 0.7, kMax = S0 * 1.3;
    for (let i = 0; i <= steps; i++) {
        const K = kMin + (i / steps) * (kMax - kMin);
        strikes.push(+K.toFixed(0));
        callPrices.push(+blackScholes(S0, K, r, sigma, T, 'call').toFixed(2));
        putPrices.push(+blackScholes(S0, K, r, sigma, T, 'put').toFixed(2));
    }
    return { strikes, callPrices, putPrices };
}

/**
 * Monte Carlo Standard Error (95% confidence interval half-width).
 */
function mcStandardError(paths, K, r, T, optionType) {
    const n = paths.length;
    const payoffs = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const ST = paths[i][paths[i].length - 1];
        payoffs[i] = optionType === 'call'
            ? Math.max(ST - K, 0)
            : Math.max(K - ST, 0);
    }
    const discFactor = Math.exp(-r * T);
    const mean = payoffs.reduce((a, b) => a + b, 0) / n;
    const variance = payoffs.reduce((acc, p) => acc + (p - mean) ** 2, 0) / (n - 1);
    return discFactor * Math.sqrt(variance / n) * 1.96;  // 95% CI
}
