/**
 * app.js — Main application orchestration
 * Wires up all UI interactions, fetches live stock data, runs simulation.
 */

'use strict';

// ── State ───────────────────────────────────────────────────────────────────
const state = {
    symbol: 'RELIANCE',
    S0: 1285,
    K: 1290,
    r: 0.0625,       // RBI Repo Rate default
    sigma: 0.26,
    T: 30,           // 30 days
    yearBasis: 250,  // NSE default
    nPaths: 10000,
    steps: 250,          // Match yearBasis
    optionType: 'call',

    // Results
    paths: null,
    t: null,
    mcPrice: null,
    bsPrice: null,
    greeks: null,
    simCancel: null,
    sensitivityTab: 'vol',
};

// ── Utilities ────────────────────────────────────────────────────────────────
function fmtINR(v) {
    if (v == null || isNaN(v)) return '—';
    return '₹' + (+v).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(v) {
    if (v == null || isNaN(v)) return '—';
    return (v * 100).toFixed(2) + '%';
}

function fmtNum(v, d = 4) {
    if (v == null || isNaN(v)) return '—';
    return (+v).toFixed(d);
}

function $(id) { return document.getElementById(id); }
function $$(sel) { return document.querySelectorAll(sel); }

function showToast(msg) {
    const t = $('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3000);
}

function updateSliderFill(input) {
    const min = parseFloat(input.min), max = parseFloat(input.max);
    const val = parseFloat(input.value);
    const pct = ((val - min) / (max - min)) * 100;
    input.style.setProperty('--pct', pct + '%');
}

// ── Stock Search ─────────────────────────────────────────────────────────────
let dropdownVisible = false;

function populateDropdown(filter = '') {
    const dd = $('stockDropdown');
    dd.innerHTML = '';
    const fl = filter.toLowerCase();
    const filtered = NIFTY50.filter(s =>
        s.symbol.toLowerCase().includes(fl) ||
        s.name.toLowerCase().includes(fl) ||
        s.sector.toLowerCase().includes(fl)
    );

    if (filtered.length === 0) {
        dd.innerHTML = '<div class="stock-dropdown-item" style="color:var(--txt-muted)">No results</div>';
        return;
    }

    filtered.forEach(stock => {
        const div = document.createElement('div');
        div.className = 'stock-dropdown-item' + (stock.symbol === state.symbol ? ' active' : '');
        div.innerHTML = `
      <div class="stock-item-main">
        <span class="stock-item-symbol">${stock.symbol}</span>
        <span class="stock-item-name">${stock.name}</span>
      </div>
      <span class="stock-item-sector">${stock.sector}</span>
    `;
        div.addEventListener('mousedown', (e) => {
            e.preventDefault();
            selectStock(stock.symbol);
        });
        dd.appendChild(div);
    });
}

function showDropdown() {
    $('stockDropdown').style.display = 'block';
    dropdownVisible = true;
}

function hideDropdown() {
    $('stockDropdown').style.display = 'none';
    dropdownVisible = false;
}

async function selectStock(symbol) {
    state.symbol = symbol;
    const stock = getStockBySymbol(symbol);
    if (!stock) return;

    $('stockSearchInput').value = `${symbol} — ${stock.name}`;
    hideDropdown();

    // Show loading state
    $('priceValue').textContent = '...';
    $('volValue').textContent = '...';

    // Fetch live quote
    await fetchLiveQuote(symbol);
}

async function fetchLiveQuote(symbol) {
    try {
        // Try Vercel serverless function first
        const resp = await fetch(`/api/quote?symbol=${symbol}`);
        if (resp.ok) {
            const data = await resp.json();
            applyQuote(data.price, data.sigma, data.name, data.high52, data.low52, true);
            return;
        }
    } catch (e) { /* fall through */ }

    // Fallback: use pre-computed data
    const stock = getStockBySymbol(symbol);
    const price = getFallbackPrice(symbol);
    applyQuote(price, stock?.typicalVol ?? 0.28, stock?.name ?? symbol, price * 1.3, price * 0.7, false);
    showToast('Using cached data — live price unavailable in local mode.');
}

function applyQuote(price, sigma, name, high52, low52, isLive) {
    state.S0 = price;
    state.sigma = sigma;
    state.K = Math.round(price * 1.02 / 5) * 5; // ~2% OTM, rounded to nearest 5

    // Update UI
    $('priceValue').textContent = fmtINR(price);
    $('volValue').textContent = fmtPct(sigma);
    $('stockName').textContent = name || state.symbol;

    // Update sliders
    updateParamSlider('S0', price, 50, 8000);
    updateParamSlider('K', state.K, 50, 8000);
    updateParamSlider('sigma', sigma, 0.01, 1.0);

    // Mark live indicator
    const liveBadge = $('liveBadge');
    if (isLive) {
        liveBadge.textContent = '● LIVE';
        liveBadge.style.color = '#10b981';
    } else {
        liveBadge.textContent = '● CACHED';
        liveBadge.style.color = '#f97316';
    }
}

function updateParamSlider(param, value, min, max) {
    const slider = $(`slider_${param}`);
    const input = $(`input_${param}`);
    if (!slider) return;
    slider.min = min;
    slider.max = max;
    slider.value = value;
    updateSliderFill(slider);
    if (input) {
        if (param === 'sigma' || param === 'r') input.value = (value * 100).toFixed(2);
        else input.value = value;
    }
}

// ── Parameter Slider Bindings ────────────────────────────────────────────────
function bindSliders() {
    const sliders = [
        { id: 'S0', min: 50, max: 8000, step: 1, format: fmtINR, key: 'S0' },
        { id: 'K', min: 50, max: 8000, step: 1, format: fmtINR, key: 'K' },
        { id: 'sigma', min: 0.05, max: 1.0, step: 0.005, format: v => fmtPct(v), key: 'sigma' },
        { id: 'r', min: 0.01, max: 0.25, step: 0.001, format: v => fmtPct(v), key: 'r' },
        { id: 'T', min: 1, max: 730, step: 1, format: v => v + ' days', key: 'T' },
        { id: 'nPaths', min: 1000, max: 50000, step: 500, format: v => (+v).toLocaleString('en-IN'), key: 'nPaths' },
    ];

    sliders.forEach(cfg => {
        const slider = $(`slider_${cfg.id}`);
        const input = $(`input_${cfg.id}`);
        if (!slider) return;

        slider.min = cfg.min;
        slider.max = cfg.max;
        slider.step = cfg.step;
        slider.value = state[cfg.key];
        updateSliderFill(slider);
        if (input) {
            if (cfg.key === 'sigma' || cfg.key === 'r') input.value = (state[cfg.key] * 100).toFixed(2);
            else input.value = state[cfg.key];
        }

        slider.addEventListener('input', () => {
            const v = parseFloat(slider.value);
            state[cfg.key] = (cfg.key === 'nPaths' || cfg.key === 'T') ? parseInt(slider.value) : v;
            if (input) {
                if (cfg.key === 'sigma' || cfg.key === 'r') input.value = (v * 100).toFixed(2);
                else input.value = v;
            }
            updateSliderFill(slider);
            if (cfg.id === 'sigma') updateVolCursor(v);
        });
    });
}

// ── Manual Keyboard Inputs ──────────────────────────────────────────────────
function bindManualInputs() {
    const inputs = [
        { id: 'S0', key: 'S0', isPct: false },
        { id: 'K', key: 'K', isPct: false },
        { id: 'sigma', key: 'sigma', isPct: true },
        { id: 'r', key: 'r', isPct: true },
        { id: 'T', key: 'T', isPct: false },
        { id: 'nPaths', key: 'nPaths', isPct: false },
    ];

    inputs.forEach(cfg => {
        const el = $(`input_${cfg.id}`);
        const slider = $(`slider_${cfg.id}`);
        if (!el || !slider) return;

        el.addEventListener('input', () => {
            let val = parseFloat(el.value);
            if (isNaN(val)) return;

            if (cfg.isPct) val = val / 100;

            // Update state
            state[cfg.key] = (cfg.key === 'nPaths' || cfg.key === 'T') ? Math.round(val) : val;

            // Sync slider
            slider.value = val;
            updateSliderFill(slider);
            if (cfg.id === 'sigma') updateVolCursor(val);
        });

        // Clamp on blur
        el.addEventListener('blur', () => {
            let val = parseFloat(el.value);
            if (isNaN(val)) {
                // Reset to state
                if (cfg.isPct) el.value = (state[cfg.key] * 100).toFixed(2);
                else el.value = state[cfg.key];
                return;
            }
            if (cfg.isPct) val = val / 100;

            const min = parseFloat(slider.min);
            const max = parseFloat(slider.max);
            const clamped = Math.max(min, Math.min(max, val));

            state[cfg.key] = (cfg.key === 'nPaths' || cfg.key === 'T') ? Math.round(clamped) : clamped;
            slider.value = clamped;
            if (cfg.isPct) el.value = (clamped * 100).toFixed(2);
            else el.value = clamped;
            updateSliderFill(slider);
        });
    });
}

// ── Stepper Buttons ──────────────────────────────────────────────────────────
function bindSteppers() {
    $$('.stepper-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const sliderId = btn.dataset.id;
            const slider = $(sliderId);
            if (!slider) return;

            const step = parseFloat(slider.step) || 1;
            const isPlus = btn.classList.contains('plus');
            let val = parseFloat(slider.value);

            if (isPlus) val += step;
            else val -= step;

            // Clamp
            val = Math.max(parseFloat(slider.min), Math.min(parseFloat(slider.max), val));

            slider.value = val;
            // Trigger input event to update state and UI
            slider.dispatchEvent(new Event('input'));
        });
    });
}

// ── Option Type Toggle ───────────────────────────────────────────────────────
function bindOptionToggle() {
    const callBtn = $('btn_call');
    const putBtn = $('btn_put');

    function setType(type) {
        state.optionType = type;
        callBtn.className = 'toggle-btn' + (type === 'call' ? ' active-call' : '');
        putBtn.className = 'toggle-btn' + (type === 'put' ? ' active-put' : '');
    }

    callBtn.addEventListener('click', () => setType('call'));
    putBtn.addEventListener('click', () => setType('put'));
    setType('call');
}

// ── Year Basis Toggle ────────────────────────────────────────────────────────
function bindBasisToggle() {
    const btns = $$('.basis-btn');
    const display = $('display_yearBasis');

    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.yearBasis = parseInt(btn.dataset.basis);
            state.steps = state.yearBasis;
            if (display) display.textContent = state.yearBasis + ' days';
            // Results are now stale
            state.mcPrice = null;
        });
    });
}

// ── Main Simulation Run ──────────────────────────────────────────────────────
async function runSimulation() {
    const btn = $('simulateBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> Simulating…';

    const { S0, K, r, sigma, nPaths, steps, optionType, yearBasis } = state;
    const T = state.T / yearBasis; // Days to Years (Configurable context)

    // Brief yield to let spinner appear
    await new Promise(resolve => setTimeout(resolve, 10));

    // 1. Run GBM paths
    const { t, paths } = simulateGBM(S0, r, sigma, T, steps, nPaths);
    state.t = t; state.paths = paths;

    // 2. MC Pricing
    const mcPrice = monteCarloPricing(paths, K, r, T, optionType);
    const bsPrice = blackScholes(S0, K, r, sigma, T, optionType);
    const ci95 = mcStandardError(paths, K, r, T, optionType);
    state.mcPrice = mcPrice;
    state.bsPrice = bsPrice;

    // 3. Greeks
    const greeks = calculateGreeks(S0, K, r, sigma, T, optionType);
    state.greeks = greeks;

    // 4. Render everything
    renderPriceCards(mcPrice, bsPrice, ci95, optionType);
    renderGreeks(greeks, S0, optionType);

    // Animated simulation chart
    if (state.simCancel) state.simCancel.cancel();
    const ctrl = renderSimulationChart('simCanvas', t, paths, K, T);
    if (ctrl) state.simCancel = ctrl;

    // Convergence chart
    const conv = computeConvergence(S0, K, r, sigma, T, optionType, Math.min(nPaths, 10000), yearBasis);
    renderConvergenceChart('convergenceCanvas', conv.pathCounts, conv.mcPrices, conv.bsPrice, optionType);

    // Sensitivity chart
    renderActiveSensitivity();

    // Payoff chart
    renderPayoffChart('payoffCanvas', S0, K, mcPrice, optionType);

    // Update chart stats overlay
    const aboveStrike = paths.filter(p => p[p.length - 1] > K).length;
    const pctAbove = ((aboveStrike / nPaths) * 100).toFixed(1);
    $('simStat').textContent = `${pctAbove}% above strike at expiry`;

    btn.disabled = false;
    btn.innerHTML = '▶ Simulate';
}

// ── Price Cards ──────────────────────────────────────────────────────────────
function renderPriceCards(mcPrice, bsPrice, ci95, optionType) {
    const colorClass = optionType === 'call' ? 'call' : 'put';

    // MC card
    $('mc_price').textContent = fmtINR(mcPrice);
    $('mc_price').className = `pm-price ${colorClass} price-animate`;
    $('mc_label').textContent = optionType === 'call' ? 'Monte Carlo — CALL' : 'Monte Carlo — PUT';
    $('mc_ci').textContent = `95% CI: ${fmtINR(mcPrice - ci95)} — ${fmtINR(mcPrice + ci95)}`;

    // BS card
    $('bs_price').textContent = fmtINR(bsPrice);
    $('bs_price').className = `pm-price ${colorClass} price-animate`;
    $('bs_label').textContent = optionType === 'call' ? 'Black-Scholes — CALL' : 'Black-Scholes — PUT';

    // Difference
    const diff = Math.abs(mcPrice - bsPrice);
    const diffPct = bsPrice > 0 ? (diff / bsPrice * 100) : 0;
    const diffEl = $('price_diff');
    diffEl.textContent = `±${fmtINR(diff)} (${diffPct.toFixed(2)}%)`;
    diffEl.className = `diff-badge ${diffPct < 2 ? 'accurate' : 'moderate'}`;
    $('diff_label').textContent = diffPct < 2
        ? '✓ MC converged to Black-Scholes'
        : 'ℹ Try more simulation paths';
}

// ── Greeks Panel ─────────────────────────────────────────────────────────────
const GREEK_META = {
    delta: {
        symbol: 'Δ', name: 'Delta', color: '#818cf8',
        tip: (v, S0, type) => type === 'call'
            ? `If ${state.symbol} rises ₹10, your CALL gains approx. ${fmtINR(Math.abs(v) * 10)}.`
            : `If ${state.symbol} falls ₹10, your PUT gains approx. ${fmtINR(Math.abs(v) * 10)}.`,
        scaleMax: 1,
    },
    gamma: {
        symbol: 'Γ', name: 'Gamma', color: '#38bdf8',
        tip: () => 'Rate of change of Delta. High gamma = option price accelerates.\nHigher near ATM and expiry.',
        scaleMax: 0.1,
    },
    vega: {
        symbol: 'V', name: 'Vega', color: '#f59e0b',
        tip: (v) => `If India VIX rises 1%, the option price changes by ${fmtINR(v)}. High vega = sensitive to market volatility.`,
        scaleMax: 20,
    },
    theta: {
        symbol: 'Θ', name: 'Theta', color: '#f43f5e',
        tip: (v) => `Time decay: each day that passes, the option loses approx. ${fmtINR(Math.abs(v))} in value (assuming no price move).`,
        scaleMax: 5,
    },
    rho: {
        symbol: 'ρ', name: 'Rho', color: '#10b981',
        tip: (v, S0, type) => type === 'call'
            ? `If RBI raises repo rate by 1%, this CALL gains ${fmtINR(Math.abs(v))}.`
            : `If RBI raises repo rate by 1%, this PUT loses ${fmtINR(Math.abs(v))}.`,
        scaleMax: 20,
    },
};

function renderGreeks(greeks, S0, optionType) {
    ['delta', 'gamma', 'vega', 'theta', 'rho'].forEach(key => {
        const meta = GREEK_META[key];
        const value = greeks[key];
        const el = $(`greek_${key}`);
        if (!el) return;

        el.querySelector('.greek-value').textContent = fmtNum(value, 4);
        el.querySelector('.greek-bar-fill').style.width =
            Math.min(Math.abs(value) / meta.scaleMax * 100, 100) + '%';
        el.querySelector('.greek-bar-fill').style.background = meta.color;
        el.querySelector('.greek-tooltip').textContent = meta.tip(value, S0, optionType);
        el.querySelector('.greek-symbol').style.color = meta.color;
    });
}

// ── Sensitivity Tabs ─────────────────────────────────────────────────────────
function renderActiveSensitivity() {
    const { S0, K, r, sigma, yearBasis } = state;
    const T = state.T / yearBasis;
    const tab = state.sensitivityTab;

    if (tab === 'vol') {
        const { vols, callPrices, putPrices } = sensitivityVolatility(S0, K, r, T, state.optionType);
        renderSensitivityChart('sensitivityCanvas', vols, callPrices, putPrices, 'Volatility (%)', state.optionType);
    } else {
        const { strikes, callPrices, putPrices } = sensitivityStrike(S0, r, sigma, T, state.optionType);
        renderSensitivityChart('sensitivityCanvas', strikes, callPrices, putPrices, 'Strike Price (₹)', state.optionType);
    }
}

function bindSensitivityTabs() {
    $$('.sens-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.sens-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.sensitivityTab = btn.dataset.tab;
            if (state.mcPrice !== null) renderActiveSensitivity();
        });
    });
}

// ── Explainer Modal ───────────────────────────────────────────────────────────
function bindModal() {
    const backdrop = $('explainerModal');
    $('helpFab').addEventListener('click', () => backdrop.classList.add('open'));
    $('modalClose').addEventListener('click', () => backdrop.classList.remove('open'));
    backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) backdrop.classList.remove('open');
    });
}

// ── Rate Preset Buttons ───────────────────────────────────────────────────────
function bindRatePresets() {
    $$('[data-rate]').forEach(btn => {
        btn.addEventListener('click', () => {
            const rate = parseFloat(btn.dataset.rate);
            state.r = rate;
            const slider = $('slider_r');
            slider.value = rate;
            updateSliderFill(slider);
            $('display_r').textContent = fmtPct(rate);
            $$('[data-rate]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

// ── Volatility Regime Cursor ─────────────────────────────────────────────────
function updateVolCursor(sigma) {
    const cursor = $('volCursor');
    const bar = cursor ? cursor.parentElement : null;
    if (!cursor || !bar) return;

    // sigma range 0.05 → 1.0; zone boundaries: calm 0–0.2, normal 0.2–0.35, high 0.35–0.55, extreme 0.55+
    const pct = Math.min(Math.max((sigma - 0.05) / (1.0 - 0.05), 0), 1);
    const barW = bar.offsetWidth;
    cursor.style.left = (pct * barW - 1.5) + 'px';

    // Highlight active zone
    const zones = bar.querySelectorAll('.vol-zone');
    zones.forEach(z => z.classList.remove('active-zone'));
    if (sigma <= 0.20) zones[0]?.classList.add('active-zone');
    else if (sigma <= 0.35) zones[1]?.classList.add('active-zone');
    else if (sigma <= 0.55) zones[2]?.classList.add('active-zone');
    else zones[3]?.classList.add('active-zone');
}

// ── Sigma Preset Buttons ──────────────────────────────────────────────────────
function bindSigmaPresets() {
    $$('[data-sigma]').forEach(btn => {
        btn.addEventListener('click', () => {
            const sigma = parseFloat(btn.dataset.sigma);
            state.sigma = sigma;
            const slider = $('slider_sigma');
            slider.value = sigma;
            updateSliderFill(slider);
            $('display_sigma').textContent = fmtPct(sigma);
            updateVolCursor(sigma);
            $$('[data-sigma]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Hook the sigma slider directly to also update the cursor
    const sigmaSlider = $('slider_sigma');
    if (sigmaSlider) {
        sigmaSlider.addEventListener('input', () => updateVolCursor(parseFloat(sigmaSlider.value)));
    }
}

// ── Initialise ───────────────────────────────────────────────────────────────
async function init() {
    // Populate stock dropdown
    populateDropdown();

    // Search filter
    const input = $('stockSearchInput');
    input.addEventListener('focus', () => { populateDropdown(input.value); showDropdown(); });
    input.addEventListener('blur', () => setTimeout(hideDropdown, 150));
    input.addEventListener('input', () => { populateDropdown(input.value); showDropdown(); });

    // Bind all sliders + steppers
    bindSliders();
    bindSteppers();
    bindManualInputs();

    // Rate presets
    bindRatePresets();

    // Sigma (volatility) presets + regime cursor
    bindSigmaPresets();
    updateVolCursor(state.sigma);

    // Option type toggle
    bindOptionToggle();

    // Year basis toggle
    bindBasisToggle();

    // Modal
    bindModal();
    bindSensitivityTabs();

    // Simulate button
    $('simulateBtn').addEventListener('click', runSimulation);

    // Load initial stock
    await selectStock('RELIANCE');

    // Auto-run initial simulation after stock loads
    await runSimulation();
}

document.addEventListener('DOMContentLoaded', init);
