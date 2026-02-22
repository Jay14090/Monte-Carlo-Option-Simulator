/**
 * charts.js — All Chart.js chart instances for the Simulator
 * Uses Chart.js 4.x loaded from CDN in index.html
 */

'use strict';

// ── Chart.js global defaults ────────────────────────────────────────────────
const CHART_FONT = "'Inter', system-ui, sans-serif";
const CHART_MONO = "'JetBrains Mono', monospace";

const TOOLTIP_BASE = {
    backgroundColor: 'rgba(10, 15, 35, 0.97)',
    borderColor: 'rgba(99,102,241,0.5)',
    borderWidth: 1,
    titleColor: '#f1f5f9',
    bodyColor: '#94a3b8',
    footerColor: '#6366f1',
    padding: 14,
    cornerRadius: 10,
    titleFont: { family: CHART_FONT, size: 12, weight: '700' },
    bodyFont: { family: CHART_MONO, size: 11 },
    footerFont: { family: CHART_FONT, size: 10, style: 'italic' },
    displayColors: true,
    boxWidth: 8,
    boxHeight: 8,
    boxPadding: 4,
};

const SCALE_BASE = {
    x: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#64748b', font: { family: CHART_FONT, size: 11 }, padding: 6 },
        border: { color: 'rgba(255,255,255,0.06)' },
    },
    y: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#64748b', font: { family: CHART_FONT, size: 11 }, padding: 6 },
        border: { color: 'rgba(255,255,255,0.06)' },
    },
};

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    animation: { duration: 700, easing: 'easeInOutCubic' },
    plugins: {
        legend: {
            labels: {
                color: '#94a3b8',
                font: { family: CHART_FONT, size: 12 },
                boxWidth: 10,
                padding: 20,
                usePointStyle: true,
                pointStyleWidth: 14,
            },
        },
        tooltip: TOOLTIP_BASE,
    },
    scales: SCALE_BASE,
};

// ── Chart Registry ──────────────────────────────────────────────────────────
const charts = {};

function destroyChart(id) {
    if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// ── Format Indian Rupees ────────────────────────────────────────────────────
function formatINR(value) {
    if (value === null || value === undefined || isNaN(value)) return '—';
    return '₹' + Number(value).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── 1. Simulation Chart (Canvas – custom renderer) ──────────────────────────
/**
 * Renders animated GBM paths directly on a <canvas> for performance.
 * Paths above strike → emerald, below → rose. Draw 60 paths max.
 */
function renderSimulationChart(canvasId, t, paths, K, T) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Show up to 120 paths for a lush, dense visual
    const displayPaths = paths.slice(0, Math.min(120, paths.length));
    const steps = t.length;
    let frame = 0;
    let animId;
    let started = false;

    function draw() {
        const rect = canvas.getBoundingClientRect();
        const W = Math.max(rect.width || canvas.clientWidth || canvas.parentElement.clientWidth, 200);
        const H = Math.max(rect.height || canvas.clientHeight || canvas.parentElement.clientHeight, 200);

        if (!started) {
            canvas.width = W * devicePixelRatio;
            canvas.height = H * devicePixelRatio;
            started = true;
        }
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(devicePixelRatio, devicePixelRatio);
        ctx.clearRect(0, 0, W, H);

        // ─── Layout padding
        const pad = { left: 80, right: 32, top: 32, bottom: 48 };
        const cW = W - pad.left - pad.right;
        const cH = H - pad.top - pad.bottom;

        // ─── Data bounds
        const endStep = Math.min(frame, displayPaths[0].length - 1);
        let minP = Infinity, maxP = -Infinity;
        for (const p of displayPaths) {
            for (let i = 0; i <= endStep; i++) {
                if (p[i] < minP) minP = p[i];
                if (p[i] > maxP) maxP = p[i];
            }
        }
        minP = Math.min(minP, K * 0.82);
        maxP = Math.max(maxP, K * 1.18);
        const range = maxP - minP || 1;

        const px = (ti) => pad.left + (ti / Math.max(steps - 1, 1)) * cW;
        const py = (v) => pad.top + cH - ((v - minP) / range) * cH;

        // ─── Background: subtle dark area between min/max band
        {
            const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
            grad.addColorStop(0, 'rgba(99,102,241,0.04)');
            grad.addColorStop(0.5, 'rgba(0,0,0,0)');
            grad.addColorStop(1, 'rgba(249,115,22,0.03)');
            ctx.fillStyle = grad;
            ctx.fillRect(pad.left, pad.top, cW, cH);
        }

        // ─── Grid lines Y
        const yTicks = 7;
        ctx.setLineDash([]);
        for (let i = 0; i <= yTicks; i++) {
            const v = minP + (i / yTicks) * range;
            const y = py(v);
            // Grid
            ctx.lineWidth = 0.5;
            ctx.strokeStyle = 'rgba(255,255,255,0.05)';
            ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
            // Tick label
            ctx.fillStyle = '#64748b';
            ctx.font = `11px ${CHART_MONO}`;
            ctx.textAlign = 'right';
            ctx.fillText('₹' + Math.round(v).toLocaleString('en-IN'), pad.left - 10, y + 4);
        }

        // ─── Grid lines X (light verticals)
        const xTicks = 6;
        for (let i = 0; i <= xTicks; i++) {
            const si = Math.round((i / xTicks) * (steps - 1));
            const x = px(si);
            ctx.lineWidth = 0.5;
            ctx.strokeStyle = 'rgba(255,255,255,0.04)';
            ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + cH); ctx.stroke();
            const tVal = (si / Math.max(steps - 1, 1)) * T;
            ctx.fillStyle = '#64748b';
            ctx.font = `11px ${CHART_FONT}`;
            ctx.textAlign = 'center';
            ctx.fillText(tVal.toFixed(2) + 'y', x, pad.top + cH + 20);
        }

        // ─── Strike dashed line (drawn before paths so paths overlay it)
        const strikeY = py(K);
        ctx.setLineDash([8, 5]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(249,115,22,0.7)';
        ctx.beginPath(); ctx.moveTo(pad.left, strikeY); ctx.lineTo(W - pad.right, strikeY); ctx.stroke();
        ctx.setLineDash([]);

        // ─── Compute final prices for color mapping
        const finalPrices = displayPaths.map(p => p[endStep]);
        const maxDist = Math.max(...finalPrices.map(f => Math.abs(f - K))) || 1;

        // ─── Draw paths — two passes: dim first, bright on top
        // Sort: most extreme last so they're drawn on top
        const indexed = finalPrices.map((f, i) => ({ i, dist: Math.abs(f - K) }));
        indexed.sort((a, b) => a.dist - b.dist);  // dimmer paths first, bright on top

        for (const { i: pi, dist } of indexed) {
            const path = displayPaths[pi];
            const ST = finalPrices[pi];
            const isAbove = ST > K;

            // Intensity 0–1 based on how far above/below strike
            const intensity = Math.min(dist / maxDist, 1);

            // Opacity: dim paths = 0.2, extreme paths = 0.75
            const alpha = 0.18 + intensity * 0.57;

            // Line width: thin at center, thicker at extremes
            const lw = 0.7 + intensity * 0.9;

            if (isAbove) {
                // Green spectrum: dim teal → bright emerald
                const g = Math.round(150 + intensity * 105);  // 150–255
                const b = Math.round(129 + intensity * 30);
                ctx.strokeStyle = `rgba(16, ${g}, ${b}, ${alpha})`;
            } else {
                // Red spectrum: dim rose → bright crimson
                const r = Math.round(200 + intensity * 55);
                const g2 = Math.round(63 - intensity * 30);
                ctx.strokeStyle = `rgba(${r}, ${g2}, 94, ${alpha})`;
            }

            // Glow on the most extreme paths
            if (intensity > 0.85) {
                ctx.shadowColor = isAbove ? 'rgba(16,185,129,0.5)' : 'rgba(244,63,94,0.5)';
                ctx.shadowBlur = 6;
            } else {
                ctx.shadowBlur = 0;
            }

            ctx.lineWidth = lw;
            ctx.beginPath();
            ctx.moveTo(px(0), py(path[0]));
            for (let j = 1; j <= endStep; j++) {
                ctx.lineTo(px(j), py(path[j]));
            }
            ctx.stroke();
        }
        ctx.shadowBlur = 0;

        // ─── Strike price label pill (drawn after paths)
        const strikeLabel = `Strike  ₹${Math.round(K).toLocaleString('en-IN')}`;
        ctx.font = `bold 11px ${CHART_MONO}`;
        const slW = ctx.measureText(strikeLabel).width + 20;
        const slX = pad.left + cW - slW - 6;
        const slY = strikeY - 15;
        ctx.fillStyle = 'rgba(249,115,22,0.20)';
        ctx.strokeStyle = 'rgba(249,115,22,0.8)';
        ctx.lineWidth = 1;
        roundRect(ctx, slX, slY, slW, 20, 5);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#f97316';
        ctx.textAlign = 'left';
        ctx.fillText(strikeLabel, slX + 10, slY + 14);

        // ─── Live IN/OUT badges at current frame
        if (endStep > 0) {
            const above = finalPrices.filter(f => f > K).length;
            const below = displayPaths.length - above;
            const pctAbove = ((above / displayPaths.length) * 100).toFixed(0);

            // Above badge
            const ab1 = `▲ ${above} above  (${pctAbove}%)`;
            ctx.font = `bold 11px ${CHART_FONT}`;
            const bw1 = ctx.measureText(ab1).width + 20;
            ctx.fillStyle = 'rgba(16,185,129,0.15)';
            ctx.strokeStyle = 'rgba(16,185,129,0.5)';
            ctx.lineWidth = 1;
            roundRect(ctx, pad.left + 8, pad.top + 8, bw1, 22, 5);
            ctx.fill(); ctx.stroke();
            ctx.fillStyle = '#10b981';
            ctx.textAlign = 'left';
            ctx.fillText(ab1, pad.left + 18, pad.top + 23);

            // Below badge
            const ab2 = `▼ ${below} below  (${100 - +pctAbove}%)`;
            const bw2 = ctx.measureText(ab2).width + 20;
            ctx.fillStyle = 'rgba(244,63,94,0.15)';
            ctx.strokeStyle = 'rgba(244,63,94,0.5)';
            roundRect(ctx, pad.left + 8 + bw1 + 8, pad.top + 8, bw2, 22, 5);
            ctx.fill(); ctx.stroke();
            ctx.fillStyle = '#f43f5e';
            ctx.fillText(ab2, pad.left + 8 + bw1 + 18, pad.top + 23);
        }

        // ─── Progress pill (top right)
        const currentT = (endStep / Math.max(steps - 1, 1)) * T;
        const progLabel = `  t = ${currentT.toFixed(3)}y  ·  step ${endStep}/${steps - 1}  `;
        ctx.font = `11px ${CHART_MONO}`;
        const plW = ctx.measureText(progLabel).width + 4;
        ctx.fillStyle = 'rgba(8,13,24,0.85)';
        ctx.strokeStyle = 'rgba(99,102,241,0.35)';
        ctx.lineWidth = 1;
        roundRect(ctx, W - pad.right - plW - 4, pad.top + 8, plW + 4, 22, 5);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#818cf8';
        ctx.textAlign = 'right';
        ctx.fillText(progLabel, W - pad.right - 6, pad.top + 23);
        ctx.textAlign = 'left';

        // ─── Axis labels
        ctx.save();
        ctx.translate(18, pad.top + cH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#64748b';
        ctx.font = `12px ${CHART_FONT}`;
        ctx.textAlign = 'center';
        ctx.fillText('Stock Price (₹)', 0, 0);
        ctx.restore();

        ctx.fillStyle = '#64748b';
        ctx.font = `12px ${CHART_FONT}`;
        ctx.textAlign = 'center';
        ctx.fillText('Time to Expiry (years)', pad.left + cW / 2, H - 6);
        ctx.textAlign = 'left';

        frame++;
        if (frame < steps) animId = requestAnimationFrame(draw);
    }

    if (animId) cancelAnimationFrame(animId);
    frame = 0;
    requestAnimationFrame(draw);

    return { cancel: () => cancelAnimationFrame(animId) };
}


function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
}

// ── 2. Convergence Chart ─────────────────────────────────────────────────────
function renderConvergenceChart(canvasId, pathCounts, mcPrices, bsPrice, optionType) {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const isCall = optionType === 'call';
    const color = isCall ? '#10b981' : '#f43f5e';
    const colorDim = isCall ? 'rgba(16,185,129,0.12)' : 'rgba(244,63,94,0.12)';

    charts[canvasId] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: pathCounts.map(n => n >= 1000 ? (n / 1000).toFixed(0) + 'K' : n),
            datasets: [
                {
                    label: `MC Price — ${isCall ? 'CALL' : 'PUT'}`,
                    data: mcPrices,
                    borderColor: color,
                    backgroundColor: colorDim,
                    borderWidth: 2.5,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    pointBackgroundColor: color,
                    pointBorderColor: 'rgba(8,13,24,0.8)',
                    pointBorderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    order: 1,
                },
                {
                    label: 'Black-Scholes (exact)',
                    data: pathCounts.map(() => bsPrice),
                    borderColor: '#818cf8',
                    borderDash: [7, 4],
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    fill: false,
                    order: 0,
                },
            ],
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                ...CHART_DEFAULTS.plugins,
                legend: {
                    ...CHART_DEFAULTS.plugins.legend,
                    position: 'top',
                },
                tooltip: {
                    ...TOOLTIP_BASE,
                    callbacks: {
                        title: (items) => `After ${items[0].label} simulations`,
                        label: (ctx) => {
                            const v = ctx.raw;
                            return `  ${ctx.dataset.label}: ${formatINR(v)}`;
                        },
                        footer: (items) => {
                            const mc = items.find(i => i.datasetIndex === 0)?.raw;
                            if (mc === undefined) return '';
                            const err = Math.abs(mc - bsPrice);
                            const pct = bsPrice > 0 ? (err / bsPrice * 100).toFixed(2) : '—';
                            return `Error vs BS: ${formatINR(err)} (${pct}%)`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ...SCALE_BASE.x,
                    title: { display: true, text: 'Number of Paths', color: '#64748b', font: { size: 11 } },
                },
                y: {
                    ...SCALE_BASE.y,
                    title: { display: true, text: 'Option Price (₹)', color: '#64748b', font: { size: 11 } },
                    ticks: {
                        ...SCALE_BASE.y.ticks,
                        callback: v => '₹' + Number(v).toFixed(2),
                    },
                },
            },
        },
    });
}

// ── 3. Sensitivity Chart ─────────────────────────────────────────────────────
function renderSensitivityChart(canvasId, labels, callData, putData, xLabel, activeOption) {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const isVolChart = xLabel.includes('olatility');

    charts[canvasId] = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Call Price (₹)',
                    data: callData,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16,185,129,0.07)',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 7,
                    pointHoverBackgroundColor: '#10b981',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    fill: activeOption === 'call',
                    tension: 0.4,
                },
                {
                    label: 'Put Price (₹)',
                    data: putData,
                    borderColor: '#f43f5e',
                    backgroundColor: 'rgba(244,63,94,0.07)',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 7,
                    pointHoverBackgroundColor: '#f43f5e',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    fill: activeOption === 'put',
                    tension: 0.4,
                },
            ],
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                ...CHART_DEFAULTS.plugins,
                legend: { ...CHART_DEFAULTS.plugins.legend, position: 'top' },
                tooltip: {
                    ...TOOLTIP_BASE,
                    callbacks: {
                        title: (items) => {
                            const lbl = items[0].label;
                            return isVolChart
                                ? `Volatility (σ) = ${lbl}%`
                                : `Strike Price (K) = ₹${lbl}`;
                        },
                        label: (ctx) => `  ${ctx.dataset.label}: ${formatINR(ctx.raw)}`,
                        footer: (items) => {
                            const call = items.find(i => i.datasetIndex === 0)?.raw ?? 0;
                            const put = items.find(i => i.datasetIndex === 1)?.raw ?? 0;
                            const pcDiff = call + put > 0 ? ((call - put) / ((call + put) / 2) * 100).toFixed(1) : '—';
                            return `Call − Put spread: ${formatINR(call - put)}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ...SCALE_BASE.x,
                    title: { display: true, text: xLabel, color: '#64748b', font: { size: 11 } },
                    ticks: {
                        ...SCALE_BASE.x.ticks,
                        maxTicksLimit: 10,
                    },
                },
                y: {
                    ...SCALE_BASE.y,
                    title: { display: true, text: 'Option Price (₹)', color: '#64748b', font: { size: 11 } },
                    ticks: {
                        ...SCALE_BASE.y.ticks,
                        callback: v => '₹' + Number(v).toFixed(2),
                    },
                },
            },
        },
    });
}

// ── 4. Payoff Diagram ────────────────────────────────────────────────────────
function renderPayoffChart(canvasId, S0, K, premium, optionType) {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const steps = 80;
    const sMin = S0 * 0.45, sMax = S0 * 1.55;
    const stockPrices = Array.from({ length: steps }, (_, i) => sMin + (i / (steps - 1)) * (sMax - sMin));

    const isCall = optionType === 'call';
    const payoffs = stockPrices.map(S => isCall ? Math.max(S - K, 0) : Math.max(K - S, 0));
    const profitLoss = payoffs.map(p => p - premium);

    const color = isCall ? '#10b981' : '#f43f5e';
    const breakEven = isCall ? K + premium : K - premium;

    const labels = stockPrices.map(s => s.toFixed(0));

    charts[canvasId] = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Profit / Loss at Expiry',
                    data: profitLoss,
                    borderColor: (ctx) => {
                        const v = ctx.raw ?? 0;
                        return v >= 0 ? '#10b981' : '#f43f5e';
                    },
                    segment: {
                        borderColor: ctx => ctx.p0.parsed.y >= 0 ? '#10b981' : '#f43f5e',
                    },
                    backgroundColor: (ctx) => {
                        const chart = ctx.chart;
                        const { ctx: c, chartArea } = chart;
                        if (!chartArea) return 'transparent';
                        const grad = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                        grad.addColorStop(0, isCall ? 'rgba(16,185,129,0.20)' : 'rgba(244,63,94,0.20)');
                        grad.addColorStop(1, 'transparent');
                        return grad;
                    },
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 7,
                    pointHoverBackgroundColor: color,
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    fill: true,
                    tension: 0.1,
                },
            ],
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                ...CHART_DEFAULTS.plugins,
                legend: { display: false },
                tooltip: {
                    ...TOOLTIP_BASE,
                    callbacks: {
                        title: (items) => `Stock at Expiry: ₹${Number(items[0].label).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`,
                        label: (ctx) => {
                            const pl = ctx.raw;
                            const arrow = pl >= 0 ? '▲ PROFIT' : '▼ LOSS';
                            return `  ${arrow}: ${pl >= 0 ? '+' : ''}${formatINR(pl)}`;
                        },
                        footer: (items) => {
                            const S = parseFloat(items[0].label);
                            const lines = [
                                `Premium paid: ${formatINR(premium)}`,
                                `Break-even:   ₹${breakEven.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`,
                            ];
                            if (isCall) {
                                lines.push(S > K ? `In-the-money: ${formatINR(S - K)}` : `Out-of-money: ₹${(K - S).toFixed(0)}`);
                            } else {
                                lines.push(S < K ? `In-the-money: ${formatINR(K - S)}` : `Out-of-money: ₹${(S - K).toFixed(0)}`);
                            }
                            return lines;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ...SCALE_BASE.x,
                    ticks: { ...SCALE_BASE.x.ticks, maxTicksLimit: 9, callback: v => '₹' + v },
                    title: { display: true, text: 'Stock Price at Expiry (₹)', color: '#64748b', font: { size: 11 } },
                },
                y: {
                    ...SCALE_BASE.y,
                    title: { display: true, text: 'Profit / Loss (₹)', color: '#64748b', font: { size: 11 } },
                    ticks: {
                        ...SCALE_BASE.y.ticks,
                        callback: v => (v >= 0 ? '+' : '') + '₹' + Math.abs(v).toFixed(0),
                    },
                },
            },
        },
    });
}
