/**
 * server.js — Local dev server for Monte Carlo Option Pricing Simulator
 *
 * Serves static files AND handles /api/quote?symbol=RELIANCE live from
 * Yahoo Finance — no Vercel, no Python, no extra dependencies.
 *
 * Requires Node.js >= 18 (native fetch built-in).
 *
 * Usage:  node server.js
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = 3000;

// ── MIME types ───────────────────────────────────────────────────────────────
const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.woff2': 'font/woff2',
};

// ── /api/quote handler ───────────────────────────────────────────────────────
async function handleQuote(symbol, res) {
    const CORS = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Content-Type': 'application/json',
    };

    if (!symbol) {
        res.writeHead(400, CORS);
        return res.end(JSON.stringify({ error: 'symbol is required' }));
    }

    const ticker = encodeURIComponent(`${symbol}.NS`);
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=1y`;

    try {
        console.log(`  → Fetching live data for ${symbol} from Yahoo Finance…`);
        const resp = await fetch(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            },
            signal: AbortSignal.timeout(10000),
        });

        if (!resp.ok) throw new Error(`Yahoo Finance returned HTTP ${resp.status}`);

        const data = await resp.json();
        const result = data?.chart?.result?.[0];
        if (!result) throw new Error('No result in Yahoo Finance response');

        const meta = result.meta;
        const closes = result.indicators?.quote?.[0]?.close ?? [];
        const validCloses = closes.filter(v => v != null && !isNaN(v));

        // Annualised historical volatility (245 NSE trading days)
        let sigma = 0.28;
        if (validCloses.length > 20) {
            const logRets = [];
            for (let i = 1; i < validCloses.length; i++) {
                logRets.push(Math.log(validCloses[i] / validCloses[i - 1]));
            }
            const n = logRets.length;
            const mean = logRets.reduce((a, b) => a + b, 0) / n;
            const variance = logRets.reduce((acc, r) => acc + (r - mean) ** 2, 0) / (n - 1);
            sigma = Math.sqrt(variance * 245);
        }

        const payload = {
            price: meta.regularMarketPrice ?? meta.previousClose ?? validCloses.at(-1),
            sigma: +sigma.toFixed(4),
            name: meta.longName ?? meta.shortName ?? symbol,
            high52: meta.fiftyTwoWeekHigh ?? null,
            low52: meta.fiftyTwoWeekLow ?? null,
            source: 'yahoo_finance_live',
        };

        console.log(`  ✓ ${symbol}: ₹${payload.price}  σ=${(payload.sigma * 100).toFixed(1)}%`);
        res.writeHead(200, CORS);
        res.end(JSON.stringify(payload));

    } catch (err) {
        console.warn(`  ✗ ${symbol}: ${err.message} — frontend will use fallback`);
        res.writeHead(503, CORS);
        res.end(JSON.stringify({ error: err.message, source: 'error' }));
    }
}

// ── Static file handler ──────────────────────────────────────────────────────
function serveStatic(urlPath, res) {
    // Default to index.html
    const safePath = urlPath === '/' ? '/index.html' : urlPath;
    const filePath = path.join(__dirname, safePath);

    // Security: prevent directory traversal
    if (!filePath.startsWith(__dirname)) {
        res.writeHead(403);
        return res.end('Forbidden');
    }

    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            return res.end(`404 Not Found: ${safePath}`);
        }
        const ext = path.extname(filePath).toLowerCase();
        const mime = MIME[ext] ?? 'application/octet-stream';
        res.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'no-cache' });
        res.end(data);
    });
}

// ── Server ───────────────────────────────────────────────────────────────────
const server = http.createServer((req, res) => {
    const urlObj = new URL(req.url, `http://localhost:${PORT}`);

    if (urlObj.pathname === '/api/quote') {
        handleQuote(urlObj.searchParams.get('symbol'), res);
    } else {
        serveStatic(urlObj.pathname, res);
    }
});

server.listen(PORT, () => {
    console.log(`\n🚀  Monte Carlo Dev Server`);
    console.log(`   http://localhost:${PORT}\n`);
    console.log(`   Serves static files + /api/quote (live Yahoo Finance)\n`);
});
