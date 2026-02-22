/**
 * Vercel Serverless Function: GET /api/quote?symbol=RELIANCE
 *
 * Proxies Yahoo Finance to avoid CORS issues.
 * Returns: { price, sigma, name, high52, low52 }
 *
 * Uses the free Yahoo Finance v8 chart API with .NS suffix for NSE stocks.
 */

export default async function handler(req, res) {
    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET');

    const { symbol } = req.query;
    if (!symbol) {
        return res.status(400).json({ error: 'symbol is required' });
    }

    const ticker = encodeURIComponent(`${symbol}.NS`);
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=1y`;

    try {
        const resp = await fetch(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            },
            signal: AbortSignal.timeout(8000),
        });

        if (!resp.ok) {
            throw new Error(`Yahoo Finance returned ${resp.status}`);
        }

        const data = await resp.json();
        const result = data?.chart?.result?.[0];
        if (!result) throw new Error('No data returned');

        const meta = result.meta;
        const closes = result.indicators?.quote?.[0]?.close ?? [];

        // Compute annualised historical volatility (245 NSE trading days)
        const validCloses = closes.filter(v => v != null && !isNaN(v));
        let sigma = 0.28; // fallback
        if (validCloses.length > 20) {
            const logRets = [];
            for (let i = 1; i < validCloses.length; i++) {
                logRets.push(Math.log(validCloses[i] / validCloses[i - 1]));
            }
            const n = logRets.length;
            const mean = logRets.reduce((a, b) => a + b, 0) / n;
            const variance = logRets.reduce((acc, r) => acc + (r - mean) ** 2, 0) / (n - 1);
            sigma = Math.sqrt(variance * 245);   // annualise for NSE
        }

        return res.status(200).json({
            price: meta.regularMarketPrice ?? meta.previousClose ?? validCloses[validCloses.length - 1],
            sigma: +sigma.toFixed(4),
            name: meta.longName ?? meta.shortName ?? symbol,
            high52: meta.fiftyTwoWeekHigh ?? null,
            low52: meta.fiftyTwoWeekLow ?? null,
            source: 'yahoo_finance',
        });

    } catch (err) {
        // Return 503 so the frontend knows to use fallback data
        return res.status(503).json({
            error: err.message,
            source: 'error',
        });
    }
}
