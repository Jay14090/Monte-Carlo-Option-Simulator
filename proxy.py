"""
proxy.py — Local dev server for Monte Carlo Option Pricing Simulator

Serves static files from this directory AND handles /api/quote?symbol=RELIANCE
with LIVE data from Yahoo Finance — no Vercel, no Node.js needed.

Usage:
    python proxy.py

Requires Python 3.8+. Uses only stdlib (http.server + urllib).
"""

import http.server
import json
import math
import os
import urllib.request
import urllib.parse
from pathlib import Path

PORT = 3000
ROOT = Path(__file__).parent.resolve()

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".json": "application/json",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
}

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET",
}


def fetch_live_quote(symbol: str) -> dict:
    """Fetch 1yr daily data from Yahoo Finance and compute annualised σ."""
    ticker = urllib.parse.quote(f"{symbol}.NS")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{ticker}?interval=1d&range=1y"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    result = data.get("chart", {}).get("result", [None])[0]
    if not result:
        raise ValueError("No result block in Yahoo Finance response")

    meta   = result.get("meta", {})
    closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    valid  = [c for c in closes if c is not None and not math.isnan(c)]

    # Annualised historical volatility — 245 NSE trading days
    sigma = 0.28  # fallback
    if len(valid) > 20:
        log_rets = [math.log(valid[i] / valid[i - 1]) for i in range(1, len(valid))]
        n    = len(log_rets)
        mean = sum(log_rets) / n
        var  = sum((r - mean) ** 2 for r in log_rets) / (n - 1)
        sigma = math.sqrt(var * 245)

    price = (
        meta.get("regularMarketPrice")
        or meta.get("previousClose")
        or valid[-1]
    )

    return {
        "price":  round(float(price), 2),
        "sigma":  round(sigma, 4),
        "name":   meta.get("longName") or meta.get("shortName") or symbol,
        "high52": meta.get("fiftyTwoWeekHigh"),
        "low52":  meta.get("fiftyTwoWeekLow"),
        "source": "yahoo_finance_live",
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Custom cleaner logging
        pass

    def send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs     = urllib.parse.parse_qs(parsed.query)

        # ── API route ────────────────────────────────────────────────────────
        if parsed.path == "/api/quote":
            symbol = qs.get("symbol", [None])[0]
            if not symbol:
                return self.send_json(400, {"error": "symbol is required"})
            try:
                print(f"  → Live quote: {symbol} …", end=" ", flush=True)
                data = fetch_live_quote(symbol)
                print(f"₹{data['price']}  σ={data['sigma']*100:.1f}%")
                self.send_json(200, data)
            except Exception as e:
                print(f"FAILED — {e}")
                self.send_json(503, {"error": str(e), "source": "error"})
            return

        # ── Static files ─────────────────────────────────────────────────────
        url_path = parsed.path.lstrip("/") or "index.html"
        file_path = (ROOT / url_path).resolve()

        # Security: prevent directory traversal
        if not str(file_path).startswith(str(ROOT)):
            self.send_response(403)
            self.end_headers()
            return

        if not file_path.is_file():
            body = f"404 Not Found: {url_path}".encode()
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        ext  = file_path.suffix.lower()
        mime = MIME_TYPES.get(ext, "application/octet-stream")
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    os.chdir(ROOT)
    server = http.server.HTTPServer(("", PORT), Handler)
    print(f"\n🚀  Monte Carlo Live Dev Server")
    print(f"   http://localhost:{PORT}")
    print(f"   Static files + /api/quote → Yahoo Finance (LIVE)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
