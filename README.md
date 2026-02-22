<<<<<<< HEAD
# 📈 Monte Carlo Option Pricer — India Edition

<div align="center">
  <img src="https://img.shields.io/badge/NSE-Standard-orange?style=for-the-badge&logo=stock-market" alt="NSE Standard" />
  <img src="https://img.shields.io/badge/Engineered_by-Jay_C-indigo?style=for-the-badge&logo=github" alt="Engineered by Jay C" />
  <img src="https://img.shields.io/badge/Build-Vanilla_JS-blue?style=for-the-badge&logo=javascript" alt="Build Vanilla JS" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License" />
</div>

---

## 🚀 Overview

A professional-grade **Monte Carlo & Black-Scholes Option Pricing Simulator** tailored for the Indian Equity Market (NSE). This engine handles high-volatility NSE stocks using a specialized **Trading Day Convention (250 days)** and real-time data synchronization with Yahoo Finance.

> [!IMPORTANT]
> **Engineered by Jay C**
> This project fuses financial engineering with a premium glassmorphism UI to provide institutional-grade insights to retail traders.

---

## ✨ Key Features

- **🎯 Precision Inputs**: Control parameters via manual keyboard entry, interactive sliders, or +/- steppers.
- **🇮🇳 NSE Optimized**: Toggle between **Standard (365)** and **Trader (250)** day counts to match professional NSE terminal premiums.
- **🔋 Dual Simulation Engines**:
  - **Monte Carlo (GBM)**: Simulates 50,000+ stock price paths using Geometric Brownian Motion.
  - **Black-Scholes**: Analytical closed-form pricing for instant verification.
- **📉 Live NSE Data**: Real-time ticker search and quote fetching for NIFTY 50 stocks.
- **🧠 5-Factor Greeks**: Live calculation of **Delta, Gamma, Vega, Theta, and Rho**.
- **🎨 Elite UI**: Dark glassmorphism interface with high-vibrancy Chart.js visualizations.

---

## 📂 Project Structure

```text
.
├── api/             # Vercel Serverless Functions (for production)
├── css/             # Styling (Glassmorphism UI)
├── js/              # Simulation & Math logic
│   ├── engine.js    # Monte Carlo & Black-Scholes implementations
│   ├── greeks.js    # Sensitivity calculations
│   └── ui.js        # DOM interactions and Chart.js integration
├── index.html       # Main Application Entry
├── proxy.py         # Python Local Dev Server (Live Prices)
├── server.js        # Node.js Local Dev Server (Live Prices)
└── vercel.json      # Deployment configuration
```

---

## 🛠️ Installation & Local Setup

To fetch live prices (NSE quotes), you need to run a local proxy server. You can choose either **Node.js** or **Python**.

### Option A: Node.js (Recommended)
1. Ensure you have [Node.js](https://nodejs.org/) installed (v18+).
2. Run the dev server:
   ```bash
   node server.js
   ```
3. Open `http://localhost:3000` in your browser.

### Option B: Python
1. Ensure you have Python 3.8+ installed.
2. Run the proxy server:
   ```bash
   python proxy.py
   ```
3. Open `http://localhost:3000` in your browser.

---

## 🧠 The Math Behind the Engine

### 1. Geometric Brownian Motion (Monte Carlo)
The simulator predicts future prices using the SDE:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$
We simulate thousands of paths to find the expected payoff in a risk-neutral world.

### 2. Black-Scholes Model
Used for lightning-fast comparisons:
$$C = S_0 N(d_1) - K e^{-rt} N(d_2)$$
$$P = K e^{-rt} N(-d_2) - S_0 N(-d_1)$$

---

## ⚖️ License & Credits

**Author**: [Jay C](https://github.com)
**License**: MIT

*Disclaimer: This tool is for educational purposes only. Financial markets involve significant risk. Always verify premiums with your broker before trading.*

---

<div align="center">
  <sub>Built with ❤️ for Indian Traders.</sub>
</div>
=======
# Monte-Carlo-Option-Simulator
Shows Stock Options details using Monte Carlo simulation
>>>>>>> origin/main
