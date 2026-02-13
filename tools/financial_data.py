"""
LangChain tools that wrap yfinance for real-time financial data.
All tools are safe to call with no API key – they use Yahoo Finance.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _fmt(val, decimals: int = 2) -> str:
    f = _safe_float(val)
    return f"${f:,.{decimals}f}" if f is not None else "N/A"


# ── stock tools ───────────────────────────────────────────────────────────────

@tool
def get_stock_quote(ticker: str) -> str:
    """
    Get the current price and key stats for a stock ticker.
    Returns price, market cap, P/E ratio, 52-week range, and volume.
    """
    try:
        tk = yf.Ticker(ticker.upper())
        info = tk.info
        if not info or "regularMarketPrice" not in info:
            # fast_info fallback
            fast = tk.fast_info
            price = _safe_float(fast.last_price)
            return json.dumps({
                "ticker": ticker.upper(),
                "price": price,
                "note": "Limited data available",
            })

        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
        change = None
        change_pct = None
        if price and prev_close:
            change = price - prev_close
            change_pct = (change / prev_close) * 100

        return json.dumps({
            "ticker": ticker.upper(),
            "company": info.get("longName", ticker.upper()),
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "52wk_high": info.get("fiftyTwoWeekHigh"),
            "52wk_low": info.get("fiftyTwoWeekLow"),
            "volume": info.get("regularMarketVolume"),
            "avg_volume": info.get("averageVolume"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_stock_history(ticker: str, period: str = "1y") -> str:
    """
    Get historical OHLCV data for a ticker.
    period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    Returns summary statistics (open, close, high, low, returns).
    """
    try:
        tk = yf.Ticker(ticker.upper())
        hist = tk.history(period=period)
        if hist.empty:
            return json.dumps({"error": "No historical data found", "ticker": ticker})

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        total_return = (end_price - start_price) / start_price * 100
        daily_returns = hist["Close"].pct_change().dropna()
        volatility = float(daily_returns.std() * (252 ** 0.5) * 100)  # annualised %

        return json.dumps({
            "ticker": ticker.upper(),
            "period": period,
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
            "start_price": start_price,
            "end_price": end_price,
            "total_return_pct": round(total_return, 2),
            "high": float(hist["High"].max()),
            "low": float(hist["Low"].min()),
            "avg_volume": float(hist["Volume"].mean()),
            "annualized_volatility_pct": round(volatility, 2),
            "trading_days": len(hist),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_stock_financials(ticker: str) -> str:
    """
    Get key financial metrics for a stock: revenue, earnings, margins,
    debt ratios, return on equity, and analyst recommendations.
    """
    try:
        tk = yf.Ticker(ticker.upper())
        info = tk.info

        recs = tk.recommendations
        analyst_summary = "N/A"
        if recs is not None and not recs.empty:
            if "period" in recs.columns:
                latest = recs[recs["period"] == "0m"]
            else:
                latest = recs.tail(5)
            if not latest.empty:
                analyst_summary = latest.to_dict(orient="records")

        return json.dumps({
            "ticker": ticker.upper(),
            "company": info.get("longName"),
            "revenue": info.get("totalRevenue"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "free_cash_flow": info.get("freeCashflow"),
            "beta": info.get("beta"),
            "analyst_target_price": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationMean"),
            "analyst_summary": analyst_summary,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


# ── portfolio tools ───────────────────────────────────────────────────────────

@tool
def analyze_portfolio(holdings_json: str) -> str:
    """
    Analyze a portfolio of stock holdings.
    Input: JSON list of {ticker, shares, avg_cost} objects.
    Returns current values, allocation %, P&L, and risk metrics.
    """
    try:
        holdings = json.loads(holdings_json)
        if not holdings:
            return json.dumps({"error": "Empty portfolio"})

        rows = []
        total_cost = 0.0
        total_value = 0.0

        for h in holdings:
            ticker = h["ticker"].upper()
            shares = float(h["shares"])
            avg_cost = float(h.get("avg_cost", 0))

            tk = yf.Ticker(ticker)
            fast = tk.fast_info
            price = _safe_float(fast.last_price) or 0.0
            company = tk.info.get("longName", ticker) if price else ticker

            current_value = price * shares
            cost_basis = avg_cost * shares
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0

            total_cost += cost_basis
            total_value += current_value

            rows.append({
                "ticker": ticker,
                "company": company,
                "shares": shares,
                "current_price": price,
                "avg_cost": avg_cost,
                "current_value": current_value,
                "cost_basis": cost_basis,
                "pnl": pnl,
                "pnl_pct": round(pnl_pct, 2),
            })

        # Compute allocation %
        for r in rows:
            r["allocation_pct"] = round(r["current_value"] / total_value * 100, 2) if total_value else 0

        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

        return json.dumps({
            "holdings": rows,
            "summary": {
                "total_cost": round(total_cost, 2),
                "total_value": round(total_value, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 2),
                "num_positions": len(rows),
            },
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_portfolio_performance(holdings_json: str, period: str = "1y") -> str:
    """
    Compare portfolio performance against the S&P 500 benchmark.
    Input: JSON list of {ticker, shares, avg_cost} objects and a period (e.g. '1y').
    Returns portfolio return % vs SPY return % for the period.
    """
    try:
        holdings = json.loads(holdings_json)
        tickers = [h["ticker"].upper() for h in holdings]
        shares_map = {h["ticker"].upper(): float(h["shares"]) for h in holdings}

        all_tickers = tickers + ["SPY"]
        data = yf.download(all_tickers, period=period, auto_adjust=True, progress=False)["Close"]

        if data.empty:
            return json.dumps({"error": "Could not fetch price history"})

        data = data.dropna(how="all")
        returns = {}

        for ticker in tickers:
            if ticker in data.columns:
                col = data[ticker].dropna()
                if len(col) >= 2:
                    r = (float(col.iloc[-1]) - float(col.iloc[0])) / float(col.iloc[0]) * 100
                    returns[ticker] = round(r, 2)

        # Weighted portfolio return by starting value
        portfolio_start = 0.0
        portfolio_end = 0.0
        for ticker in tickers:
            if ticker in data.columns:
                col = data[ticker].dropna()
                if len(col) >= 2:
                    s = shares_map.get(ticker, 0)
                    portfolio_start += float(col.iloc[0]) * s
                    portfolio_end += float(col.iloc[-1]) * s

        portfolio_return = (
            (portfolio_end - portfolio_start) / portfolio_start * 100
            if portfolio_start else 0
        )

        spy_return = None
        if "SPY" in data.columns:
            spy = data["SPY"].dropna()
            if len(spy) >= 2:
                spy_return = round(
                    (float(spy.iloc[-1]) - float(spy.iloc[0])) / float(spy.iloc[0]) * 100, 2
                )

        return json.dumps({
            "period": period,
            "portfolio_return_pct": round(portfolio_return, 2),
            "benchmark_spy_return_pct": spy_return,
            "alpha_pct": round(portfolio_return - (spy_return or 0), 2),
            "individual_returns": returns,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── market tools ─────────────────────────────────────────────────────────────

@tool
def get_market_overview() -> str:
    """
    Get current market snapshot: major indices (SPY, QQQ, DIA, IWM),
    VIX (fear index), and treasury yield.
    """
    try:
        tickers = {
            "SPY": "S&P 500 ETF",
            "QQQ": "Nasdaq 100 ETF",
            "DIA": "Dow Jones ETF",
            "IWM": "Russell 2000 ETF",
            "^VIX": "VIX Fear Index",
            "^TNX": "10-Year Treasury Yield",
            "GLD": "Gold ETF",
            "USO": "Oil ETF",
        }
        result = {}
        for sym, name in tickers.items():
            try:
                tk = yf.Ticker(sym)
                fast = tk.fast_info
                price = _safe_float(fast.last_price)
                prev = _safe_float(fast.previous_close)
                change_pct = ((price - prev) / prev * 100) if (price and prev) else None
                result[sym] = {
                    "name": name,
                    "price": price,
                    "change_pct": round(change_pct, 2) if change_pct is not None else None,
                }
            except Exception:
                result[sym] = {"name": name, "price": None}

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_sector_performance(period: str = "1mo") -> str:
    """
    Get performance of the 11 S&P 500 sectors using SPDR ETFs.
    period options: 1d, 5d, 1mo, 3mo, 6mo, 1y
    """
    sector_etfs = {
        "XLK": "Technology",
        "XLV": "Healthcare",
        "XLF": "Financials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLU": "Utilities",
        "XLC": "Communication Services",
    }
    try:
        results = {}
        data = yf.download(
            list(sector_etfs.keys()), period=period, auto_adjust=True, progress=False
        )["Close"]

        for sym, sector in sector_etfs.items():
            if sym in data.columns:
                col = data[sym].dropna()
                if len(col) >= 2:
                    ret = (float(col.iloc[-1]) - float(col.iloc[0])) / float(col.iloc[0]) * 100
                    results[sector] = round(ret, 2)

        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        return json.dumps({"period": period, "sector_returns_pct": sorted_results})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── tax tools ─────────────────────────────────────────────────────────────────

@tool
def calculate_capital_gains(
    ticker: str,
    shares: float,
    avg_cost_per_share: float,
    holding_period_days: int,
) -> str:
    """
    Calculate capital gains tax estimate for selling a position.
    Uses current market price and US tax rates (short-term vs long-term).
    Provide ticker, shares, avg_cost_per_share, and holding_period_days.
    """
    try:
        tk = yf.Ticker(ticker.upper())
        price = _safe_float(tk.fast_info.last_price)
        if price is None:
            return json.dumps({"error": f"Could not get price for {ticker}"})

        cost_basis = shares * avg_cost_per_share
        current_value = shares * price
        gain = current_value - cost_basis
        is_long_term = holding_period_days >= 365

        # Simplified US rates
        st_rate = 0.37   # top short-term rate (ordinary income)
        lt_rates = [0.0, 0.15, 0.20]  # 0%, 15%, 20% based on income
        lt_rate = 0.15    # most common rate

        tax_estimate = gain * (lt_rate if is_long_term else st_rate) if gain > 0 else 0

        return json.dumps({
            "ticker": ticker.upper(),
            "shares": shares,
            "avg_cost": avg_cost_per_share,
            "current_price": price,
            "cost_basis": round(cost_basis, 2),
            "current_value": round(current_value, 2),
            "gain_loss": round(gain, 2),
            "is_long_term": is_long_term,
            "holding_period_days": holding_period_days,
            "applicable_rate": lt_rate if is_long_term else st_rate,
            "estimated_tax": round(tax_estimate, 2),
            "after_tax_proceeds": round(current_value - tax_estimate, 2),
            "note": "Tax estimate is simplified; consult a tax professional for accuracy.",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def find_tax_loss_opportunities(holdings_json: str) -> str:
    """
    Scan a portfolio for tax-loss harvesting opportunities.
    Input: JSON list of {ticker, shares, avg_cost} objects.
    Returns positions with unrealised losses that could offset gains.
    """
    try:
        holdings = json.loads(holdings_json)
        losers = []
        for h in holdings:
            ticker = h["ticker"].upper()
            shares = float(h["shares"])
            avg_cost = float(h.get("avg_cost", 0))

            try:
                price = _safe_float(yf.Ticker(ticker).fast_info.last_price) or 0.0
            except Exception:
                price = 0.0

            current_value = price * shares
            cost_basis = avg_cost * shares
            pnl = current_value - cost_basis

            if pnl < 0:
                losers.append({
                    "ticker": ticker,
                    "shares": shares,
                    "current_price": price,
                    "avg_cost": avg_cost,
                    "unrealized_loss": round(pnl, 2),
                    "loss_pct": round(pnl / cost_basis * 100, 2) if cost_basis else 0,
                })

        losers.sort(key=lambda x: x["unrealized_loss"])
        total_harvestable = sum(p["unrealized_loss"] for p in losers)

        return json.dumps({
            "tax_loss_candidates": losers,
            "total_harvestable_loss": round(total_harvestable, 2),
            "num_candidates": len(losers),
            "note": (
                "Selling these positions realizes losses that can offset capital gains. "
                "Be aware of the 30-day wash-sale rule when repurchasing similar securities."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── exported collections ──────────────────────────────────────────────────────

STOCK_TOOLS = [get_stock_quote, get_stock_history, get_stock_financials]
PORTFOLIO_TOOLS = [analyze_portfolio, get_portfolio_performance, get_stock_quote]
MARKET_TOOLS = [get_market_overview, get_sector_performance, get_stock_history]
TAX_TOOLS = [calculate_capital_gains, find_tax_loss_opportunities]

ALL_TOOLS = list({t.name: t for t in STOCK_TOOLS + PORTFOLIO_TOOLS + MARKET_TOOLS + TAX_TOOLS}.values())
