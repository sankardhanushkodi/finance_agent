"""
News tools — fetch and surface financial news headlines.
Uses yfinance for ticker-specific news and standard-library urllib + ElementTree
for market-wide RSS feeds. No additional API keys required.
"""
from __future__ import annotations

import json
import time
import urllib.request
import xml.etree.ElementTree as ET

from langchain_core.tools import tool


# ── RSS feed registry ──────────────────────────────────────────────────────────

_RSS_FEEDS = {
    "top_stories":  "https://finance.yahoo.com/rss/topfinstories",
    "markets":      "https://finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    "technology":   "https://finance.yahoo.com/rss/2.0/headline?s=%5ENDX&region=US&lang=en-US",
    "crypto":       "https://finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
    "economy":      "https://finance.yahoo.com/rss/2.0/headline?s=%5ETNX&region=US&lang=en-US",
    "earnings":     "https://finance.yahoo.com/rss/topfinstories",  # fallback
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; finance-news-agent/1.0)"}


def _fetch_rss(url: str, max_items: int) -> list[dict]:
    """Download and parse an RSS feed; return cleaned article dicts."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=12) as resp:
        content = resp.read()

    root = ET.fromstring(content)
    articles = []
    for item in root.iter("item"):
        title   = (item.findtext("title")   or "").strip()
        link    = (item.findtext("link")    or "").strip()
        pub     = (item.findtext("pubDate") or "").strip()
        desc    = (item.findtext("description") or "").strip()
        # Strip basic HTML tags from description
        import re
        desc = re.sub(r"<[^>]+>", " ", desc).strip()
        if len(desc) > 350:
            desc = desc[:350] + "…"
        if title:
            articles.append({"title": title, "published": pub, "summary": desc, "link": link})
        if len(articles) >= max_items:
            break
    return articles


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def get_stock_news(ticker: str, max_items: int = 8) -> str:
    """
    Fetch recent news headlines for a specific stock ticker.

    Provide the ticker symbol (e.g. 'AAPL', 'TSLA', 'NVDA').
    Returns up to max_items recent articles with title, publisher, and publish date.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker.upper().strip())
        try:
            raw_news = tk.news or []
        except Exception:
            raw_news = []

        articles = []
        for item in raw_news[:max_items]:
            pub_ts = item.get("providerPublishTime", 0)
            pub_date = (
                time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(pub_ts))
                if pub_ts else "Unknown"
            )
            articles.append({
                "title":            item.get("title", ""),
                "publisher":        item.get("publisher", ""),
                "published":        pub_date,
                "link":             item.get("link", ""),
                "related_tickers":  item.get("relatedTickers", []),
            })

        return json.dumps({
            "ticker":        ticker.upper(),
            "article_count": len(articles),
            "articles":      articles,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_market_news(category: str = "top_stories", max_items: int = 10) -> str:
    """
    Fetch recent financial market news from Yahoo Finance RSS feeds.

    category options (choose one):
      'top_stories'  – broad financial headlines (default)
      'markets'      – S&P 500 / equity market news
      'technology'   – Nasdaq / tech sector news
      'crypto'       – Bitcoin / crypto news
      'economy'      – macro / Treasury / interest-rate news
      'earnings'     – earnings-related headlines

    Returns up to max_items articles with title, publish date, and brief summary.
    """
    try:
        key = category.lower().replace(" ", "_").replace("-", "_")
        url = _RSS_FEEDS.get(key, _RSS_FEEDS["top_stories"])
        articles = _fetch_rss(url, max_items)
        return json.dumps({
            "category":      category,
            "article_count": len(articles),
            "articles":      articles,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "category": category})


@tool
def get_trending_tickers_news(max_tickers: int = 5) -> str:
    """
    Fetch today's most-active / trending tickers and one headline each.

    Returns the top trending US equities from Yahoo Finance along with
    a recent news headline for each ticker, giving a quick pulse of what
    the market is focused on right now.
    """
    try:
        import yfinance as yf
        trending = yf.screener.screen("most_actives", size=max_tickers)
        quotes = trending.get("quotes", []) if isinstance(trending, dict) else []

        results = []
        for q in quotes[:max_tickers]:
            sym = q.get("symbol", "")
            name = q.get("longName") or q.get("shortName") or sym
            price = q.get("regularMarketPrice")
            change_pct = q.get("regularMarketChangePercent")

            # Grab one headline
            headline = ""
            try:
                news = yf.Ticker(sym).news or []
                if news:
                    headline = news[0].get("title", "")
            except Exception:
                pass

            results.append({
                "ticker":       sym,
                "name":         name,
                "price":        price,
                "change_pct":   round(change_pct, 2) if change_pct is not None else None,
                "top_headline": headline,
            })

        return json.dumps({
            "trending_count": len(results),
            "trending":       results,
            "note":           "Prices are real-time or 15-min delayed.",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_sector_news(sector: str, max_items: int = 6) -> str:
    """
    Fetch recent news for a major market sector.

    sector options: 'technology', 'healthcare', 'energy', 'financials',
    'consumer', 'industrials', 'utilities', 'materials', 'real_estate',
    'communication', 'crypto'

    Maps the sector to a representative ETF ticker (e.g. technology → XLK)
    and returns recent news headlines for that ETF.
    """
    _SECTOR_ETF = {
        "technology":    "XLK",
        "tech":          "XLK",
        "healthcare":    "XLV",
        "health":        "XLV",
        "energy":        "XLE",
        "financials":    "XLF",
        "finance":       "XLF",
        "consumer":      "XLP",
        "industrials":   "XLI",
        "utilities":     "XLU",
        "materials":     "XLB",
        "real_estate":   "XLRE",
        "realestate":    "XLRE",
        "communication": "XLC",
        "crypto":        "IBIT",
    }
    key = sector.lower().replace(" ", "_").replace("-", "_")
    etf = _SECTOR_ETF.get(key)
    if not etf:
        return json.dumps({
            "error": f"Unknown sector '{sector}'.",
            "available": list(_SECTOR_ETF.keys()),
        })
    try:
        import yfinance as yf
        news = yf.Ticker(etf).news or []
        articles = []
        for item in news[:max_items]:
            pub_ts = item.get("providerPublishTime", 0)
            pub_date = (
                time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(pub_ts))
                if pub_ts else "Unknown"
            )
            articles.append({
                "title":     item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "published": pub_date,
                "link":      item.get("link", ""),
            })
        return json.dumps({
            "sector":        sector,
            "etf_proxy":     etf,
            "article_count": len(articles),
            "articles":      articles,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "sector": sector})


# ── Exported collection ────────────────────────────────────────────────────────

NEWS_TOOLS = [
    get_stock_news,
    get_market_news,
    get_trending_tickers_news,
    get_sector_news,
]
