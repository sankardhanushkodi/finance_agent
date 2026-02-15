---
title: Finance Multi-Agent System
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Finance Multi-Agent System

A multi-agent system built with **LangGraph** that answers finance questions and analyses portfolios. Four specialist agents collaborate under a Supervisor orchestrator.

## Architecture

```
User Query
    ↓
[Supervisor] ── routes ──→ [stock_agent]
                      ──→ [portfolio_agent]
                      ──→ [market_agent]
                      ──→ [tax_agent]
    ↑ results ──────────────────────┘
    ↓
[Synthesiser] → Final Answer
```

### Agents

| Agent | Responsibilities | Tools |
|---|---|---|
| **stock_agent** | Price quotes, history, fundamentals, analyst ratings | `get_stock_quote`, `get_stock_history`, `get_stock_financials` |
| **portfolio_agent** | Allocation, P&L, performance vs benchmark, risk | `analyze_portfolio`, `get_portfolio_performance` |
| **market_agent** | Indices, VIX, sectors, macro trends | `get_market_overview`, `get_sector_performance` |
| **tax_agent** | Capital gains estimates, tax-loss harvesting | `calculate_capital_gains`, `find_tax_loss_opportunities` |

All financial data is fetched **live** from Yahoo Finance via `yfinance` — no paid API key needed.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your LLM provider
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY

# 3. Launch the Streamlit app
streamlit run app.py
```

## Configuration (`.env`)

```
# Use Anthropic Claude (recommended)
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001   # fast and cheap; or claude-sonnet-4-5-20250929

# — OR — use OpenAI
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

## Command-line test

```bash
python test_graph.py
```

## Example questions

- *"What is Apple's P/E ratio and how does it compare to the sector?"*
- *"How is the market performing today? Show me sector performance."*
- *"Analyse my portfolio — am I over-concentrated?"*
- *"If I sell my NVDA (bought at $400, 50 shares, held 18 months), what's my tax?"*
- *"Find tax-loss harvesting opportunities in my portfolio."*

## Deploy to Hugging Face Spaces

The project ships with a `Dockerfile` targeting HF Spaces (Docker SDK).

1. Create a new Space → **Docker** SDK at huggingface.co/new-space
2. Push the repo:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
   git push hf main
   ```
3. Add your API key under **Settings → Repository Secrets**:
   | Secret name | Value |
   |---|---|
   | `ANTHROPIC_API_KEY` | `sk-ant-...` |
   | `LLM_PROVIDER` | `anthropic` |
   | `LLM_MODEL` | `claude-haiku-4-5-20251001` |

HF Spaces will build the image automatically and expose the app on port 7860.

## Local Docker build

```bash
docker build -t finance-agent .
docker run -p 7860:7860 \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    -e LLM_PROVIDER=anthropic \
    finance-agent
# → open http://localhost:7860
```

## Disclaimer

This system is for **educational and informational purposes only**. Nothing constitutes personalised financial, investment, or tax advice. Consult a licensed professional before making financial decisions.
