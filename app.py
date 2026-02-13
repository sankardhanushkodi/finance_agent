"""
Finance Multi-Agent System — Streamlit Frontend
================================================
Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Make sure project root is on path regardless of how streamlit is launched
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance Multi-Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []  # list of {ticker, shares, avg_cost}
if "graph" not in st.session_state:
    st.session_state.graph = None
if "graph_error" not in st.session_state:
    st.session_state.graph_error = None
# thread_id scopes the LangGraph MemorySaver checkpoint to this browser session.
# A new uuid here = a fresh conversation with no prior memory.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ── Graph loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising agents…")
def load_graph():
    from graph.orchestrator import build_graph
    return build_graph()


def get_graph():
    if st.session_state.graph is None and st.session_state.graph_error is None:
        try:
            st.session_state.graph = load_graph()
        except Exception as e:
            st.session_state.graph_error = str(e)
    return st.session_state.graph


# ── Helper: run graph ─────────────────────────────────────────────────────────
def run_query(query: str, portfolio: list, thread_id: str) -> tuple[str, list[str]]:
    """Run a query through the graph. Returns (final_answer, agent_steps)."""
    from graph.orchestrator import stream_graph

    graph = get_graph()
    if graph is None:
        return f"Graph initialisation error: {st.session_state.graph_error}", []

    steps = []
    final_answer = ""

    for chunk in stream_graph(graph, query, portfolio, thread_id=thread_id):
        for node_name, state_update in chunk.items():
            if node_name == "supervisor":
                nxt = state_update.get("next", "")
                if nxt and nxt != "FINISH":
                    steps.append(f"🔀 Routing to **{nxt}**")
            elif node_name in ("stock_agent", "portfolio_agent", "market_agent", "tax_agent"):
                steps.append(f"✅ **{node_name}** completed")
            elif node_name == "synthesiser":
                msgs = state_update.get("messages", [])
                if msgs:
                    final_answer = msgs[-1].content

    return final_answer, steps


# ── Portfolio chart helpers ───────────────────────────────────────────────────
def portfolio_to_df(portfolio: list) -> pd.DataFrame:
    return pd.DataFrame(portfolio) if portfolio else pd.DataFrame(
        columns=["ticker", "shares", "avg_cost"]
    )


def fetch_portfolio_values(portfolio: list) -> pd.DataFrame | None:
    """Fetch live prices and compute allocation for charting."""
    if not portfolio:
        return None
    try:
        import yfinance as yf
        rows = []
        for h in portfolio:
            ticker = h["ticker"].upper()
            try:
                price = yf.Ticker(ticker).fast_info.last_price or 0.0
            except Exception:
                price = 0.0
            value = float(h["shares"]) * price
            cost = float(h["shares"]) * float(h.get("avg_cost", 0))
            rows.append({
                "Ticker": ticker,
                "Shares": h["shares"],
                "Current Price": price,
                "Market Value": value,
                "Cost Basis": cost,
                "P&L": value - cost,
                "P&L %": round((value - cost) / cost * 100, 2) if cost else 0,
            })
        df = pd.DataFrame(rows)
        total = df["Market Value"].sum()
        df["Allocation %"] = df["Market Value"] / total * 100 if total else 0
        return df
    except Exception as e:
        st.warning(f"Could not fetch live prices: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Portfolio Manager")
    st.caption("Positions are sent to the agents automatically")

    # Add position
    with st.expander("➕ Add Position", expanded=len(st.session_state.portfolio) == 0):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_ticker = st.text_input("Ticker", placeholder="AAPL").upper()
        with col2:
            new_shares = st.number_input("Shares", min_value=0.0, step=1.0, value=10.0)
        with col3:
            new_cost = st.number_input("Avg Cost $", min_value=0.0, step=1.0, value=150.0)

        if st.button("Add", use_container_width=True):
            if new_ticker:
                # Update existing or add new
                existing = next(
                    (i for i, h in enumerate(st.session_state.portfolio)
                     if h["ticker"] == new_ticker), None
                )
                entry = {"ticker": new_ticker, "shares": new_shares, "avg_cost": new_cost}
                if existing is not None:
                    st.session_state.portfolio[existing] = entry
                    st.success(f"Updated {new_ticker}")
                else:
                    st.session_state.portfolio.append(entry)
                    st.success(f"Added {new_ticker}")
                st.rerun()

    # Show current holdings
    if st.session_state.portfolio:
        st.subheader("Holdings")
        for i, h in enumerate(st.session_state.portfolio):
            col_t, col_s, col_c, col_x = st.columns([2, 1.5, 1.5, 0.8])
            col_t.write(f"**{h['ticker']}**")
            col_s.write(f"{h['shares']} sh")
            col_c.write(f"${h['avg_cost']:.2f}")
            if col_x.button("✕", key=f"del_{i}", help="Remove"):
                st.session_state.portfolio.pop(i)
                st.rerun()

        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.portfolio = []
            st.rerun()

        # Quick chart
        st.subheader("Live Allocation")
        with st.spinner("Fetching prices…"):
            df_vals = fetch_portfolio_values(st.session_state.portfolio)
        if df_vals is not None and not df_vals.empty:
            fig_pie = px.pie(
                df_vals,
                names="Ticker",
                values="Market Value",
                hole=0.4,
                height=280,
            )
            fig_pie.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.2),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("⚙️ Config")
    provider = st.selectbox(
        "LLM Provider",
        ["anthropic", "openai"],
        index=0 if os.getenv("LLM_PROVIDER", "anthropic") == "anthropic" else 1,
    )
    os.environ["LLM_PROVIDER"] = provider

    model_default = (
        "claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o-mini"
    )
    model = st.text_input("Model", value=os.getenv("LLM_MODEL", model_default))
    os.environ["LLM_MODEL"] = model

    if st.button("🔄 Reload Graph", use_container_width=True):
        st.session_state.graph = None
        st.session_state.graph_error = None
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("💬 Memory")
    turn_count = len(st.session_state.messages) // 2
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")
    st.caption(f"Turns in memory: **{turn_count}**")
    if st.button("🆕 New Conversation", use_container_width=True, help="Clears memory and starts fresh"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📈 Finance Multi-Agent System")
st.caption(
    "Powered by LangGraph · Specialist agents: Stock · Portfolio · Market · Tax · "
    "Persistent memory across turns via MemorySaver"
)

# Portfolio summary table (if holdings present)
if st.session_state.portfolio:
    with st.expander("📋 Portfolio Summary", expanded=False):
        df_vals = fetch_portfolio_values(st.session_state.portfolio)
        if df_vals is not None:
            # Style P&L column
            def color_pnl(val):
                color = "green" if val > 0 else "red" if val < 0 else "gray"
                return f"color: {color}"

            styled = (
                df_vals.style
                .applymap(color_pnl, subset=["P&L", "P&L %"])
                .format({
                    "Current Price": "${:.2f}",
                    "Market Value": "${:,.2f}",
                    "Cost Basis": "${:,.2f}",
                    "P&L": "${:,.2f}",
                    "P&L %": "{:.2f}%",
                    "Allocation %": "{:.1f}%",
                })
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            total_val = df_vals["Market Value"].sum()
            total_pnl = df_vals["P&L"].sum()
            total_pnl_pct = total_pnl / df_vals["Cost Basis"].sum() * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Value", f"${total_val:,.2f}")
            c2.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")
            c3.metric("Positions", len(df_vals))

# Example questions
st.subheader("💬 Ask the Finance Agents")

with st.expander("📝 Example questions", expanded=False):
    examples = [
        "What is Apple's current stock price and key financials?",
        "How is the overall stock market performing today?",
        "Which sectors have performed best this month?",
        "Analyse my portfolio and suggest any rebalancing",
        "Compare NVDA and AMD – which looks more attractive?",
        "What is the VIX and what does it mean for investors?",
        "If I sell my AAPL position (100 shares, bought at $150, held 2 years), what are my taxes?",
        "Are there any tax-loss harvesting opportunities in my portfolio?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=False):
            st.session_state.pending_query = ex

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander("🔍 Agent trace", expanded=False):
                for step in msg["steps"]:
                    st.markdown(step)

# Handle example button click
if "pending_query" in st.session_state:
    prompt = st.session_state.pop("pending_query")
else:
    prompt = st.chat_input("Ask anything about stocks, markets, portfolios, or taxes…")

if prompt:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agents
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        step_display = []

        with st.spinner("Agents working…"):
            try:
                status_placeholder.info("🤖 Supervisor analysing query…")
                answer, steps = run_query(
                    prompt,
                    st.session_state.portfolio,
                    st.session_state.thread_id,
                )

                status_placeholder.empty()

                if answer:
                    st.markdown(answer)
                    if steps:
                        with st.expander("🔍 Agent trace", expanded=False):
                            for s in steps:
                                st.markdown(s)
                else:
                    answer = "I wasn't able to generate a response. Please check your API key configuration."
                    st.warning(answer)

            except Exception as e:
                status_placeholder.empty()
                answer = f"❌ Error: {str(e)}\n\nPlease check your `.env` file has valid API keys."
                st.error(answer)
                steps = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "steps": steps,
    })

# Footer
st.divider()
st.caption(
    "⚠️ This tool is for educational and informational purposes only. "
    "Nothing here constitutes financial, investment, or tax advice. "
    "Consult a licensed professional before making financial decisions."
)
