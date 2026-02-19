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
            elif node_name in ("stock_agent", "portfolio_agent", "market_agent", "tax_agent", "goal_agent", "tax_education_agent", "news_agent"):
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
    "Powered by LangGraph · Agents: Stock · Portfolio · Market · Tax · Goal Planning · Tax Education · News · "
    "Persistent memory via MemorySaver"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_portfolio, tab_goals = st.tabs(["💬 Chat", "📋 Portfolio", "🎯 Goal Planner"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    with st.expander("📝 Example questions", expanded=False):
        st.caption("**Stocks & Markets**")
        stock_examples = [
            "What is Apple's current stock price and key financials?",
            "How is the overall stock market performing today?",
            "Which sectors have performed best this month?",
            "Compare NVDA and AMD – which looks more attractive?",
            "What is the VIX and what does it mean for investors?",
        ]
        for ex in stock_examples:
            if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=False):
                st.session_state.pending_query = ex

        st.caption("**Portfolio & Tax Planning**")
        portfolio_examples = [
            "Analyse my portfolio and suggest any rebalancing",
            "If I sell my AAPL position (100 shares, bought at $150, held 2 years), what are my taxes?",
            "Are there any tax-loss harvesting opportunities in my portfolio?",
        ]
        for ex in portfolio_examples:
            if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=False):
                st.session_state.pending_query = ex

        st.caption("**Goal Planning**")
        goal_examples = [
            "How much do I need to save monthly to retire at 65 with $80,000/year income?",
            "How long will it take to save $100,000 for a house down payment?",
            "What asset allocation should I use for a goal 15 years away?",
            "How big should my emergency fund be if I spend $5,000/month?",
        ]
        for ex in goal_examples:
            if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=False):
                st.session_state.pending_query = ex

        st.caption("**Tax Education**")
        tax_edu_examples = [
            "What is the difference between a Roth IRA and a Traditional IRA?",
            "How much can I contribute to my 401k in 2025?",
            "What are the 2025 federal income tax brackets for a single filer?",
            "If I earn $120,000 as a single filer, what is my marginal and effective tax rate?",
            "What is an HSA and how does the triple tax advantage work?",
            "Should I contribute to a Roth or Traditional IRA if I'm in the 22% bracket now?",
            "What is the 529 plan superfunding strategy?",
            "How does a SEP-IRA differ from a Solo 401k for self-employed individuals?",
        ]
        for ex in tax_edu_examples:
            if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=False):
                st.session_state.pending_query = ex

        st.caption("**News & Market Updates**")
        news_examples = [
            "What is in the news about Tesla today?",
            "Summarize today's top financial news",
            "What news is driving the technology sector right now?",
            "Which stocks are trending today and what are the headlines?",
            "What are the latest headlines around the Fed and interest rates?",
            "Any major earnings news or announcements today?",
            "What is happening in the crypto market today?",
            "Summarize recent news about NVIDIA and its AI outlook",
        ]
        for ex in news_examples:
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
        prompt = st.chat_input("Ask about stocks, markets, portfolios, taxes, or financial goals…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
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
                        steps = []
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Portfolio summary
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    if not st.session_state.portfolio:
        st.info("Add holdings in the sidebar to see your portfolio summary here.")
    else:
        with st.spinner("Fetching live prices…"):
            df_vals = fetch_portfolio_values(st.session_state.portfolio)

        if df_vals is not None and not df_vals.empty:
            total_val = df_vals["Market Value"].sum()
            total_pnl = df_vals["P&L"].sum()
            total_cost = df_vals["Cost Basis"].sum()
            total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Value", f"${total_val:,.2f}")
            c2.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")
            c3.metric("Cost Basis", f"${total_cost:,.2f}")
            c4.metric("Positions", len(df_vals))

            st.divider()

            def _color_pnl(val):
                color = "green" if val > 0 else "red" if val < 0 else "gray"
                return f"color: {color}"

            styled = (
                df_vals.style
                .map(_color_pnl, subset=["P&L", "P&L %"])
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

            # P&L bar chart
            fig_bar = px.bar(
                df_vals,
                x="Ticker",
                y="P&L",
                color="P&L",
                color_continuous_scale=["red", "lightgray", "green"],
                color_continuous_midpoint=0,
                title="Unrealised P&L by Position",
                labels={"P&L": "P&L ($)"},
            )
            fig_bar.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Goal Planner
# ══════════════════════════════════════════════════════════════════════════════
with tab_goals:
    st.subheader("Financial Goal Calculator")
    st.caption("Calculate savings plans and project growth — powered by the goal_agent")

    goal_type = st.selectbox(
        "Goal type",
        ["Retirement", "House Down Payment", "Education / College", "Emergency Fund", "Custom Goal"],
        key="goal_type_select",
    )

    st.divider()

    # ── Inputs vary by goal type ──────────────────────────────────────────────
    if goal_type == "Retirement":
        col1, col2, col3 = st.columns(3)
        current_age     = col1.number_input("Current age", 18, 80, 35, key="g_cur_age")
        retirement_age  = col2.number_input("Retirement age", 40, 90, 65, key="g_ret_age")
        desired_income  = col3.number_input("Desired annual income (today's $)", 10_000, 500_000, 80_000, step=5_000, key="g_income")
        col4, col5, col6 = st.columns(3)
        current_savings = col4.number_input("Current retirement savings ($)", 0, 10_000_000, 50_000, step=5_000, key="g_cur_sav")
        monthly_contrib = col5.number_input("Monthly contribution ($)", 0, 50_000, 1_000, step=100, key="g_monthly")
        return_rate     = col6.number_input("Expected annual return (%)", 1.0, 15.0, 7.0, step=0.5, key="g_return")

        query_template = (
            f"I am {current_age} years old and want to retire at {retirement_age} "
            f"with ${desired_income:,}/year income (in today's dollars). "
            f"I have ${current_savings:,} saved and contribute ${monthly_contrib:,}/month. "
            f"Assume {return_rate}% annual return and 3% inflation. "
            f"Am I on track? How much more do I need to save monthly? "
            f"Also show me a suggested asset allocation."
        )

    elif goal_type == "House Down Payment":
        col1, col2 = st.columns(2)
        house_price     = col1.number_input("Target home price ($)", 50_000, 5_000_000, 500_000, step=10_000, key="g_house")
        down_pct        = col2.slider("Down payment %", 5, 30, 20, key="g_down_pct")
        col3, col4, col5 = st.columns(3)
        years           = col3.number_input("Years to purchase", 1, 30, 5, key="g_years")
        current_savings = col4.number_input("Current savings ($)", 0, 2_000_000, 10_000, step=1_000, key="g_cur_sav2")
        return_rate     = col5.number_input("Expected annual return (%)", 1.0, 10.0, 4.5, step=0.5, key="g_return2")
        goal_amount     = house_price * down_pct / 100

        query_template = (
            f"I want to buy a ${house_price:,} home in {years} years with a {down_pct}% "
            f"down payment (${goal_amount:,.0f}). I have ${current_savings:,} saved already. "
            f"Assume {return_rate}% annual return. "
            f"How much do I need to save monthly? "
            f"Also suggest an appropriate asset allocation for this time horizon."
        )

    elif goal_type == "Education / College":
        col1, col2, col3 = st.columns(3)
        college_cost    = col1.number_input("Estimated total college cost ($)", 20_000, 500_000, 120_000, step=10_000, key="g_college")
        years           = col2.number_input("Years until enrollment", 1, 20, 10, key="g_years3")
        current_savings = col3.number_input("Current 529/education savings ($)", 0, 500_000, 5_000, step=1_000, key="g_cur_sav3")
        return_rate     = st.number_input("Expected annual return (%)", 1.0, 12.0, 6.0, step=0.5, key="g_return3")

        query_template = (
            f"I need ${college_cost:,} for college in {years} years. "
            f"I already have ${current_savings:,} in a 529 plan. "
            f"Assume {return_rate}% annual return. "
            f"How much should I save monthly? "
            f"What asset allocation is appropriate for a {years}-year education goal?"
        )

    elif goal_type == "Emergency Fund":
        col1, col2, col3 = st.columns(3)
        monthly_expenses  = col1.number_input("Monthly expenses ($)", 500, 50_000, 5_000, step=500, key="g_expenses")
        months_coverage   = col2.slider("Months of coverage", 3, 12, 6, key="g_months")
        current_emergency = col3.number_input("Current emergency savings ($)", 0, 500_000, 0, step=500, key="g_emerg")

        query_template = (
            f"I spend ${monthly_expenses:,}/month and want {months_coverage} months "
            f"of emergency fund coverage. I currently have ${current_emergency:,} saved. "
            f"How much do I need in total, what is the gap, "
            f"and how quickly can I fill it at different savings rates?"
        )

    else:  # Custom Goal
        col1, col2 = st.columns(2)
        goal_amount     = col1.number_input("Goal amount ($)", 1_000, 10_000_000, 100_000, step=5_000, key="g_custom_amt")
        years           = col2.number_input("Years to reach goal", 1, 50, 10, key="g_custom_years")
        col3, col4, col5 = st.columns(3)
        current_savings = col3.number_input("Current savings ($)", 0, 5_000_000, 0, step=1_000, key="g_custom_sav")
        monthly_contrib = col4.number_input("Monthly contribution ($)", 0, 50_000, 500, step=100, key="g_custom_monthly")
        return_rate     = col5.number_input("Expected annual return (%)", 1.0, 15.0, 7.0, step=0.5, key="g_custom_return")

        query_template = (
            f"I want to save ${goal_amount:,} in {years} years. "
            f"I have ${current_savings:,} saved and can contribute ${monthly_contrib:,}/month. "
            f"Assume {return_rate}% annual return. "
            f"How much do I need to save monthly to reach my goal? "
            f"Also project how my savings will grow year by year."
        )

    # ── Run goal calculation ──────────────────────────────────────────────────
    if st.button("📊 Calculate & Plan", type="primary", use_container_width=True):
        with st.spinner("Goal agent calculating…"):
            try:
                answer, steps = run_query(
                    query_template,
                    st.session_state.portfolio,
                    st.session_state.thread_id,
                )
                st.markdown(answer)
                if steps:
                    with st.expander("🔍 Agent trace", expanded=False):
                        for s in steps:
                            st.markdown(s)
                # Store in chat history so it appears in the Chat tab too
                st.session_state.messages.append({"role": "user", "content": query_template})
                st.session_state.messages.append({"role": "assistant", "content": answer, "steps": steps})
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # ── Local projection chart (no LLM) ──────────────────────────────────────
    st.divider()
    st.subheader("📈 Quick Projection Chart")
    st.caption("Instant chart — no agent call needed")

    qc1, qc2, qc3, qc4 = st.columns(4)
    q_initial  = qc1.number_input("Starting amount ($)", 0, 5_000_000, 10_000, step=1_000, key="qc_init")
    q_monthly  = qc2.number_input("Monthly contribution ($)", 0, 50_000, 500, step=100, key="qc_monthly")
    q_return   = qc3.number_input("Annual return (%)", 0.0, 20.0, 7.0, step=0.5, key="qc_return")
    q_years    = qc4.number_input("Years", 1, 50, 20, step=1, key="qc_years")

    r_monthly = q_return / 100 / 12
    balance = float(q_initial)
    total_contrib = float(q_initial)
    chart_data = [{"Year": 0, "Portfolio Value": balance, "Total Contributions": balance}]
    for yr in range(1, int(q_years) + 1):
        for _ in range(12):
            balance = balance * (1 + r_monthly) + q_monthly
            total_contrib += q_monthly
        chart_data.append({
            "Year": yr,
            "Portfolio Value": round(balance, 2),
            "Total Contributions": round(total_contrib, 2),
        })

    df_chart = pd.DataFrame(chart_data)
    fig_growth = px.area(
        df_chart,
        x="Year",
        y=["Portfolio Value", "Total Contributions"],
        labels={"value": "Amount ($)", "variable": ""},
        title=f"${q_initial:,} + ${q_monthly:,}/month @ {q_return}% for {q_years} years",
        color_discrete_map={"Portfolio Value": "#00b4d8", "Total Contributions": "#90e0ef"},
    )
    fig_growth.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_growth, use_container_width=True)

    final_val = df_chart["Portfolio Value"].iloc[-1]
    growth = final_val - total_contrib
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Final Value", f"${final_val:,.0f}")
    mc2.metric("Total Contributions", f"${total_contrib:,.0f}")
    mc3.metric("Growth from Returns", f"${growth:,.0f}", f"{growth/total_contrib*100:.0f}% of contributions" if total_contrib else "")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool is for educational and informational purposes only. "
    "Nothing here constitutes financial, investment, or tax advice. "
    "Consult a licensed professional before making financial decisions."
)
