"""
Finance Multi-Agent Orchestrator
=================================
Supervisor pattern using LangGraph with persistent conversation memory.

  User Query
      ↓
  [Supervisor]  ← uses MemorySaver to recall prior turns; routes to specialists
      ↓
  [stock_agent | portfolio_agent | market_agent | tax_agent]
      ↓
  [Synthesiser] → writes (question, answer) pair back into conversation_history
      ↓
  Final Response

Memory works at two levels:
  1. LangGraph MemorySaver checkpointer: persists the entire AgentState between
     invocations for the same thread_id (survives re-runs in the same process).
  2. conversation_history list: a compact (role, content) log passed to every
     LLM call so the model understands references like "what about its P/E?" or
     "compare that stock to the one we discussed earlier".
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents import (
    create_market_agent,
    create_portfolio_agent,
    create_stock_agent,
    create_tax_agent,
)
from graph.state import AgentState

load_dotenv()

# ── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm():
    """Instantiate the LLM based on env-var configuration."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
        return ChatAnthropic(model=model, temperature=0)


# ── Conversation history helpers ──────────────────────────────────────────────

def _format_history(history: list, max_turns: int = 6) -> str:
    """
    Format the last `max_turns` conversation pairs as a readable string block
    to inject into LLM prompts.  Truncates long answers so prompts stay lean.
    """
    if not history:
        return ""

    recent = history[-(max_turns * 2):]  # each turn = 2 entries (user + assistant)
    lines = ["\n\n### Conversation history (most recent turns):"]
    for entry in recent:
        role_label = "User" if entry["role"] == "user" else "Assistant"
        content = entry["content"]
        # Truncate very long assistant answers to keep prompt size reasonable
        if entry["role"] == "assistant" and len(content) > 400:
            content = content[:400] + "… [truncated]"
        lines.append(f"{role_label}: {content}")
    return "\n".join(lines)


# ── Supervisor ────────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are the orchestrator of a finance multi-agent system.

You have four specialist agents available:
- **stock_agent**     – individual stock prices, history, fundamentals, analyst ratings
- **portfolio_agent** – portfolio composition, performance vs benchmark, risk analysis
- **market_agent**    – market overview, sector rotation, macro trends, VIX, yields
- **tax_agent**       – capital gains calculation, tax-loss harvesting, wash-sale rules

Your job:
1. Read the current user question AND the conversation history to understand context.
2. Resolve any pronouns or references (e.g. "it", "that stock", "the one we discussed")
   using the conversation history before deciding which agent to call.
3. Decide which agent(s) should handle the question.
4. Once all needed agents have responded, set next = "FINISH" to produce the final answer.

Routing rules:
- For questions about a specific stock → stock_agent
- For questions about a portfolio (holdings provided) → portfolio_agent
- For questions about the overall market, sectors, or macro → market_agent
- For questions about taxes, capital gains, or tax-loss harvesting → tax_agent
- For complex questions spanning multiple domains, call agents in sequence
- For pure follow-up questions that need no new data (e.g. "explain that more") → FINISH

Always respond with just the agent name or 'FINISH'."""


AGENT_NAMES = ["stock_agent", "portfolio_agent", "market_agent", "tax_agent"]


def make_supervisor_node(llm):
    """Create the supervisor node that routes using conversation history for context."""

    options = AGENT_NAMES + ["FINISH"]
    routing_instruction = (
        "\n\nRespond with ONLY the name of the next agent to call, or 'FINISH'. "
        "Choose from: " + ", ".join(options)
    )

    def supervisor_node(state: AgentState):
        history_block = _format_history(state.get("conversation_history") or [])

        results_context = ""
        if state.get("agent_results"):
            results_context = "\n\n### Results collected so far:\n"
            for agent, result in state["agent_results"].items():
                results_context += f"\n**{agent}**:\n{result}\n"

        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(
                content=(
                    f"Current question: {state['query']}"
                    f"{history_block}"
                    f"{results_context}"
                    f"{routing_instruction}"
                )
            ),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        next_agent = "FINISH"
        for option in options:
            if option.lower() in content.lower():
                next_agent = option
                break

        return {"next": next_agent}

    return supervisor_node


# ── Synthesiser ───────────────────────────────────────────────────────────────

SYNTHESISER_SYSTEM = """You are a knowledgeable finance assistant providing the final answer
to the user. You have access to the full conversation history and data just collected by
specialist agents. Your response should:
- Directly answer the current question
- Reference prior conversation context where relevant (e.g. "As we discussed, AAPL …")
- Be clear, well-structured, and use markdown formatting
- Avoid repeating information already given unless the user is asking for a recap
- If no new data was collected, answer from context/knowledge and note it"""


def make_synthesiser_node(llm):
    """Synthesise a final answer and persist it into conversation_history."""

    def synthesiser_node(state: AgentState):
        history_block = _format_history(state.get("conversation_history") or [])

        results_context = ""
        if state.get("agent_results"):
            results_context = "\n\n### Data collected from specialist agents:\n"
            for agent, result in state["agent_results"].items():
                results_context += f"\n**{agent}**:\n{result}\n"

        messages = [
            SystemMessage(content=SYNTHESISER_SYSTEM),
            HumanMessage(
                content=(
                    f"Current question: {state['query']}"
                    f"{history_block}"
                    f"{results_context}"
                    "\n\nProvide the final answer now."
                )
            ),
        ]

        response = llm.invoke(messages)
        final = response.content

        # Persist this turn into conversation_history (operator.add appends the list)
        new_history_entries = [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": final},
        ]

        return {
            "messages": [AIMessage(content=final)],
            "conversation_history": new_history_entries,
        }

    return synthesiser_node


# ── Agent wrapper nodes ───────────────────────────────────────────────────────

def make_agent_node(agent, agent_name: str):
    """
    Wrap a specialist agent. Injects conversation history so the agent can
    resolve references like 'that stock' or 'my largest position'.
    """

    def agent_node(state: AgentState):
        import json

        portfolio_note = ""
        if state.get("portfolio"):
            portfolio_note = (
                f"\n\n[Portfolio context: {json.dumps(state['portfolio'])}]"
            )

        history_note = ""
        history = state.get("conversation_history") or []
        if history:
            history_note = _format_history(history, max_turns=4)

        messages = [HumanMessage(content=state["query"] + portfolio_note + history_note)]
        result = agent.invoke({"messages": messages})

        last_msg = result["messages"][-1]
        result_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        current_results = dict(state.get("agent_results") or {})
        current_results[agent_name] = result_text

        return {"agent_results": current_results}

    return agent_node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """Construct and compile the multi-agent LangGraph with MemorySaver."""
    llm = _build_llm()

    stock = create_stock_agent(llm)
    portfolio = create_portfolio_agent(llm)
    market = create_market_agent(llm)
    tax = create_tax_agent(llm)

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", make_supervisor_node(llm))
    workflow.add_node("synthesiser", make_synthesiser_node(llm))
    workflow.add_node("stock_agent", make_agent_node(stock, "stock_agent"))
    workflow.add_node("portfolio_agent", make_agent_node(portfolio, "portfolio_agent"))
    workflow.add_node("market_agent", make_agent_node(market, "market_agent"))
    workflow.add_node("tax_agent", make_agent_node(tax, "tax_agent"))

    workflow.add_edge(START, "supervisor")

    def route_supervisor(state: AgentState) -> str:
        nxt = state.get("next", "FINISH")
        return "synthesiser" if nxt == "FINISH" else nxt

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "stock_agent": "stock_agent",
            "portfolio_agent": "portfolio_agent",
            "market_agent": "market_agent",
            "tax_agent": "tax_agent",
            "synthesiser": "synthesiser",
        },
    )

    for agent_name in AGENT_NAMES:
        workflow.add_edge(agent_name, "supervisor")

    workflow.add_edge("synthesiser", END)

    # MemorySaver persists state between invocations for the same thread_id
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ── Convenience runner ────────────────────────────────────────────────────────

def stream_graph(
    graph,
    query: str,
    portfolio: list | None = None,
    thread_id: str = "default",
):
    """
    Run the graph for one turn and yield state-update chunks.

    The thread_id identifies the conversation. Using the same thread_id across
    calls causes LangGraph to load the prior checkpoint (conversation_history,
    messages, etc.) and continue the conversation from where it left off.

    Use a new uuid for thread_id to start a fresh conversation.
    """
    # On each turn we supply query/portfolio/agent_results/next.
    # - messages:              add_messages appends the new HumanMessage to prior ones
    # - conversation_history:  operator.add appends; we send [] here (synthesiser writes it)
    # - agent_results / next:  no reducer → overwritten each turn (correct)
    input_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "portfolio": portfolio or [],
        "agent_results": {},
        "next": "",
        "conversation_history": [],  # synthesiser will append this turn's entries
    }
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in graph.stream(input_state, config=config, stream_mode="updates"):
        yield chunk
