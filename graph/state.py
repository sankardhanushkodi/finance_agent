"""Shared state for the finance multi-agent graph."""
import operator
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class PortfolioHolding(TypedDict):
    ticker: str
    shares: float
    avg_cost: float  # average cost per share


class AgentState(TypedDict):
    """State passed between all nodes in the graph."""

    # Full LangChain message log – add_messages appends and deduplicates
    messages: Annotated[list, add_messages]

    # The current user question/request
    query: str

    # Portfolio holdings provided by the user (optional)
    portfolio: list[PortfolioHolding]

    # Results contributed by each specialist agent within a single turn
    agent_results: dict  # {agent_name: result_string}

    # Supervisor's routing decision for the next node
    next: str

    # Human-readable conversation history used to give LLMs cross-turn context.
    # operator.add concatenates new entries onto the existing list each turn.
    conversation_history: Annotated[list, operator.add]  # list of {"role", "content"} dicts
