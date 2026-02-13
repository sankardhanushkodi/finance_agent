
# README.md

## Finance Q&A Multi-Agent System with LangGraph

A multi-agent system built with LangGraph for answering finance-related questions.

### Features

- **Multiple Specialized Agents**: Stock analysis, portfolio management, tax planning, and market insights
- **Agent Collaboration**: Agents coordinate through LangGraph's state management
- **Real-time Data Integration**: Connect to financial APIs for live market data
- **Context Awareness**: Maintains conversation history for coherent responses

### Architecture

```
User Query
    ↓
Router Agent
    ↓
[Stock Agent | Portfolio Agent | Tax Agent | Market Agent]
    ↓
Response Aggregator
    ↓
Final Answer
```

### Installation

```bash
pip install langgraph langchain openai
```

### Quick Start

```python
from langgraph.graph import StateGraph
from finance_agents import StockAgent, PortfolioAgent

graph = StateGraph()
# Add agent nodes and edges
# Configure routing logic
```

### Agents

- **Stock Agent**: Stock prices, technical analysis, fundamentals
- **Portfolio Agent**: Asset allocation, performance tracking
- **Tax Agent**: Tax-loss harvesting, capital gains
- **Market Agent**: Market trends, economic indicators

### Configuration

Set your API keys in `.env`:
```
OPENAI_API_KEY=your_key
FINANCIAL_API_KEY=your_key
```
