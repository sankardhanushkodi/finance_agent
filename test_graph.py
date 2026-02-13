"""
Quick CLI smoke-test for the finance multi-agent graph.
Run: python test_graph.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from graph.orchestrator import build_graph, stream_graph

SAMPLE_PORTFOLIO = [
    {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
    {"ticker": "MSFT", "shares": 5,  "avg_cost": 300.0},
    {"ticker": "NVDA", "shares": 3,  "avg_cost": 500.0},
]

QUERIES = [
    ("Stock Q&A",      "What is Apple's current stock price and P/E ratio?",  []),
    ("Market overview","How is the overall market performing today?",          []),
    ("Portfolio",      "Analyse my portfolio and give me a summary.",          SAMPLE_PORTFOLIO),
]


def main():
    print("Building graph…")
    graph = build_graph()
    print("Graph built.\n")

    for label, query, portfolio in QUERIES:
        print(f"{'='*60}")
        print(f"TEST: {label}")
        print(f"QUERY: {query}")
        if portfolio:
            print(f"PORTFOLIO: {[h['ticker'] for h in portfolio]}")
        print("-" * 60)

        steps = []
        final = ""
        for chunk in stream_graph(graph, query, portfolio):
            for node, update in chunk.items():
                if node == "supervisor":
                    nxt = update.get("next", "")
                    if nxt:
                        steps.append(f"  → Supervisor routes to: {nxt}")
                elif node in ("stock_agent", "portfolio_agent", "market_agent", "tax_agent"):
                    steps.append(f"  ✓ {node} completed")
                elif node == "synthesiser":
                    msgs = update.get("messages", [])
                    if msgs:
                        final = msgs[-1].content

        for s in steps:
            print(s)
        print("\nFINAL ANSWER:")
        print(final[:1000] + ("…" if len(final) > 1000 else ""))
        print()


if __name__ == "__main__":
    main()
