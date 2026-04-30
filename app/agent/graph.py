"""LangGraph workflow definition for the ticket-reply agent."""
from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from app.agent.nodes import classify_node, retrieve_node, generate_node, score_node
from app.models.ticket import TicketCategory, TicketReply


def _build_graph() -> Any:
    graph = StateGraph(dict)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("score", score_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "score")
    graph.add_edge("score", END)

    return graph.compile()


_agent = _build_graph()


def run_agent(
    ticket_id: str,
    user_id: str,
    content: str,
    history: list[dict] | None = None,
) -> TicketReply:
    """Execute the full agent workflow and return a structured reply."""
    initial_state: dict[str, Any] = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "content": content,
        "history": history or [],
    }

    final_state = _agent.invoke(initial_state)

    return TicketReply(
        ticket_id=ticket_id,
        category=final_state.get("category", TicketCategory.OTHER),
        reply=final_state.get("reply", ""),
        confidence=final_state.get("confidence", 0.0),
        auto_sent=final_state.get("auto_sent", False),
        references=final_state.get("references", []),
    )
