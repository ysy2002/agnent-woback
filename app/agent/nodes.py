"""LangGraph node implementations for the ticket-reply agent."""
from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from app.agent.prompts import CLASSIFY_PROMPT, GENERATE_REPLY_PROMPT, SCORE_PROMPT
from app.config import settings
from app.models.ticket import TicketCategory
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key, temperature=0.2)


# ---------------------------------------------------------------------------
# Node: classify
# ---------------------------------------------------------------------------

def classify_node(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the ticket into one of the predefined categories."""
    content: str = state["content"]
    prompt = CLASSIFY_PROMPT.format(content=content)
    raw = _llm.invoke(prompt).content.strip().lower()

    try:
        category = TicketCategory(raw)
    except ValueError:
        logger.warning("Unknown category '%s', defaulting to OTHER", raw)
        category = TicketCategory.OTHER

    return {**state, "category": category}


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------

def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
    """Retrieve relevant knowledge-base chunks."""
    docs = retrieve(state["content"])
    context = "\n\n".join(d.page_content for d in docs)
    references = [d.metadata.get("source", d.page_content[:60]) for d in docs]
    return {**state, "context": context, "references": references}


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

def generate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a reply using the LLM."""
    history_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in state.get("history", [])
    ) or "（无历史对话）"

    prompt = GENERATE_REPLY_PROMPT.format(
        category=state.get("category", "other"),
        content=state["content"],
        history=history_text,
        context=state.get("context", "（无相关知识库内容）"),
    )
    reply = _llm.invoke(prompt).content.strip()
    return {**state, "reply": reply}


# ---------------------------------------------------------------------------
# Node: score
# ---------------------------------------------------------------------------

def score_node(state: dict[str, Any]) -> dict[str, Any]:
    """Assign a confidence score to the generated reply."""
    prompt = SCORE_PROMPT.format(content=state["content"], reply=state["reply"])
    raw = _llm.invoke(prompt).content.strip()

    try:
        confidence = float(raw)
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        logger.warning("Could not parse confidence score '%s', defaulting to 0.5", raw)
        confidence = 0.5

    auto_sent = confidence >= settings.auto_reply_threshold
    return {**state, "confidence": confidence, "auto_sent": auto_sent}
