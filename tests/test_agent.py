"""Unit tests for agent nodes and API routes."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.models.ticket import TicketCategory


# ---------------------------------------------------------------------------
# classify_node tests
# ---------------------------------------------------------------------------

class TestClassifyNode:
    @patch("app.agent.nodes._llm")
    def test_known_category(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="refund")
        from app.agent.nodes import classify_node

        result = classify_node({"content": "我要退款"})
        assert result["category"] == TicketCategory.REFUND

    @patch("app.agent.nodes._llm")
    def test_unknown_category_defaults_to_other(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="unknown_xyz")
        from app.agent.nodes import classify_node

        result = classify_node({"content": "随机内容"})
        assert result["category"] == TicketCategory.OTHER


# ---------------------------------------------------------------------------
# score_node tests
# ---------------------------------------------------------------------------

class TestScoreNode:
    @patch("app.agent.nodes._llm")
    def test_score_within_bounds(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="0.92")
        from app.agent.nodes import score_node

        result = score_node({"content": "问题", "reply": "回复"})
        assert abs(result["confidence"] - 0.92) < 1e-6

    @patch("app.agent.nodes._llm")
    def test_score_clipped_to_range(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="1.5")
        from app.agent.nodes import score_node

        result = score_node({"content": "问题", "reply": "回复"})
        assert result["confidence"] == 1.0

    @patch("app.agent.nodes._llm")
    def test_invalid_score_defaults_to_half(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="not_a_number")
        from app.agent.nodes import score_node

        result = score_node({"content": "问题", "reply": "回复"})
        assert result["confidence"] == 0.5

    @patch("app.agent.nodes._llm")
    def test_auto_sent_flag(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="0.95")
        from app.agent.nodes import score_node

        result = score_node({"content": "问题", "reply": "回复"})
        # default threshold is 0.80
        assert result["auto_sent"] is True


# ---------------------------------------------------------------------------
# FastAPI route tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("app.api.routes.run_agent")
def test_reply_ticket(mock_run, client):
    from app.models.ticket import TicketReply

    mock_run.return_value = TicketReply(
        ticket_id="T001",
        category=TicketCategory.INQUIRY,
        reply="您好，感谢您的咨询。",
        confidence=0.90,
        auto_sent=True,
        references=[],
    )
    payload = {
        "ticket_id": "T001",
        "user_id": "U001",
        "content": "我想了解产品功能",
        "history": [],
    }
    resp = client.post("/api/v1/tickets/reply", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticket_id"] == "T001"
    assert data["auto_sent"] is True


def test_feedback(client):
    payload = {"ticket_id": "T001", "satisfied": True, "comment": "很满意"}
    resp = client.post("/api/v1/tickets/feedback", json=payload)
    assert resp.status_code == 200
    assert "反馈已记录" in resp.json()["message"]
