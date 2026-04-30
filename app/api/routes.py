"""FastAPI route definitions."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.agent.graph import run_agent
from app.models.ticket import FeedbackRequest, FeedbackResponse, TicketReply, TicketRequest
from app.rag.vectorstore import add_documents

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory feedback store (replace with a DB in production)
_feedback_store: dict[str, dict] = {}


@router.post("/tickets/reply", response_model=TicketReply, summary="处理工单并生成回复")
async def reply_ticket(req: TicketRequest) -> TicketReply:
    """Run the agent pipeline and return the generated reply."""
    try:
        result = run_agent(
            ticket_id=req.ticket_id,
            user_id=req.user_id,
            content=req.content,
            history=req.history,
        )
    except Exception as exc:
        logger.exception("Agent failed for ticket %s", req.ticket_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/tickets/feedback", response_model=FeedbackResponse, summary="提交工单满意度反馈")
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    _feedback_store[req.ticket_id] = {
        "satisfied": req.satisfied,
        "comment": req.comment,
    }
    logger.info("Feedback recorded for ticket %s: satisfied=%s", req.ticket_id, req.satisfied)
    return FeedbackResponse(ticket_id=req.ticket_id, message="反馈已记录，感谢您的评价！")


@router.post("/knowledge/add", summary="向知识库添加文档")
async def add_knowledge(texts: list[str]) -> dict:
    """Ingest plain-text documents into the Chroma knowledge base."""
    if not texts:
        raise HTTPException(status_code=400, detail="texts 列表不能为空")
    add_documents(texts)
    return {"message": f"成功添加 {len(texts)} 条知识库文档"}
