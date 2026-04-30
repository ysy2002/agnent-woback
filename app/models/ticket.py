from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TicketCategory(str, Enum):
    INQUIRY = "inquiry"        # 咨询
    COMPLAINT = "complaint"    # 投诉
    FAULT = "fault"            # 故障
    REFUND = "refund"          # 退款
    OTHER = "other"            # 其他


class TicketRequest(BaseModel):
    ticket_id: str = Field(..., description="工单唯一标识")
    user_id: str = Field(..., description="用户 ID")
    content: str = Field(..., description="工单内容")
    history: list[dict] = Field(default_factory=list, description="历史对话记录")


class TicketReply(BaseModel):
    ticket_id: str
    category: TicketCategory
    reply: str = Field(..., description="生成的回复内容")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度评分 [0, 1]")
    auto_sent: bool = Field(..., description="是否已自动发送")
    references: list[str] = Field(default_factory=list, description="引用的知识库条目")


class FeedbackRequest(BaseModel):
    ticket_id: str
    satisfied: bool
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    ticket_id: str
    message: str
