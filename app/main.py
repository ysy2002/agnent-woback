"""FastAPI application entry point."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="agnent-woback",
    description="自动化工单回复 AI Agent —— 基于 OpenAI GPT-4o + LangGraph + RAG",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}
