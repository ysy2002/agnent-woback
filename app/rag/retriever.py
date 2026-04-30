"""RAG retriever: fetch top-k relevant chunks for a query."""
from __future__ import annotations

from langchain_core.documents import Document

from app.rag.vectorstore import get_vectorstore


def retrieve(query: str, k: int = 4) -> list[Document]:
    """Return the top-k most relevant documents for *query*."""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)
