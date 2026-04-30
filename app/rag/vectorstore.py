"""Chroma-backed vector store wrapper."""
from __future__ import annotations

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import settings


def get_vectorstore() -> Chroma:
    """Return (or create) the persistent Chroma vector store."""
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
    )


def add_documents(texts: list[str], metadatas: list[dict] | None = None) -> None:
    """Add plain-text documents to the knowledge base."""
    vs = get_vectorstore()
    vs.add_texts(texts, metadatas=metadatas or [{} for _ in texts])
