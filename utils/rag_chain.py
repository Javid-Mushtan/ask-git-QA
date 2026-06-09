from __future__ import annotations

import os
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from utils.config import TOP_K
from utils.vector_database import load_vector_store, get_embeddings
from utils.llm import get_llm
from utils.prompt import prompt

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
RRF_K = 60


def _reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k: int = RRF_K,
) -> List[Document]:
    """
    Merge two ranked lists (dense + BM25) using Reciprocal Rank Fusion.
    Returns a deduplicated list ordered by fused score (highest first).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


# ---------------------------------------------------------------------------
# BM25 retriever built from the Chroma collection
# ---------------------------------------------------------------------------

def _build_bm25(all_docs: List[Document]) -> Tuple[BM25Okapi, List[Document]]:
    """Tokenise all stored docs and return a BM25 index + the doc list."""
    tokenised = [doc.page_content.lower().split() for doc in all_docs]
    return BM25Okapi(tokenised), all_docs


def _bm25_search(
    query: str,
    bm25: BM25Okapi,
    all_docs: List[Document],
    top_k: int,
) -> List[Document]:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [all_docs[i] for i in top_indices]

def ask_question(
    query: str,
    repo_id: str,
    chat_history_dicts=None,
) -> Tuple[str, List[str]]:
    """
    Run hybrid (dense + BM25) retrieval, fuse results with RRF, then
    generate an answer with the LLM.

    Returns (answer, [unique source paths]).
    """
    db = load_vector_store(repo_id)

    # ── 1. Dense retrieval (cosine similarity via Chroma) ──────────────────
    dense_results = db.similarity_search(query, k=TOP_K)

    # ── 2. BM25 sparse retrieval over ALL stored chunks ────────────────────
    # Fetch everything from the collection (Chroma stores them in memory).
    collection = db._collection.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(collection["documents"], collection["metadatas"])
    ]

    bm25_index, all_docs = _build_bm25(all_docs)
    sparse_results = _bm25_search(query, bm25_index, all_docs, top_k=TOP_K)

    # ── 3. Fuse rankings with Reciprocal Rank Fusion ───────────────────────
    fused_docs = _reciprocal_rank_fusion(dense_results, sparse_results)
    # Keep only the top-K after fusion
    context_docs = fused_docs[:TOP_K]

    # ── 4. Build context string and call LLM ──────────────────────────────
    context = "\n\n".join(
        f"# {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in context_docs
    )

    chain = prompt | get_llm()
    response = chain.invoke({"context": context, "input": query})
    answer = response.content if hasattr(response, "content") else str(response)

    sources = list({doc.metadata.get("source", "unknown") for doc in context_docs})
    return answer, sources


# ---------------------------------------------------------------------------
# Simple chain helper (kept for any code that imports get_qa_chain)
# ---------------------------------------------------------------------------

def get_qa_chain(repo_id: str):
    """
    Lightweight wrapper that returns a callable mimicking the old
    `rag_chain.invoke({"input": query})` interface.
    Used by routes that haven't been migrated to ask_question yet.
    """
    class _HybridChain:
        def invoke(self, payload: dict) -> dict:
            answer, sources = ask_question(payload["input"], repo_id)
            return {"answer": answer, "sources": sources}

    return _HybridChain()