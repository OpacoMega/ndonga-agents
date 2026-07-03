"""Semantic memory for Ndonga: store and retrieve relevant conversation context.

Two-tier retrieval strategy:
  1. pgvector cosine similarity — when NDONGA_EMBEDDING_MODEL is set.
  2. PostgreSQL full-text search (tsvector) — always available as fallback.

Embedding calls go to OpenRouter's /v1/embeddings endpoint using the model
named by NDONGA_EMBEDDING_MODEL (e.g. 'openai/text-embedding-3-small').
The dimension must match the pgvector column (default: 1536).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import asyncpg
import httpx

logger = logging.getLogger("ndonga.memory")

_EMBEDDING_MODEL = os.getenv("NDONGA_EMBEDDING_MODEL", "openai/text-embedding-3-small")
_EMBEDDING_DIM = int(os.getenv("NDONGA_EMBEDDING_DIM", "1536"))
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


async def _fetch_embedding(text: str) -> list[float] | None:
    """Call OpenRouter /v1/embeddings. Returns None on any failure."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{_OPENROUTER_BASE}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://yapahub.com",
                    "X-Title": "Ndonga",
                },
                json={"model": _EMBEDDING_MODEL, "input": text[:8192]},
            )
        if resp.status_code != 200:
            logger.warning("embedding_api_error | status=%d | %s", resp.status_code, resp.text[:120])
            return None
        data = resp.json()
        return (data.get("data") or [{}])[0].get("embedding")
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding_failed | model=%s | error=%s", _EMBEDDING_MODEL, exc)
        return None


async def index_turn(
    *,
    user_id: str,
    tenant_id: str,
    conversation_id: str | None,
    turn_index: int,
    role: str,
    content: str,
    db_pool: asyncpg.Pool,
) -> None:
    """Store one conversation turn in the semantic memory index.

    Embedding is computed and stored when OPENROUTER_API_KEY + NDONGA_EMBEDDING_MODEL
    are configured. Rows without embeddings are still queryable via full-text search.
    """
    if not content or not content.strip():
        return

    embedding = await _fetch_embedding(content)

    try:
        if embedding and len(embedding) == _EMBEDDING_DIM:
            embedding_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
            await db_pool.execute(
                """
                INSERT INTO conversation_embeddings
                  (user_id, tenant_id, conversation_id, turn_index, role, content, embedding, model_used)
                VALUES ($1, $2, $3::uuid, $4, $5, $6, $7::vector, $8)
                """,
                user_id, tenant_id, conversation_id, turn_index,
                role, content[:4000], embedding_str, _EMBEDDING_MODEL,
            )
        else:
            await db_pool.execute(
                """
                INSERT INTO conversation_embeddings
                  (user_id, tenant_id, conversation_id, turn_index, role, content)
                VALUES ($1, $2, $3::uuid, $4, $5, $6)
                """,
                user_id, tenant_id, conversation_id, turn_index,
                role, content[:4000],
            )
    except asyncpg.PostgresError as exc:
        logger.warning("memory_index_failed | user=%s | error=%s", user_id, exc)


async def recall(
    *,
    user_id: str,
    tenant_id: str,
    query: str,
    db_pool: asyncpg.Pool,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve semantically relevant past turns for *query*.

    Tries pgvector cosine similarity first; falls back to full-text search.
    Returns an empty list (never raises) so callers can skip context injection
    when memory is unavailable.
    """
    if not query or not query.strip():
        return []

    embedding = await _fetch_embedding(query)

    if embedding and len(embedding) == _EMBEDDING_DIM:
        try:
            embedding_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
            rows = await db_pool.fetch(
                """
                SELECT role, content,
                       1 - (embedding <=> $3::vector) AS similarity
                FROM conversation_embeddings
                WHERE user_id = $1
                  AND tenant_id = $2
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> $3::vector
                LIMIT $4
                """,
                user_id, tenant_id, embedding_str, limit,
            )
            if rows:
                return [
                    {"role": r["role"], "content": r["content"], "similarity": float(r["similarity"])}
                    for r in rows
                ]
        except asyncpg.PostgresError as exc:
            logger.debug("pgvector_recall_failed | falling_back_to_fts | %s", exc)

    # Full-text fallback
    try:
        tsquery = " | ".join(
            w for w in query.replace("'", "").split()
            if len(w) > 2
        )
        if not tsquery:
            return []
        rows = await db_pool.fetch(
            """
            SELECT role, content,
                   ts_rank(content_tsv, to_tsquery('english', $3)) AS rank
            FROM conversation_embeddings
            WHERE user_id = $1
              AND tenant_id = $2
              AND content_tsv @@ to_tsquery('english', $3)
            ORDER BY rank DESC
            LIMIT $4
            """,
            user_id, tenant_id, tsquery, limit,
        )
        return [
            {"role": r["role"], "content": r["content"], "similarity": float(r["rank"])}
            for r in rows
        ]
    except asyncpg.PostgresError as exc:
        logger.debug("fts_recall_failed | user=%s | %s", user_id, exc)
        return []


def format_memory_context(turns: list[dict[str, Any]]) -> str | None:
    """Format recalled turns into a compact context block for the system prompt."""
    if not turns:
        return None
    lines = ["[Relevant past context from this user's earlier conversations:]"]
    for t in turns:
        label = "User" if t["role"] == "user" else "Ndonga"
        lines.append(f"  {label}: {t['content'][:200]}")
    return "\n".join(lines)
