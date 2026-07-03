"""Persistent user goal management."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg
from pydantic import BaseModel, Field


class GoalCreate(BaseModel):
    user_id: str
    description: str
    target_value: float | None = None
    deadline: datetime | None = None
    related_agents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoalProgress(BaseModel):
    goal_id: str
    current_value: float
    status: str | None = None


class GoalManager:
    def __init__(self, db_pool: asyncpg.Pool) -> None:
        self.db_pool = db_pool

    async def create(self, request: GoalCreate) -> dict[str, Any]:
        row = await self.db_pool.fetchrow(
            """
            INSERT INTO user_goals (
              user_id, description, target_value, deadline, related_agents, metadata
            )
            VALUES ($1, $2, $3, $4, $5::text[], $6::jsonb)
            RETURNING *
            """,
            request.user_id,
            request.description,
            request.target_value,
            request.deadline,
            request.related_agents,
            json.dumps(request.metadata),
        )
        return dict(row)

    async def update_progress(self, progress: GoalProgress) -> dict[str, Any] | None:
        status_expr = progress.status
        row = await self.db_pool.fetchrow(
            """
            UPDATE user_goals
            SET current_value = $1,
                status = COALESCE($2, status),
                updated_at = NOW(),
                completed_at = CASE WHEN COALESCE($2, status) = 'completed' THEN NOW() ELSE completed_at END
            WHERE id = $3::uuid
            RETURNING *
            """,
            progress.current_value,
            status_expr,
            progress.goal_id,
        )
        return dict(row) if row else None

    async def get_active_goals(self, user_id: str) -> list[dict[str, Any]]:
        rows = await self.db_pool.fetch(
            """
            SELECT *
            FROM user_goals
            WHERE user_id = $1 AND status = 'active'
            ORDER BY created_at DESC
            """,
            user_id,
        )
        return [dict(row) for row in rows]

    async def check_triggers(self) -> list[dict[str, Any]]:
        rows = await self.db_pool.fetch(
            """
            SELECT *
            FROM user_goals
            WHERE status = 'active'
              AND (
                (target_value IS NOT NULL AND current_value >= target_value)
                OR (deadline IS NOT NULL AND deadline <= NOW() + INTERVAL '24 hours')
              )
            ORDER BY updated_at ASC
            LIMIT 100
            """
        )
        return [dict(row) for row in rows]
