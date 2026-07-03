"""Background autonomous goal executor."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import asyncpg

from .goals import GoalManager

logger = logging.getLogger("ndonga.autonomous")


class AutonomousExecutor:
    def __init__(self, db_pool: asyncpg.Pool, interval_seconds: int = 3600) -> None:
        self.db_pool = db_pool
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def run_once(self) -> int:
        manager = GoalManager(self.db_pool)
        goals = await manager.check_triggers()
        inserted = 0
        for goal in goals:
            agents = goal.get("related_agents") or ["nenda"]
            agent = agents[0] if agents else "nenda"
            await self.db_pool.execute(
                """
                INSERT INTO proactive_notifications (goal_id, user_id, agent, message)
                VALUES ($1, $2, $3, $4)
                """,
                goal["id"],
                goal["user_id"],
                agent,
                f"Goal trigger reached: {goal['description']}",
            )
            inserted += 1
        await self._maybe_run_learning_loop()
        return inserted

    async def _maybe_run_learning_loop(self) -> None:
        """Fire the learning loop per tenant when enough unscored traces accumulate.

        Opt-in: set ENABLE_LEARNING_LOOP=true in the environment.
        Threshold: LEARNING_LOOP_TRACE_THRESHOLD (default 20) unscored traces.
        Covers all tenants present in agent_traces — not just hapakule.
        Training is always opt-in (run_training=False here; requires manual flag).
        """
        if not os.getenv("ENABLE_LEARNING_LOOP", "").strip().lower() in {"1", "true", "yes"}:
            return
        try:
            # Resolve the ndonga root (packages/ndonga-agents/ndonga_agents/ → root)
            _root = str(Path(__file__).resolve().parent.parent.parent.parent)
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from scripts.learning_loop import run_learning_loop  # lazy import — root-level package

            threshold = int(os.getenv("LEARNING_LOOP_TRACE_THRESHOLD", "20"))
            min_quality = float(os.getenv("LEARNING_LOOP_MIN_QUALITY", "0.8"))

            # Query unscored trace counts per tenant — covers machant, kaya, alsabil, hapakule, etc.
            rows = await self.db_pool.fetch(
                """
                SELECT tenant_id, COUNT(*) AS count
                FROM agent_traces
                WHERE ai_judge_score IS NULL AND final_response IS NOT NULL
                GROUP BY tenant_id
                """
            )
            for row in rows:
                tenant_id: str = row["tenant_id"]
                count: int = row["count"]
                if count < threshold:
                    continue
                asyncio.create_task(
                    run_learning_loop(
                        tenant_id=tenant_id,
                        judge_batch_size=min(count, 50),
                        min_quality_score=min_quality,
                        create_manifest=True,
                        run_training=False,
                    )
                )
                logger.info(
                    "Learning loop triggered | tenant=%s | unscored_traces=%d",
                    tenant_id, count,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Learning loop check failed: %s", exc)

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Autonomous executor cycle failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                continue

    def start_background_tasks(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
