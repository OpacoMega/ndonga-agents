"""
Ndonga Swarm Synthesizer — dynamically generates bespoke specialist personas.

When the static 221 + 50 agent catalog doesn't have a good match for a query,
this module calls the Starlight model to synthesize a purpose-built system
prompt on the fly. Generated prompts are cached to the Neon DB so they survive
Fly.io deploys — identical queries get the pre-synthesized specialist on the
next call without a round-trip to the LLM.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger("ndonga.swarm_synthesizer")

_ARCHITECT_SYSTEM = """\
You are an Expert Agent Architect for Ndonga, Africa's premier AI platform. \
Your job is to design authoritative AI specialist personas.

When given a user query and its domain context, you create a comprehensive \
System Prompt that will be used to make an LLM behave as the perfect specialist \
for that exact situation.

Your output must be ONLY the System Prompt itself — no preamble, no labels, \
no meta-commentary. Start directly with "You are [Agent Name]..."

Requirements for every System Prompt you generate:
1. Give the agent a memorable, descriptive name tied to its specialisation.
2. Be highly specific to the domain. Generic agents help no one.
3. When the query relates to Kenya or East Africa, ground the agent deeply in:
   - Local institutions (KRA, NTSA, eCitizen, NSE, CBK, NLC, NHIF/SHIF, etc.)
   - Local law (Employment Act 2007, Land Act 2012, Companies Act 2015, etc.)
   - Local financial instruments (T-bills/bonds via CBK DhowCSD, M-Pesa, SACCOs)
   - Languages: respond naturally in English; use Swahili/Sheng phrases where authentic.
4. Include the agent's core expertise areas, knowledge base, and analytical style.
5. State clear boundaries: what the agent will NOT do, when to recommend a professional.
6. Target length: 450–600 words.\
"""

_ARCHITECT_USER_TMPL = """\
Design a specialist AI agent for the following:

SPECIALIST ROLE NEEDED: {specialist_role}

USER'S ACTUAL QUESTION: {user_query}

Generate the complete System Prompt now.\
"""


# Free-tier model fallback chain — same pool as ndonga_engine.py MODEL_CHAIN.
# Paid models (glm) are last-resort only; synthesis prompts are small and cheap.
_SYNTHESIS_MODEL_CHAIN: list[str] = [
    os.getenv("NDONGA_STARLIGHT_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "z-ai/glm-4.7-flash",
]


def _compute_hash(specialist_role: str, user_query: str) -> str:
    """SHA-256 of 'specialist_role|user_query[:200]', returned as hex."""
    key = f"{specialist_role}|{user_query[:200]}"
    return hashlib.sha256(key.encode()).hexdigest()


class SwarmSynthesizer:
    """Dynamically synthesizes specialist agent system prompts via Starlight."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        model_chain: list[str] | None = None,
        db_pool: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.db_pool = db_pool
        # Build deduped model chain: preferred model first, then fallbacks
        preferred = model or _SYNTHESIS_MODEL_CHAIN[0]
        chain: list[str] = []
        for m in [preferred, *_SYNTHESIS_MODEL_CHAIN, *(model_chain or [])]:
            if m and m not in chain:
                chain.append(m)
        self.model_chain = chain

    async def synthesize_agent(
        self,
        specialist_role: str,
        user_query: str,
    ) -> str:
        """
        Generate a bespoke system prompt for the given specialist role and query.

        Walks self.model_chain on 429/rate-limit errors, consistent with how
        ndonga_engine.py handles free-tier rate limits.

        Returns:
            The synthesized system prompt as a plain string.

        Raises:
            RuntimeError: if every model in the chain fails.
        """
        from .llm_gateway import OpenRouterProvider, LLMGatewayError

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set — cannot synthesize agent.")

        user_content = _ARCHITECT_USER_TMPL.format(
            specialist_role=specialist_role,
            user_query=user_query,
        )
        provider = OpenRouterProvider(api_key=self.api_key)
        last_exc: Exception | None = None

        for model in self.model_chain:
            try:
                response = await provider.achat(
                    model=model,
                    messages=[
                        {"role": "system", "content": _ARCHITECT_SYSTEM},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.4,
                    max_tokens=900,
                )
                prompt = response.content.strip()
                if not prompt:
                    logger.warning("synthesis_empty_response | model=%s | role=%s", model, specialist_role)
                    continue
                logger.info(
                    "agent_synthesized | role=%s | model=%s | chars=%d",
                    specialist_role, model, len(prompt),
                )
                return prompt
            except LLMGatewayError as exc:
                last_exc = exc
                status = exc.status_code
                if status == 429 or status == 503:
                    logger.warning(
                        "synthesis_rate_limited | model=%s | status=%s | trying_next",
                        model, status,
                    )
                    continue
                # Non-retryable error (4xx other than 429) — fail immediately
                logger.error("synthesis_failed | model=%s | role=%s | %s", model, specialist_role, exc)
                raise RuntimeError(f"SwarmSynthesizer LLM error ({status}): {exc}") from exc

        raise RuntimeError(
            f"SwarmSynthesizer: all {len(self.model_chain)} models exhausted. "
            f"Last error: {last_exc}"
        ) from last_exc

    async def _save_to_db(
        self,
        specialist_role: str,
        system_prompt: str,
        query_hash: str,
    ) -> None:
        """Upsert synthesized prompt into dynamic_swarm_cache. Logs on failure, never raises."""
        if not self.db_pool:
            logger.warning("db_pool not set — dynamic prompt not persisted to DB")
            return
        try:
            await self.db_pool.execute(
                """
                INSERT INTO dynamic_swarm_cache (query_hash, agent_name, system_prompt)
                VALUES ($1, $2, $3)
                ON CONFLICT (query_hash) DO UPDATE
                    SET last_used_at = NOW()
                """,
                query_hash, specialist_role, system_prompt,
            )
            logger.debug("cached_db | role=%s | hash=%s", specialist_role, query_hash[:16])
        except Exception as exc:
            logger.error("cache_db_write_error | role=%s | %s", specialist_role, exc)

    async def synthesize_and_cache(
        self,
        specialist_role: str,
        user_query: str,
    ) -> tuple[str, str]:
        """
        Check Neon DB cache first; synthesize and persist on miss.
        Returns (system_prompt, query_hash).
        """
        query_hash = _compute_hash(specialist_role, user_query)

        if self.db_pool:
            try:
                row = await self.db_pool.fetchrow(
                    "SELECT system_prompt FROM dynamic_swarm_cache WHERE query_hash = $1",
                    query_hash,
                )
                if row:
                    await self.db_pool.execute(
                        "UPDATE dynamic_swarm_cache SET last_used_at = NOW() WHERE query_hash = $1",
                        query_hash,
                    )
                    logger.info("cache_hit_db | role=%s | hash=%s", specialist_role, query_hash[:16])
                    return row["system_prompt"], query_hash
            except Exception as exc:
                logger.warning("cache_db_read_error | role=%s | %s — synthesizing fresh", specialist_role, exc)

        prompt = await self.synthesize_agent(specialist_role, user_query)
        await self._save_to_db(specialist_role, prompt, query_hash)
        return prompt, query_hash
