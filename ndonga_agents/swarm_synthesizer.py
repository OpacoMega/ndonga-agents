"""
Ndonga Swarm Synthesizer — dynamically generates bespoke specialist personas.

When the static 221 + 50 agent catalog doesn't have a good match for a query,
this module calls the Starlight model to synthesize a purpose-built system
prompt on the fly. Generated prompts are cached to disk so identical queries
get the pre-synthesized specialist on the next call.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("ndonga.swarm_synthesizer")

_CACHE_DIR = Path(__file__).resolve().parent.parent / "yapa-local" / "dynamic"

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


class SwarmSynthesizer:
    """Dynamically synthesizes specialist agent system prompts via Starlight."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        model_chain: list[str] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
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

    def cache_prompt(
        self,
        specialist_role: str,
        user_query: str,
        system_prompt: str,
    ) -> Path:
        """
        Persist a synthesized prompt to yapa-local/dynamic/<slug>.json.
        Returns the path it was written to.
        """
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        slug = _slugify(specialist_role)
        path = _CACHE_DIR / f"{slug}.json"
        payload: dict = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {}
        payload.update({
            "slug": slug,
            "query": specialist_role,
            "sample_user_query": user_query,
            "system_prompt": system_prompt,
            "use_count": payload.get("use_count", 0) + 1,
        })
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("cached | slug=%s | path=%s", slug, path)
        return path

    async def synthesize_and_cache(
        self,
        specialist_role: str,
        user_query: str,
    ) -> tuple[str, Path]:
        """Synthesize a system prompt and immediately cache it. Returns (prompt, cache_path)."""
        prompt = await self.synthesize_agent(specialist_role, user_query)
        path = self.cache_prompt(specialist_role, user_query, prompt)
        return prompt, path


def _slugify(text: str) -> str:
    """Turn a free-text role description into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return slug[:80].strip("_") or "agent"
