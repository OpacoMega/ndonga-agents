"""
Ndonga Swarm Router — routes a specialist query through the 4-tier agent catalog.

Routing stages (in priority order):
  1. Yapa-local (Tier 2): African-contextual agents get first look + a priority boost.
  2. Upstream (Tier 3): 221 generic domain specialists as the broad fallback.
  3. Dynamic synthesis (Tier 4): If confidence is below threshold, the SwarmSynthesizer
     generates a bespoke persona on the fly.

The router does NOT call an LLM for intent classification — it uses keyword
overlap scoring, which is deterministic, zero-latency, and zero-cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from .specialist_index import (
    CONFIDENCE_THRESHOLD,
    SpecialistEntry,
    search_with_confidence,
    search_dynamic_cache,
    load_specialist_prompt,
)

logger = logging.getLogger("ndonga.swarm_router")

RoutingType = Literal["yapa-local", "upstream", "dynamic-cache", "dynamic-synthesize"]


@dataclass
class RoutingDecision:
    routing_type: RoutingType
    specialist: SpecialistEntry | None
    system_prompt: str | None
    confidence: float
    cached_entry: dict | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def needs_synthesis(self) -> bool:
        return self.routing_type == "dynamic-synthesize"

    @property
    def has_prompt(self) -> bool:
        return bool(self.system_prompt)


class SwarmRouter:
    """
    Keyword-based 3-stage router for the Ndonga Omni-Swarm.

    Stage 1: Search yapa-local (Tier 2) specialists with confidence scoring.
    Stage 2: Search upstream (Tier 3) specialists with confidence scoring.
    Stage 3: Check the dynamic cache for a previously synthesized match.
    Stage 4: Signal that a SwarmSynthesizer call is needed.

    The router picks the highest-confidence result across all stages and
    returns a RoutingDecision that tells the ConsultSpecialistTool what to do.
    """

    COMPOSITE_THRESHOLD: float = 0.55  # above this, try to find a second agent

    def route(self, specialist_role: str) -> RoutingDecision:
        """
        Synchronous route — no LLM calls, purely keyword-based.

        Returns a RoutingDecision. If `needs_synthesis` is True, the caller
        must invoke SwarmSynthesizer to get the system prompt.
        """
        # Search static catalog (yapa-local gets a 0.05 boost inside search_with_confidence)
        results = search_with_confidence(specialist_role, top_k=5)

        if results:
            best_conf, best_entry = results[0]
            if best_conf >= CONFIDENCE_THRESHOLD:
                system_prompt = load_specialist_prompt(best_entry.path)
                routing_type: RoutingType = (
                    "yapa-local" if best_entry.origin == "yapa-local" else "upstream"
                )
                logger.info(
                    "routed_static | role=%s | agent=%s | origin=%s | confidence=%.2f",
                    specialist_role, best_entry.name, best_entry.origin, best_conf,
                )
                return RoutingDecision(
                    routing_type=routing_type,
                    specialist=best_entry,
                    system_prompt=system_prompt,
                    confidence=best_conf,
                    metadata={
                        "runner_up": results[1][1].name if len(results) > 1 else None,
                        "runner_up_confidence": results[1][0] if len(results) > 1 else 0.0,
                    },
                )
            # Below threshold — record the best static match for logging
            static_best = (best_conf, best_entry)
        else:
            static_best = (0.0, None)

        # Check dynamic cache
        cached = search_dynamic_cache(specialist_role)
        if cached:
            system_prompt = cached.get("system_prompt", "")
            logger.info(
                "routed_dynamic_cache | role=%s | slug=%s",
                specialist_role, cached.get("slug"),
            )
            return RoutingDecision(
                routing_type="dynamic-cache",
                specialist=None,
                system_prompt=system_prompt,
                confidence=CONFIDENCE_THRESHOLD,  # cache hit = at-threshold confidence
                cached_entry=cached,
                metadata={"slug": cached.get("slug")},
            )

        # No static match above threshold, no cache hit → need synthesis
        logger.info(
            "routed_dynamic_synthesize | role=%s | best_static_confidence=%.2f",
            specialist_role, static_best[0],
        )
        return RoutingDecision(
            routing_type="dynamic-synthesize",
            specialist=static_best[1],  # best static match (may be None), for logging
            system_prompt=None,          # caller must synthesize
            confidence=static_best[0],
            metadata={"best_static": static_best[1].name if static_best[1] else None},
        )

    def top_candidates(
        self, specialist_role: str, top_k: int = 3
    ) -> list[tuple[float, SpecialistEntry]]:
        """Return top_k (confidence, entry) pairs — useful for debugging or admin UIs."""
        return search_with_confidence(specialist_role, top_k=top_k)
