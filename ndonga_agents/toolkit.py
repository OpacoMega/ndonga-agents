"""Typed tool abstraction for Ndonga."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel, Field

logger = logging.getLogger("ndonga.toolkit")


class BaseTool(ABC):
    name: str
    description: str
    args_schema: Type[BaseModel]

    def __init__(self, db_pool: Any | None = None) -> None:
        self.db_pool = db_pool

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_schema.model_json_schema(),
        }

    @abstractmethod
    async def arun(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class SearchExperiencesArgs(BaseModel):
    location: str = Field(description="City or area, e.g. 'Nairobi', 'Mombasa', 'Diani', 'Masai Mara'.")
    max_price: int = Field(description="Maximum budget in KES (Kenyan Shillings).")


class SearchEventsArgs(BaseModel):
    location: str = Field(description="City or area, e.g. Nairobi or Diani.")
    max_price: int = Field(description="Maximum budget in KES.")
    date_range: str | None = Field(default=None, description="Natural language date range, e.g. 'this weekend'.")


class SearchPlacesArgs(BaseModel):
    location: str = Field(description="City or area.")
    vibe: str | None = Field(default=None, description="Desired vibe or category, e.g. brunch, romantic, family.")
    max_price: int | None = Field(default=None, description="Optional max budget in KES.")


class BuildItineraryArgs(BaseModel):
    location: str = Field(description="City or area.")
    budget: int = Field(description="Total budget in KES.")
    interests: list[str] | None = Field(default=None, description="User interests such as food, nature, nightlife, family.")


class SearchExperiencesTool(BaseTool):
    name = "search_experiences"
    description = (
        "Search available East African experiences by location and budget. "
        "ALWAYS call this before quoting prices or availability. "
        "Returns matching experiences with names, prices (KES), and descriptions."
    )
    args_schema = SearchExperiencesArgs

    async def arun(self, **kwargs: Any) -> Any:
        from tools import search_experiences

        args = self.args_schema.model_validate(kwargs)
        return await search_experiences(args.location, args.max_price, self.db_pool)


class SearchEventsTool(BaseTool):
    name = "search_events"
    description = "Search East African events by location, optional date range, and budget."
    args_schema = SearchEventsArgs

    async def arun(self, **kwargs: Any) -> Any:
        from tools import search_events

        args = self.args_schema.model_validate(kwargs)
        return await search_events(args.location, args.date_range, args.max_price, self.db_pool)


class SearchPlacesTool(BaseTool):
    name = "search_places"
    description = "Search restaurants, cafes, stays, venues, and lifestyle places."
    args_schema = SearchPlacesArgs

    async def arun(self, **kwargs: Any) -> Any:
        from tools import search_places

        args = self.args_schema.model_validate(kwargs)
        return await search_places(args.location, args.vibe, args.max_price, self.db_pool)


class BuildItineraryTool(BaseTool):
    name = "build_itinerary"
    description = "Build a lifestyle itinerary from available experiences and places."
    args_schema = BuildItineraryArgs

    async def arun(self, **kwargs: Any) -> Any:
        from tools import build_itinerary

        args = self.args_schema.model_validate(kwargs)
        return await build_itinerary(args.location, args.budget, args.interests, self.db_pool)


class Toolkit:
    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get_provider_format(self, provider: str = "openai") -> list[dict[str, Any]]:
        if provider == "anthropic":
            return [tool.to_anthropic_format() for tool in self._tools.values()]
        return [tool.to_openai_format() for tool in self._tools.values()]


# ── Specialist delegation tool ────────────────────────────────────────────────

class ConsultSpecialistArgs(BaseModel):
    specialist_role: str = Field(
        description=(
            "Type of expert needed from the Ndonga Omni-Swarm. Use plain English: "
            "'KRA tax specialist', 'M-Pesa merchant support', 'contract lawyer', "
            "'python developer', 'financial analyst', 'social media manager', etc. "
            "Be specific — the more precise the role, the better the match."
        )
    )
    query: str = Field(
        description="The specific question or task for the specialist. Be detailed."
    )


class ConsultSpecialistTool(BaseTool):
    name = "consult_specialist"
    description = (
        "Delegate a task to the Ndonga Omni-Swarm — 271+ experts spanning Kenyan citizen "
        "services (KRA, eCitizen, M-Pesa, NHIF, SACCO), law, engineering, finance, "
        "marketing, design, HR, data science, security, and more. For truly novel "
        "queries, the Swarm dynamically synthesizes a bespoke specialist on the fly. "
        "Do NOT use for travel, experiences, events, stays, or lifestyle topics."
    )
    args_schema = ConsultSpecialistArgs

    async def arun(self, **kwargs: Any) -> Any:
        from .swarm_router import SwarmRouter
        from .swarm_synthesizer import SwarmSynthesizer
        from .llm_gateway import OpenRouterProvider, LLMGatewayError

        args = self.args_schema.model_validate(kwargs)
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            return {"error": "OPENROUTER_API_KEY not configured — cannot consult specialist."}

        # ── Stage 1–3: Route through the static catalog + dynamic cache ───────
        router = SwarmRouter(db_pool=self.db_pool)
        decision = await router.route(args.specialist_role)

        system_prompt: str
        routing_label: str

        if decision.has_prompt:
            # Static catalog match (yapa-local / upstream) or dynamic cache hit
            system_prompt = decision.system_prompt  # type: ignore[assignment]
            routing_label = decision.routing_type
            specialist_name = (
                decision.specialist.name if decision.specialist
                else decision.cached_entry.get("agent_name", "cached") if decision.cached_entry
                else "cached"
            )
            category = (
                decision.specialist.category if decision.specialist
                else "dynamic"
            )
            logger.info(
                "specialist_routed | role=%s | agent=%s | routing=%s | confidence=%.2f",
                args.specialist_role, specialist_name, routing_label, decision.confidence,
            )
        else:
            # ── Stage 4: Dynamic synthesis ────────────────────────────────────
            routing_label = "dynamic-synthesize"
            synthesizer = SwarmSynthesizer(api_key=api_key, db_pool=self.db_pool)
            try:
                system_prompt, query_hash = await synthesizer.synthesize_and_cache(
                    specialist_role=args.specialist_role,
                    user_query=args.query,
                )
                specialist_name = f"Synthesized: {args.specialist_role}"
                category = "dynamic"
                logger.info(
                    "specialist_synthesized | role=%s | hash=%s",
                    args.specialist_role, query_hash[:16],
                )
            except RuntimeError as exc:
                logger.error("synthesis_failed | role=%s | %s", args.specialist_role, exc)
                # Graceful degradation: if synthesis fails, use best static match anyway
                if decision.specialist:
                    from .specialist_index import load_specialist_prompt
                    system_prompt = load_specialist_prompt(decision.specialist.path)
                    specialist_name = decision.specialist.name
                    category = decision.specialist.category
                    routing_label = f"{decision.specialist.origin}-fallback"
                else:
                    return {"error": f"Specialist synthesis failed: {exc}"}

        # ── Execute the query with the resolved specialist persona ────────────
        model = os.getenv(
            "NDONGA_STARLIGHT_MODEL",
            "meta-llama/llama-3.3-70b-instruct:free",
        )
        provider = OpenRouterProvider(api_key=api_key)
        try:
            response = await provider.achat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": args.query},
                ],
                max_tokens=1400,
                temperature=0.3,
            )
        except LLMGatewayError as exc:
            logger.warning(
                "specialist_call_failed | agent=%s | routing=%s | error=%s",
                specialist_name, routing_label, exc,
            )
            return {"error": f"Specialist call failed: {exc}"}

        return {
            "specialist": specialist_name,
            "category": category,
            "routing": routing_label,
            "confidence": round(decision.confidence, 3),
            "response": response.content,
        }


# ── Per-tenant toolkit factories ──────────────────────────────────────────────

def nenda_toolkit(db_pool: Any | None = None) -> Toolkit:
    """Hapa Kule — travel, experiences, local lifestyle + specialist delegation."""
    return Toolkit([
        SearchExperiencesTool(db_pool),
        SearchEventsTool(db_pool),
        SearchPlacesTool(db_pool),
        BuildItineraryTool(db_pool),
        ConsultSpecialistTool(db_pool),
    ])


def baza_toolkit(db_pool: Any | None = None) -> Toolkit:
    """Machant — SME commerce, sales, inventory + specialist delegation."""
    return Toolkit([ConsultSpecialistTool(db_pool)])


def panga_toolkit(db_pool: Any | None = None) -> Toolkit:
    """Kaya — wealth planning, financial clarity + specialist delegation."""
    return Toolkit([ConsultSpecialistTool(db_pool)])


def safiri_toolkit(db_pool: Any | None = None) -> Toolkit:
    """Al Sabil — Islamic travel + specialist delegation."""
    return Toolkit([ConsultSpecialistTool(db_pool)])
