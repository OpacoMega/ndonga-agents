"""Typed tool abstraction for Ndonga."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel, Field


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


def nenda_toolkit(db_pool: Any | None = None) -> Toolkit:
    return Toolkit([
        SearchExperiencesTool(db_pool),
        SearchEventsTool(db_pool),
        SearchPlacesTool(db_pool),
        BuildItineraryTool(db_pool),
    ])
