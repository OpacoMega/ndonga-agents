"""Lightweight orchestrator for Ndonga's multi-agent routing."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .llm_gateway import BaseLLMProvider


CATALOG_ROOT = Path(__file__).resolve().parent.parent
NON_AGENT_DIRS = {"examples", "integrations", "scripts", "strategy"}

# Yapa agent → tenant_id mapping (matches TENANT_REGISTRY keys in config.py)
_AGENT_TO_TENANT: dict[str, str] = {
    "nenda": "hapakule",
    "baza": "machant",
    "panga": "kaya",
    "safiri": "alsabil",
    "general": "hapakule",
}

# In-memory Yapa agent registry — avoids walking 276 generic catalog .md files.
# These are the five agents Ndonga actually routes to.
_YAPA_AGENTS: list[dict[str, str]] = [
    {
        "slug": "nenda",
        "name": "Nenda",
        "description": "East Africa travel, lifestyle, safaris, events, and local experiences concierge.",
        "division": "hapakule",
    },
    {
        "slug": "baza",
        "name": "Baza",
        "description": "SMB commerce operations — sales, inventory, pricing, and merchant intelligence.",
        "division": "machant",
    },
    {
        "slug": "panga",
        "name": "Panga",
        "description": "Wealth clarity and financial planning across generations for Kenyan families.",
        "division": "kaya",
    },
    {
        "slug": "safiri",
        "name": "Safiri",
        "description": "Islamic travel guide — halal-friendly experiences and Umrah planning.",
        "division": "alsabil",
    },
    {
        "slug": "general",
        "name": "General",
        "description": "General-purpose assistant for requests that don't fit a specialised agent.",
        "division": "hapakule",
    },
]


@dataclass(frozen=True)
class AgentDefinition:
    slug: str
    name: str
    description: str
    division: str
    path: str
    body: str
    metadata: dict[str, str]


class OrchestratorTask(BaseModel):
    agent: str = Field(description="One of nenda, baza, panga, safiri, general.")
    task: str
    rationale: str = ""


class OrchestratorResult(BaseModel):
    agent: str
    task: str
    result: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, parts[2].lstrip()


class YapaOrchestrator:
    """Catalog reader used by Ndonga to discover and route agent definitions."""

    def __init__(self, catalog_root: str | Path | None = None, llm_provider: BaseLLMProvider | None = None) -> None:
        self.catalog_root = Path(catalog_root) if catalog_root else CATALOG_ROOT
        self.llm_provider = llm_provider

    def divisions(self) -> dict[str, Any]:
        path = self.catalog_root / "divisions.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8")).get("divisions", {})

    def tools(self) -> dict[str, Any]:
        path = self.catalog_root / "tools.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8")).get("tools", {})

    def list_agents(self, division: str | None = None) -> list[AgentDefinition]:
        """Return Yapa agents from the in-memory registry (O(1) — no disk scan).

        The generic 276-file catalog inherited from the upstream fork is excluded;
        none of those entries are Yapa-relevant. Pass division= to filter by
        tenant/product line (hapakule, machant, kaya, alsabil).
        """
        agents = []
        for entry in _YAPA_AGENTS:
            if division and entry["division"] != division:
                continue
            agents.append(
                AgentDefinition(
                    slug=entry["slug"],
                    name=entry["name"],
                    description=entry["description"],
                    division=entry["division"],
                    path=f"{entry['division']}/{entry['slug']}.md",
                    body="",
                    metadata={},
                )
            )
        return agents

    def get_agent(self, slug: str) -> AgentDefinition | None:
        for agent in self.list_agents():
            if agent.slug == slug or agent.name.lower() == slug.lower():
                return agent
        return None

    def route(self, query: str, division: str | None = None) -> AgentDefinition | None:
        """Pick a deterministic best-effort agent by keyword overlap."""
        query_terms = {term.lower() for term in query.replace("-", " ").split() if len(term) > 2}
        best_agent: AgentDefinition | None = None
        best_score = 0
        for agent in self.list_agents(division=division):
            haystack = f"{agent.slug} {agent.name} {agent.description}".lower()
            score = sum(1 for term in query_terms if term in haystack)
            if score > best_score:
                best_score = score
                best_agent = agent
        return best_agent

    async def decompose_request(self, message: str) -> list[OrchestratorTask]:
        if self.llm_provider is not None:
            prompt = (
                "Return JSON only — no other text, no markdown fence, no explanation. "
                "Output a JSON array where each element has exactly three fields: "
                "\"agent\" (one of: nenda, baza, panga, safiri, general), "
                "\"task\" (the sub-task to execute), "
                "\"rationale\" (one sentence explaining why). "
                "Allowed agents map to: nenda=travel/lifestyle, baza=commerce/merchant, "
                "panga=finance/tax, safiri=Islamic travel, general=everything else. "
                f"User request: {message}"
            )
            # Use the Starlight model alias (fast, resolved from env/config)
            starlight_model = os.getenv(
                "NDONGA_STARLIGHT_MODEL",
                "meta-llama/llama-3.3-70b-instruct:free",
            )
            response = await self.llm_provider.achat(
                model=starlight_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            try:
                parsed = json.loads(response.content)
                return [OrchestratorTask.model_validate(item) for item in parsed]
            except Exception:
                return self._heuristic_decompose(message)
        return self._heuristic_decompose(message)

    def _heuristic_decompose(self, message: str) -> list[OrchestratorTask]:
        text = message.lower()
        tasks: list[OrchestratorTask] = []
        if any(term in text for term in ["trip", "travel", "event", "restaurant", "nairobi", "experience"]):
            tasks.append(OrchestratorTask(agent="nenda", task=message, rationale="Lifestyle or travel intent"))
        if any(term in text for term in ["sales", "inventory", "merchant", "business", "shop"]):
            tasks.append(OrchestratorTask(agent="baza", task=message, rationale="Commerce operations intent"))
        if any(term in text for term in ["wealth", "budget", "invest", "estate", "finance"]):
            tasks.append(OrchestratorTask(agent="panga", task=message, rationale="Financial planning intent"))
        if any(term in text for term in ["halal", "islamic", "umrah", "sabil"]):
            tasks.append(OrchestratorTask(agent="safiri", task=message, rationale="Islamic travel intent"))
        return tasks or [OrchestratorTask(agent="general", task=message, rationale="General request")]

    async def execute_workflow(self, tasks: list[OrchestratorTask]) -> list[OrchestratorResult]:
        """Execute each task against its tenant agent via the LLM provider.

        When self.llm_provider is set, each task gets a real LLM response using
        the matching tenant's system prompt from TENANT_REGISTRY. Falls back to
        a placeholder string when no provider is injected (e.g. in tests).
        """
        results: list[OrchestratorResult] = []
        for task in tasks:
            if self.llm_provider is not None:
                try:
                    from config import TENANT_REGISTRY  # root-level config; on sys.path via main.py
                    tenant_id = _AGENT_TO_TENANT.get(task.agent, "hapakule")
                    tenant = TENANT_REGISTRY.get(tenant_id)
                    system_prompt = (
                        tenant.system_prompt
                        if tenant
                        else f"You are {task.agent.title()}, a helpful AI assistant."
                    )
                    starlight_model = os.getenv(
                        "NDONGA_STARLIGHT_MODEL",
                        "meta-llama/llama-3.3-70b-instruct:free",
                    )
                    response = await self.llm_provider.achat(
                        model=starlight_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": task.task},
                        ],
                        max_tokens=800,
                        temperature=0.35,
                    )
                    result_text = response.content or f"{task.agent.title()} completed: {task.task}"
                except Exception as exc:  # noqa: BLE001
                    result_text = f"Error from {task.agent}: {exc}"
            else:
                result_text = f"{task.agent.title()} analysis queued: {task.task}"
            results.append(OrchestratorResult(agent=task.agent, task=task.task, result=result_text))
        return results

    async def synthesize_results(self, results: list[OrchestratorResult]) -> str:
        if not results:
            return "I could not determine the right agent workflow for this request."
        lines = ["Here is the coordinated Ndonga response:"]
        for result in results:
            lines.append(f"- {result.agent.title()}: {result.result}")
        return "\n".join(lines)

    async def orchestrate(self, message: str) -> dict[str, Any]:
        tasks = await self.decompose_request(message)
        results = await self.execute_workflow(tasks)
        synthesis = await self.synthesize_results(results)
        return {
            "tasks": [task.model_dump() for task in tasks],
            "results": [result.model_dump() for result in results],
            "response": synthesis,
        }
