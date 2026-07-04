"""
Ndonga Specialist Index — in-memory search over the full agent catalog.

Scans all category directories at import time, parses YAML-style frontmatter,
and provides keyword-based specialist search with confidence scoring.

Catalog tiers:
  upstream  — 221 generic agents from the ndonga-agents base catalog
  yapa-local — Yapa's own hyper-localised African specialists (50 agents)
  dynamic   — synthesized-on-the-fly prompts cached in yapa-local/dynamic/
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

# Root of the catalog (one level above this package directory)
_CATALOG_ROOT = Path(__file__).resolve().parent.parent

# Upstream (generic) category directories
_UPSTREAM_DIRS: frozenset[str] = frozenset({
    "academic", "design", "engineering", "finance", "game-development",
    "gis", "integrations", "marketing", "paid-media", "product",
    "project-management", "sales", "security", "spatial-computing",
    "specialized", "strategy", "support", "testing",
})

# Yapa-local contextual directory (may include subdirs like yapa-local/dynamic/)
_YAPA_LOCAL_DIR = "yapa-local"

# All scannable catalog dirs
_CATALOG_DIRS: frozenset[str] = _UPSTREAM_DIRS | {_YAPA_LOCAL_DIR}

# Files to skip even if they end in .md
_SKIP_FILENAMES: frozenset[str] = frozenset({
    "README.md", "CONTRIBUTING.md", "CONTRIBUTING_zh-CN.md", "SECURITY.md",
    "CHANGELOG.md", "EXECUTIVE-BRIEF.md", "QUICKSTART.md", "nexus-strategy.md",
})

# Confidence threshold: scores below this trigger dynamic synthesis
CONFIDENCE_THRESHOLD: float = 0.30

# Sprint D: tenant-scoped swarm categories
TENANT_SWARM_CATEGORIES: dict[str, frozenset[str]] = {
    "hapakule": frozenset({"marketing", "design", "strategy", "product", "yapa-local", "specialized"}),
    "machant": frozenset({"sales", "finance", "engineering", "product", "project-management", "yapa-local"}),
    "kaya": frozenset({"finance", "strategy", "legal", "yapa-local", "specialized"}),
    "alsabil": frozenset({"specialized", "strategy", "marketing", "yapa-local"}),
}


class SpecialistEntry(NamedTuple):
    name: str
    description: str
    vibe: str
    category: str
    path: str               # absolute path to the .md file
    tokens: frozenset[str]  # pre-tokenised for fast search
    origin: str = "upstream"      # "upstream" | "yapa-local" | "dynamic"
    tags: frozenset[str] = frozenset()  # explicit keyword tags from frontmatter


def _parse_frontmatter(text: str) -> dict[str, str | list[str]]:
    """Extract key: value pairs from YAML-style --- frontmatter block."""
    result: dict[str, str | list[str]] = {}
    if not text.startswith("---"):
        return result
    end = text.find("---", 3)
    if end == -1:
        return result
    block = text[3:end]
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip().strip('"').strip("'")
        if key == "tags":
            result[key] = [t.strip() for t in val.split(",") if t.strip()]
        else:
            result[key] = val
    return result


def _tokenise(text: str) -> frozenset[str]:
    """Lowercase word tokens, dropping short stop-words."""
    _STOP = {"a", "an", "the", "and", "or", "for", "of", "in", "to", "is", "at", "with"}
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(t for t in tokens if len(t) > 2 and t not in _STOP)


def _build_index() -> list[SpecialistEntry]:
    entries: list[SpecialistEntry] = []
    for cat in sorted(_CATALOG_DIRS):
        cat_dir = _CATALOG_ROOT / cat
        if not cat_dir.is_dir():
            continue
        origin = "yapa-local" if cat == _YAPA_LOCAL_DIR else "upstream"
        for path in sorted(cat_dir.rglob("*.md")):
            # Skip cached dynamic agents and meta files
            if path.name in _SKIP_FILENAMES:
                continue
            if "dynamic" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            name = (fm.get("name") or "")
            if isinstance(name, list):
                name = " ".join(name)
            name = name.strip()
            if not name:
                continue
            description = fm.get("description", "")
            if isinstance(description, list):
                description = " ".join(description)
            vibe = fm.get("vibe", "")
            if isinstance(vibe, list):
                vibe = " ".join(vibe)
            raw_tags: list[str] = fm.get("tags", [])  # type: ignore[assignment]
            if isinstance(raw_tags, str):
                raw_tags = [raw_tags]
            tags_set = frozenset(t.lower() for t in raw_tags if t)
            tokens = _tokenise(f"{name} {description} {vibe} {cat} {' '.join(raw_tags)}")
            # Honour explicit origin override from frontmatter
            file_origin = fm.get("origin", origin)
            if isinstance(file_origin, list):
                file_origin = file_origin[0]
            entries.append(SpecialistEntry(
                name=name,
                description=str(description),
                vibe=str(vibe),
                category=cat,
                path=str(path),
                tokens=tokens,
                origin=str(file_origin),
                tags=tags_set,
            ))
    return entries


@lru_cache(maxsize=1)
def get_index() -> list[SpecialistEntry]:
    """Return the full catalog index (built once, cached for the process lifetime)."""
    return _build_index()


def search_specialists(query: str, top_k: int = 3) -> list[SpecialistEntry]:
    """Return the top_k specialists whose name/description best match the query."""
    results = search_with_confidence(query, top_k=top_k)
    return [entry for _, entry in results]


def search_with_confidence(
    query: str,
    top_k: int = 3,
    tenant_id: str | None = None,
) -> list[tuple[float, SpecialistEntry]]:
    """
    Return (confidence_score, entry) tuples for the top_k best matches.

    confidence_score is in [0.0, 1.0]:
      1.0 = every query token matched the entry
      0.0 = no query tokens matched

    Yapa-local entries get a small priority boost (0.05) to prefer contextual
    African specialists over generic upstream ones at equal token overlap.
    """
    query_tokens = _tokenise(query)
    if not query_tokens:
        return []
    allowed_categories = TENANT_SWARM_CATEGORIES.get(tenant_id or "", None)
    n = len(query_tokens)
    scored: list[tuple[float, int, SpecialistEntry]] = []
    for entry in get_index():
        if allowed_categories and entry.category not in allowed_categories and entry.origin != "yapa-local":
            continue
        overlap = len(query_tokens & entry.tokens)
        if overlap == 0:
            continue
        base_confidence = overlap / n
        # Priority boost for yapa-local agents
        boost = 0.05 if entry.origin == "yapa-local" else 0.0
        # Secondary tie-break: name token hits
        name_tokens = _tokenise(entry.name)
        name_hits = len(query_tokens & name_tokens)
        confidence = min(1.0, base_confidence + boost)
        scored.append((confidence, name_hits, entry))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(conf, entry) for conf, _, entry in scored[:top_k]]


def load_specialist_prompt(path: str) -> str:
    """Return the markdown body below the frontmatter block as the system prompt."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return text.strip()
    end = text.find("---", 3)
    if end == -1:
        return text.strip()
    body_start = text.index("\n", end) + 1 if "\n" in text[end:] else end + 3
    return text[body_start:].strip()


def load_dynamic_cache() -> list[dict]:
    """Load all cached dynamic agent entries from yapa-local/dynamic/*.json."""
    dynamic_dir = _CATALOG_ROOT / _YAPA_LOCAL_DIR / "dynamic"
    if not dynamic_dir.is_dir():
        return []
    entries = []
    for path in sorted(dynamic_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_path"] = str(path)
            entries.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return entries


def search_dynamic_cache(query: str) -> dict | None:
    """
    Find a previously synthesized agent in the dynamic cache that matches query.
    Returns the cache entry dict, or None if no match found.
    """
    q_tokens = _tokenise(query)
    if not q_tokens:
        return None
    best: tuple[float, dict] | None = None
    for entry in load_dynamic_cache():
        cached_tokens = _tokenise(entry.get("query", "") + " " + entry.get("slug", ""))
        if not cached_tokens:
            continue
        overlap = len(q_tokens & cached_tokens)
        confidence = overlap / len(q_tokens)
        if confidence >= CONFIDENCE_THRESHOLD:
            if best is None or confidence > best[0]:
                best = (confidence, entry)
    return best[1] if best else None


def catalog_summary() -> dict[str, int]:
    """Return agent count per category — useful for /health or admin endpoints."""
    counts: dict[str, int] = {}
    for entry in get_index():
        counts[entry.category] = counts.get(entry.category, 0) + 1
    return dict(sorted(counts.items()))


def origin_summary() -> dict[str, int]:
    """Return agent count per origin tier."""
    counts: dict[str, int] = {}
    for entry in get_index():
        counts[entry.origin] = counts.get(entry.origin, 0) + 1
    # Include dynamic cache count
    counts["dynamic"] = len(load_dynamic_cache())
    return dict(sorted(counts.items()))
