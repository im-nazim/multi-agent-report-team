"""Web search tool used by the Research Agent."""

from typing import List, Dict
from ddgs import DDGS


def web_search(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Returns a list of {title, body, href} dicts instead of a flat string,
    so downstream agents can track exactly which source backs which claim."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "Untitled"),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })
    except Exception as e:
        results.append({"title": "Search error", "body": f"[Search failed for '{query}': {e}]", "href": ""})
    return results


def format_sources_for_prompt(sources: List[Dict[str, str]]) -> str:
    """Renders a numbered source list as plain text for inclusion in an LLM prompt."""
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(f"[{i}] {s['title']}: {s['body']} (URL: {s['href']})")
    return "\n".join(lines) if lines else "[No results found]"