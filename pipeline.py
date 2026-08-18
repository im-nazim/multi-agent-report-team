"""Orchestration layer: wires the agents together into one full run.

run_pipeline() is the single entry point used by BOTH the main "Run Agent Team"
button and the Evaluation Dashboard, so there is exactly one implementation of
the Planner -> Research -> Fact Checker -> Writer -> Editor flow to maintain.
"""

import re
import time
from typing import List, Dict

import streamlit as st

from schemas import Finding, ResearchOutput
from agents import planner_agent, research_agent, fact_checker_agent, writer_agent, editor_agent


def build_global_sources(all_research_with_sources):
    """Deduplicates sources (by URL) across all subtopics and remaps each finding's
    local source_index to a single unified, global source list, so the writer's
    [n] markers line up with one final Sources section instead of colliding
    across subtopics."""
    global_sources: List[Dict[str, str]] = []
    url_to_global_idx: Dict[str, int] = {}

    remapped_research: List[ResearchOutput] = []

    for research, raw_sources in all_research_with_sources:
        local_to_global = {}
        for local_idx, src in enumerate(raw_sources, 1):
            url = src["href"] or f"no-url-{src['title']}"
            if url not in url_to_global_idx:
                global_sources.append(src)
                url_to_global_idx[url] = len(global_sources)
            local_to_global[local_idx] = url_to_global_idx[url]

        remapped_findings = []
        for f in research.findings:
            global_idx = local_to_global.get(f.source_index)
            if global_idx is not None:
                remapped_findings.append(Finding(claim=f.claim, source_index=global_idx))
            # silently drop findings with invalid/hallucinated source indexes

        remapped_research.append(ResearchOutput(subtopic=research.subtopic, findings=remapped_findings))

    return remapped_research, global_sources


def render_sources_markdown(global_sources: List[Dict[str, str]]) -> str:
    if not global_sources:
        return ""
    lines = ["\n---\n## Sources\n"]
    for i, s in enumerate(global_sources, 1):
        title = s["title"] or "Untitled"
        url = s["href"] or ""
        if url.startswith("doc://"):
            lines.append(f"{i}. 📄 {title} *(uploaded document)*")
        elif url:
            lines.append(f"{i}. [{title}]({url})")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def analyze_citations(report_text: str, global_sources: List[Dict[str, str]]) -> Dict:
    """Counts inline [n] citation markers used in the final report and checks
    how many actually correspond to a real entry in the source list."""
    used = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", report_text)))
    valid = [n for n in used if 1 <= n <= len(global_sources)]
    return {
        "citations_used": len(used),
        "valid_citations": len(valid),
        "invalid_citations": len(used) - len(valid),
    }


def run_pipeline(topic: str, llm, api_key: str, num_search_results: int, num_doc_chunks: int,
                  enable_fact_check: bool, trace_container=None) -> Dict:
    """Runs the full Planner -> Research -> Fact Checker -> Writer -> Editor pipeline
    for a single topic. If trace_container is given, renders live agent cards into it
    (used by the main 'Run Agent Team' button). If it's None, runs silently and
    suppresses warnings (used by the Evaluation Dashboard, which runs many topics in a
    loop and would otherwise get cluttered with per-topic UI noise).
    Returns a dict with the final report, sources, and run statistics."""
    silent = trace_container is None
    stats = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
    start_time = time.time()

    def trace(html: str):
        if trace_container is not None:
            with trace_container:
                st.markdown(html, unsafe_allow_html=True)

    # --- Planner ---
    trace('<div class="agent-card"><div class="agent-title">🧭 Planner Agent</div>Breaking topic into sub-questions...</div>')
    subtopics = planner_agent(llm, topic, stats=stats)
    trace(f'<div class="agent-card"><div class="agent-title">🧭 Planner Agent — Done</div>{"<br>".join(subtopics)}</div>')

    # --- Researcher + Fact Checker ---
    all_research_with_sources = []
    findings_before_total = 0
    findings_after_total = 0
    for i, sub in enumerate(subtopics, 1):
        trace(f'<div class="agent-card"><div class="agent-title">🔎 Research Agent ({i}/{len(subtopics)})</div>Researching: {sub}</div>')
        try:
            research_result, raw_sources = research_agent(
                llm, sub, num_search_results, api_key=api_key, num_doc_results=num_doc_chunks,
                stats=stats, silent=silent,
            )
        except Exception as e:
            research_result, raw_sources = ResearchOutput(subtopic=sub, findings=[]), []
            if not silent:
                st.warning(f"Research failed for '{sub}': {e}")

        original_count = len(research_result.findings)
        findings_before_total += original_count

        if enable_fact_check and research_result.findings:
            trace(f'<div class="agent-card"><div class="agent-title">✅ Fact Checker Agent</div>Verifying {original_count} finding(s) against their sources...</div>')
            try:
                verified_findings, _ = fact_checker_agent(
                    llm, sub, research_result.findings, raw_sources, stats=stats, silent=silent
                )
                research_result = ResearchOutput(subtopic=sub, findings=verified_findings)
            except Exception as e:
                if not silent:
                    st.warning(f"Fact-checker crashed for '{sub}', keeping unverified findings: {e}")

        findings_after_total += len(research_result.findings)
        all_research_with_sources.append((research_result, raw_sources))

        findings_html = "<br>".join(f"• {f.claim} [{f.source_index}]" for f in research_result.findings) or "No findings extracted."
        trace(f'<div class="agent-card"><div class="agent-title">🔎 Findings: {sub}</div>{findings_html}</div>')
        if trace_container is not None and enable_fact_check and original_count:
            kept = len(research_result.findings)
            with trace_container:
                if kept < original_count:
                    st.caption(f"✅ Fact Checker: {kept}/{original_count} findings verified — {original_count - kept} discarded as unsupported.")
                else:
                    st.caption(f"✅ Fact Checker: all {kept} findings verified.")

    # --- Deduplicate + renumber sources globally ---
    remapped_research, global_sources = build_global_sources(all_research_with_sources)

    # --- Writer ---
    trace('<div class="agent-card"><div class="agent-title">✍️ Writer Agent</div>Drafting the full report with citations...</div>')
    draft = writer_agent(llm, topic, remapped_research, global_sources, stats=stats)

    # --- Editor ---
    trace('<div class="agent-card"><div class="agent-title">🪄 Editor Agent</div>Polishing final report...</div>')
    try:
        final_report = editor_agent(llm, draft, stats=stats)
    except Exception as e:
        final_report = draft
        if not silent:
            st.warning(f"Editor Agent failed, showing unedited draft: {e}")

    final_report_with_sources = final_report + render_sources_markdown(global_sources)
    elapsed = time.time() - start_time
    fact_check_pass_rate = (findings_after_total / findings_before_total) if findings_before_total else None

    return {
        "topic": topic,
        "subtopics": subtopics,
        "final_report": final_report_with_sources,
        "global_sources": global_sources,
        "stats": stats,
        "elapsed": elapsed,
        "findings_before": findings_before_total,
        "findings_after": findings_after_total,
        "fact_check_pass_rate": fact_check_pass_rate,
    }