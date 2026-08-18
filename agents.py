"""The agent team: Planner, Research, Fact Checker, Writer, Editor, and the
Evaluation Judge. Each agent is a plain function that takes an llm plus
whatever context it needs, and returns either plain text or a structured
Pydantic object.

Every agent optionally accepts a `stats` dict to accumulate token usage into
(see _accumulate_usage) and a `silent` flag to suppress st.warning() calls
during batch evaluation runs.
"""

from typing import List, Dict, Optional

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from schemas import Finding, ResearchOutput, FactCheckOutput, RelevanceJudgement
from search_tools import web_search, format_sources_for_prompt
from rag import doc_search


# ----------------------------- LLM SETUP -----------------------------

def get_llm(api_key: str, model_name: str):
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.4)


def _accumulate_usage(stats: Optional[Dict], response) -> None:
    """Adds a response's token usage to a running stats dict, if both are present.
    Used to build the token/cost numbers shown in the Evaluation Dashboard."""
    if stats is None or response is None:
        return
    usage = getattr(response, "usage_metadata", None)
    if usage:
        stats["input_tokens"] = stats.get("input_tokens", 0) + (usage.get("input_tokens") or 0)
        stats["output_tokens"] = stats.get("output_tokens", 0) + (usage.get("output_tokens") or 0)
    stats["llm_calls"] = stats.get("llm_calls", 0) + 1


def call_agent(llm, system_prompt: str, user_prompt: str, stats: Optional[Dict] = None) -> str:
    """Plain (non-structured) LLM call used by the Planner, Writer, and Editor."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    _accumulate_usage(stats, response)
    content = response.content
    # Some newer Gemini models return content as a list of blocks instead of a plain string
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        content = "\n".join(parts)
    return content.strip() if isinstance(content, str) else str(content)


# ----------------------------- AGENTS -----------------------------

def planner_agent(llm, topic: str, stats: Optional[Dict] = None):
    system = (
        "You are the Planner Agent in a multi-agent research team. "
        "Break the user's topic into 3-5 focused, non-overlapping sub-questions "
        "that together would produce a comprehensive report. "
        "Return ONLY a numbered list, nothing else."
    )
    result = call_agent(llm, system, f"Topic: {topic}", stats=stats)
    subtopics = [line.split(".", 1)[-1].strip(" -") for line in result.strip().split("\n") if line.strip()]
    return [s for s in subtopics if s]


def research_agent(llm, subtopic: str, num_results: int, api_key: str = "", num_doc_results: int = 0,
                    stats: Optional[Dict] = None, silent: bool = False):
    """Runs a web search AND (if a knowledge base is indexed) a document search,
    merges both into one numbered source list, then asks the LLM to produce
    STRUCTURED findings, each tagged with which local source index it came from.
    Returns (ResearchOutput, raw_sources_list)."""
    raw_sources = web_search(subtopic, num_results)

    if num_doc_results > 0 and st.session_state.get("kb_chunks"):
        raw_sources = raw_sources + doc_search(subtopic, api_key, num_doc_results)

    sources_text = format_sources_for_prompt(raw_sources)

    structured_llm = llm.with_structured_output(ResearchOutput, include_raw=True)

    system = (
        "You are the Research Agent. You are given a sub-question and a numbered list of sources "
        "(a mix of web search results and excerpts from the user's own uploaded documents). "
        "Extract 4-6 concise, factual findings relevant to the sub-question. "
        "Every finding MUST include the source_index of the numbered source it came from. "
        "Do not invent information not present in the results. "
        "Do not invent source indexes that don't exist in the list. "
        "Prefer findings backed by the uploaded documents when they are relevant and specific."
    )
    user = f"Sub-question: {subtopic}\n\nNumbered sources:\n{sources_text}"

    try:
        response = structured_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        result: ResearchOutput = response["parsed"]
        if result is None:
            raise ValueError(str(response.get("parsing_error")) or "No parsed result returned")
        _accumulate_usage(stats, response["raw"])
    except Exception as e:
        # Fallback: no structured findings, but don't crash the pipeline
        result = ResearchOutput(subtopic=subtopic, findings=[])
        if not silent:
            st.warning(f"Structured research parsing failed for '{subtopic}': {e}")

    return result, raw_sources


def fact_checker_agent(llm, subtopic: str, findings: List[Finding], raw_sources: List[Dict[str, str]],
                        stats: Optional[Dict] = None, silent: bool = False):
    """Checks each finding against the ACTUAL TEXT of the source it cited.
    Drops findings that aren't genuinely supported. Fails open (keeps everything)
    if the check itself errors out, so a flaky API call can't kill the whole run.
    Returns (verified_findings, verdicts)."""
    if not findings:
        return findings, []

    findings_text = "\n".join(
        f"{i}. Claim: \"{f.claim}\" — cites source #{f.source_index}"
        for i, f in enumerate(findings, 1)
    )
    sources_text = format_sources_for_prompt(raw_sources)

    structured_llm = llm.with_structured_output(FactCheckOutput, include_raw=True)

    system = (
        "You are the Fact Checker Agent. You are given a numbered list of claims, each one citing "
        "a specific numbered source, and the full text of those sources. "
        "For EACH claim, check whether the text of its cited source actually supports it. "
        "Mark supported=false if the source doesn't mention it, contradicts it, or only loosely relates to it. "
        "Return one verdict per claim, using the same finding_number as given."
    )
    user = f"Sub-question: {subtopic}\n\nClaims to verify:\n{findings_text}\n\nFull source texts:\n{sources_text}"

    try:
        response = structured_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        result: FactCheckOutput = response["parsed"]
        if result is None:
            raise ValueError(str(response.get("parsing_error")) or "No parsed result returned")
        _accumulate_usage(stats, response["raw"])
    except Exception as e:
        if not silent:
            st.warning(f"Fact-checking failed for '{subtopic}', keeping all findings unverified: {e}")
        return findings, []

    verdict_map = {v.finding_number: v for v in result.verdicts}
    verified = []
    for i, f in enumerate(findings, 1):
        verdict = verdict_map.get(i)
        if verdict is None or verdict.supported:
            verified.append(f)

    return verified, result.verdicts


def writer_agent(llm, topic: str, all_research: List[ResearchOutput], global_sources: List[Dict[str, str]],
                  stats: Optional[Dict] = None):
    """Writes the report using structured findings, citing claims as [n]
    where n refers to the GLOBAL (deduplicated) source list."""
    notes_lines = []
    for r in all_research:
        notes_lines.append(f"### {r.subtopic}")
        for f in r.findings:
            notes_lines.append(f"- {f.claim} [source #{f.source_index}]")
    notes_text = "\n".join(notes_lines)

    sources_list_text = "\n".join(
        f"[{i}] {s['title']} — {s['href']}" for i, s in enumerate(global_sources, 1)
    )

    system = (
        "You are the Writer Agent. Using the structured research notes provided, write a clear, "
        "well-structured draft report on the main topic. Use Markdown headings and short paragraphs. "
        "Every factual claim you write MUST end with an inline citation marker like [1] or [2], "
        "using the EXACT source numbers given in the notes — do not renumber or invent new ones. "
        "Do not fabricate facts beyond what's in the notes. "
        "Do not add a Sources section yourself — that will be appended separately."
    )
    user = (
        f"Main topic: {topic}\n\n"
        f"Research notes (each finding already tagged with its source number):\n{notes_text}\n\n"
        f"Reference — global source list (for your awareness only, do not repeat this list):\n{sources_list_text}"
    )
    return call_agent(llm, system, user, stats=stats)


def editor_agent(llm, draft: str, stats: Optional[Dict] = None):
    system = (
        "You are the Editor Agent. Polish the draft report: improve clarity, fix flow, "
        "add a short executive summary at the top and a 'Key Takeaways' section at the end. "
        "Keep formatting in clean Markdown. "
        "IMPORTANT: preserve every inline citation marker like [1], [2] exactly as they appear — "
        "do not remove, renumber, or add new ones."
    )
    return call_agent(llm, system, draft, stats=stats)


def judge_agent(llm, topic: str, report_text: str, stats: Optional[Dict] = None) -> RelevanceJudgement:
    """An independent LLM judge that scores how relevant/complete a final report
    is relative to its topic. Used only by the Evaluation Dashboard."""
    structured_llm = llm.with_structured_output(RelevanceJudgement, include_raw=True)
    system = (
        "You are an independent Evaluation Judge, not part of the writing team. "
        "Rate how well the given report addresses the given topic, from 1 (irrelevant or incomplete) "
        "to 5 (thorough and directly on-topic). Be strict and critical."
    )
    user = f"Topic: {topic}\n\nReport:\n{report_text[:6000]}"
    try:
        response = structured_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        result: RelevanceJudgement = response["parsed"]
        if result is None:
            raise ValueError(str(response.get("parsing_error")) or "No parsed result returned")
        _accumulate_usage(stats, response["raw"])
        return result
    except Exception as e:
        return RelevanceJudgement(score=0, reasoning=f"Judge failed: {e}")