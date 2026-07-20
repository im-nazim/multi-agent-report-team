"""
AI Research & Report Team — Multi-Agent Automation System
-----------------------------------------------------------
A 4-agent pipeline (Planner -> Researcher -> Writer -> Editor) that takes
a topic/task from the user and produces a polished, structured report.

Built with: Streamlit + LangChain + Google Gemini + DuckDuckGo Search
"""

import os
import time
import streamlit as st
from duckduckgo_search import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="AI Research & Report Team", page_icon="", layout="wide")

st.markdown(
    """
    <style>
    .agent-card {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border-left: 5px solid #6C63FF;
        background-color: #f7f7fb;
    }
    .agent-title {
        font-weight: 700;
        color: #6C63FF;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Research & Report Team")
st.caption("A multi-agent system: Planner → Researcher → Writer → Editor")

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get one free at aistudio.google.com")
    model_name = st.selectbox("Model", ["gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-flash-lite-latest"], index=0)
    num_search_results = st.slider("Web sources per sub-topic", 2, 8, 4)
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. 🧭 Planner breaks your topic into sub-questions\n"
        "2. 🔎 Researcher searches the web for each\n"
        "3. ✍️ Writer drafts a full report\n"
        "4. 🪄 Editor polishes and formats the final version"
    )

# ----------------------------- HELPERS -----------------------------

def get_llm(api_key: str, model_name: str):
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.4)


def call_agent(llm, system_prompt: str, user_prompt: str) -> str:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
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


def web_search(query: str, max_results: int = 4):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"- {r.get('title', '')}: {r.get('body', '')} (Source: {r.get('href', '')})")
    except Exception as e:
        results.append(f"[Search failed for '{query}': {e}]")
    return "\n".join(results) if results else "[No results found]"


# ----------------------------- AGENTS -----------------------------

def planner_agent(llm, topic: str):
    system = (
        "You are the Planner Agent in a multi-agent research team. "
        "Break the user's topic into 3-5 focused, non-overlapping sub-questions "
        "that together would produce a comprehensive report. "
        "Return ONLY a numbered list, nothing else."
    )
    result = call_agent(llm, system, f"Topic: {topic}")
    subtopics = [line.split(".", 1)[-1].strip(" -") for line in result.strip().split("\n") if line.strip()]
    return [s for s in subtopics if s]


def research_agent(llm, subtopic: str, num_results: int):
    raw_results = web_search(subtopic, num_results)
    system = (
        "You are the Research Agent. Summarize the following raw web search results "
        "into 4-6 concise, factual bullet points relevant to the sub-question. "
        "Do not invent information not present in the results."
    )
    summary = call_agent(llm, system, f"Sub-question: {subtopic}\n\nRaw results:\n{raw_results}")
    return summary


def writer_agent(llm, topic: str, research_notes: dict):
    notes_text = "\n\n".join([f"### {k}\n{v}" for k, v in research_notes.items()])
    system = (
        "You are the Writer Agent. Using the research notes provided, write a clear, "
        "well-structured draft report on the main topic. Use headings and short paragraphs. "
        "Do not fabricate facts beyond what's in the notes."
    )
    draft = call_agent(llm, system, f"Main topic: {topic}\n\nResearch notes:\n{notes_text}")
    return draft


def editor_agent(llm, draft: str):
    system = (
        "You are the Editor Agent. Polish the draft report: improve clarity, fix flow, "
        "add a short executive summary at the top and a 'Key Takeaways' section at the end. "
        "Keep formatting in clean Markdown."
    )
    final = call_agent(llm, system, draft)
    return final


# ----------------------------- MAIN UI -----------------------------

topic = st.text_input("📝 Enter a topic or task for your report", placeholder="e.g. Impact of AI on small business marketing in 2026")
run = st.button("🚀 Run Agent Team", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    if not topic.strip():
        st.error("Please enter a topic.")
        st.stop()

    llm = get_llm(api_key, model_name)
    trace_container = st.container()

    # --- Planner ---
    with trace_container:
        st.markdown('<div class="agent-card"><div class="agent-title">🧭 Planner Agent</div>Breaking topic into sub-questions...</div>', unsafe_allow_html=True)
    try:
        subtopics = planner_agent(llm, topic)
    except Exception as e:
        st.error(f"Planner Agent failed: {e}")
        st.stop()

    with trace_container:
        st.markdown(f'<div class="agent-card"><div class="agent-title">🧭 Planner Agent — Done</div>{"<br>".join(subtopics)}</div>', unsafe_allow_html=True)

    # --- Researcher ---
    research_notes = {}
    for i, sub in enumerate(subtopics, 1):
        with trace_container:
            st.markdown(f'<div class="agent-card"><div class="agent-title">🔎 Research Agent ({i}/{len(subtopics)})</div>Researching: {sub}</div>', unsafe_allow_html=True)
        try:
            notes = research_agent(llm, sub, num_search_results)
        except Exception as e:
            notes = f"[Research failed: {e}]"
        research_notes[sub] = notes
        with trace_container:
            st.markdown(f'<div class="agent-card"><div class="agent-title">🔎 Findings: {sub}</div>{notes}</div>', unsafe_allow_html=True)

    # --- Writer ---
    with trace_container:
        st.markdown('<div class="agent-card"><div class="agent-title">✍️ Writer Agent</div>Drafting the full report...</div>', unsafe_allow_html=True)
    try:
        draft = writer_agent(llm, topic, research_notes)
    except Exception as e:
        st.error(f"Writer Agent failed: {e}")
        st.stop()

    # --- Editor ---
    with trace_container:
        st.markdown('<div class="agent-card"><div class="agent-title">🪄 Editor Agent</div>Polishing final report...</div>', unsafe_allow_html=True)
    try:
        final_report = editor_agent(llm, draft)
    except Exception as e:
        final_report = draft
        st.warning(f"Editor Agent failed, showing unedited draft: {e}")

    st.markdown("---")
    st.header("📄 Final Report")
    st.markdown(final_report)

    st.download_button(
        "⬇️ Download Report (Markdown)",
        data=final_report,
        file_name=f"{topic[:40].strip().replace(' ', '_')}_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

else:
    st.info("Enter your API key and a topic above, then click **Run Agent Team** to see the agents work together.")
