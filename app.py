"""AI Research & Report Team — Streamlit UI.

This file is intentionally UI-only. All the actual logic lives in:
  schemas.py       - Pydantic data models
  search_tools.py  - web search
  rag.py           - document upload / chunking / embedding / retrieval
  agents.py        - Planner, Research, Fact Checker, Writer, Editor, Judge
  pipeline.py       - orchestrates the agents into one run_pipeline() call
"""

import pandas as pd
import streamlit as st

from rag import build_knowledge_base, hash_uploaded_files
from agents import get_llm, judge_agent
from pipeline import run_pipeline, analyze_citations

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
    .source-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #eee;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Research & Report Team")
st.caption("A multi-agent system: Planner → Researcher → Fact Checker → Writer → Editor (with citations)")

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get one free at aistudio.google.com")
    model_name = st.selectbox("Model", ["gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-flash-lite-latest"], index=0)
    num_search_results = st.slider("Web sources per sub-topic", 2, 8, 4)

    st.markdown("---")
    st.subheader("📁 Knowledge Base (RAG)")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="The Research Agent will pull findings from these documents in addition to the web.",
    )
    num_doc_chunks = st.slider("Document chunks per sub-topic", 1, 8, 4)
    index_clicked = st.button("📚 Index Documents", use_container_width=True)

    if "kb_chunks" not in st.session_state:
        st.session_state.kb_chunks = []
        st.session_state.kb_vectors = None
        st.session_state.kb_file_hash = None

    if st.session_state.kb_chunks:
        st.caption(f"✅ {len(st.session_state.kb_chunks)} chunks indexed from {len(uploaded_files or [])} file(s).")

    st.markdown("---")
    enable_fact_check = st.checkbox(
        "✅ Enable fact-checking",
        value=True,
        help="Adds a Fact Checker Agent that verifies each finding against its cited source before writing. Uses 1 extra LLM call per sub-topic.",
    )

    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. 🧭 Planner breaks your topic into sub-questions\n"
        "2. 🔎 Researcher searches the web + your documents, and cites each finding\n"
        "3. ✅ Fact Checker verifies each finding against its source\n"
        "4. ✍️ Writer drafts a full report with inline citations\n"
        "5. 🪄 Editor polishes and formats the final version\n"
        "6. 📚 Sources are listed at the end"
    )

# ----------------------------- KNOWLEDGE BASE INDEXING -----------------------------

if index_clicked:
    if not api_key:
        st.error("Enter your Gemini API key in the sidebar before indexing documents.")
    elif not uploaded_files:
        st.warning("Upload at least one PDF, DOCX, or TXT file first.")
    else:
        with st.spinner("Extracting, chunking, and embedding your documents..."):
            build_knowledge_base(uploaded_files, api_key)
            st.session_state.kb_file_hash = hash_uploaded_files(uploaded_files)
        st.success(f"Indexed {len(st.session_state.kb_chunks)} chunks from {len(uploaded_files)} file(s).")

# ----------------------------- MAIN RUN -----------------------------

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

    try:
        result = run_pipeline(
            topic, llm, api_key, num_search_results, num_doc_chunks, enable_fact_check,
            trace_container=trace_container,
        )
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    final_report_with_sources = result["final_report"]

    st.markdown("---")
    st.header("📄 Final Report")
    st.markdown(final_report_with_sources)

    with st.expander("📈 Run stats"):
        st.write(
            f"⏱️ {result['elapsed']:.1f}s · "
            f"🔎 Findings kept {result['findings_after']}/{result['findings_before']} · "
            f"🔤 Tokens {result['stats']['input_tokens']} in / {result['stats']['output_tokens']} out · "
            f"📞 {result['stats']['llm_calls']} LLM calls"
        )

    st.download_button(
        "⬇️ Download Report (Markdown)",
        data=final_report_with_sources,
        file_name=f"{topic[:40].strip().replace(' ', '_')}_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

else:
    st.info("Enter your API key and a topic above, then click **Run Agent Team** to see the agents work together.")


# ----------------------------- EVALUATION DASHBOARD -----------------------------

st.markdown("---")
with st.expander("📊 Evaluation Dashboard"):
    st.caption(
        "Runs the full pipeline across several test topics and scores each run: "
        "an independent LLM judge rates relevance/completeness, plus fact-check pass rate "
        "(faithfulness), citation accuracy, latency, and token usage."
    )
    default_topics = (
        "Impact of AI chatbots on small business customer support in 2026\n"
        "How remote work is reshaping urban housing markets\n"
        "The role of nuclear energy in reducing carbon emissions"
    )
    eval_topics_text = st.text_area("Test topics (one per line)", value=default_topics, height=110)
    eval_fact_check = st.checkbox("Fact-check during evaluation", value=enable_fact_check, key="eval_fc")
    run_eval = st.button("▶️ Run Evaluation", use_container_width=True)

    if run_eval:
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            eval_topics = [t.strip() for t in eval_topics_text.split("\n") if t.strip()]
            if not eval_topics:
                st.warning("Add at least one test topic.")
            else:
                eval_llm = get_llm(api_key, model_name)
                rows = []
                progress = st.progress(0.0)
                for idx, t in enumerate(eval_topics, 1):
                    with st.spinner(f"Running topic {idx}/{len(eval_topics)}: {t[:60]}"):
                        try:
                            run_result = run_pipeline(
                                t, eval_llm, api_key, num_search_results, num_doc_chunks,
                                eval_fact_check, trace_container=None,
                            )
                            judgement = judge_agent(eval_llm, t, run_result["final_report"], stats=run_result["stats"])
                            citation_stats = analyze_citations(run_result["final_report"], run_result["global_sources"])
                            pass_rate = run_result["fact_check_pass_rate"]
                            rows.append({
                                "Topic": t,
                                "Relevance (1-5)": judgement.score,
                                "Latency (s)": round(run_result["elapsed"], 1),
                                "Sub-topics": len(run_result["subtopics"]),
                                "Findings kept/total": f"{run_result['findings_after']}/{run_result['findings_before']}",
                                "Fact-Check Pass %": round(pass_rate * 100) if pass_rate is not None else None,
                                "Citations used": citation_stats["citations_used"],
                                "Invalid citations": citation_stats["invalid_citations"],
                                "Sources": len(run_result["global_sources"]),
                                "Input tokens": run_result["stats"]["input_tokens"],
                                "Output tokens": run_result["stats"]["output_tokens"],
                                "LLM calls": run_result["stats"]["llm_calls"],
                            })
                        except Exception as e:
                            st.warning(f"Evaluation run failed for '{t}': {e}")
                            rows.append({
                                "Topic": t, "Relevance (1-5)": None, "Latency (s)": None, "Sub-topics": None,
                                "Findings kept/total": None, "Fact-Check Pass %": None, "Citations used": None,
                                "Invalid citations": None, "Sources": None, "Input tokens": None,
                                "Output tokens": None, "LLM calls": None,
                            })
                    progress.progress(idx / len(eval_topics))

                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)

                    ok = df.dropna(subset=["Relevance (1-5)"])
                    if not ok.empty:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Avg Relevance", f"{ok['Relevance (1-5)'].astype(float).mean():.1f}/5")
                        c2.metric("Avg Latency", f"{ok['Latency (s)'].astype(float).mean():.1f}s")
                        if eval_fact_check and ok["Fact-Check Pass %"].notna().any():
                            c3.metric("Avg Fact-Check Pass", f"{ok['Fact-Check Pass %'].astype(float).mean():.0f}%")
                        else:
                            c3.metric("Avg Fact-Check Pass", "N/A")
                        total_tokens = ok["Input tokens"].astype(float).sum() + ok["Output tokens"].astype(float).sum()
                        c4.metric("Total Tokens", f"{int(total_tokens):,}")

                    st.download_button(
                        "⬇️ Download results (CSV)",
                        data=df.to_csv(index=False),
                        file_name="evaluation_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )