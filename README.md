# AI Research & Report Team

A multi-agent GenAI application that turns a single topic into a fully cited, fact-checked research report — combining live web search with your own uploaded documents (RAG), and including a built-in evaluation dashboard to measure output quality.

Built with **LangChain**, **Google Gemini API**, and **Streamlit**.

---

## What it does

Give it a topic (e.g. *"Impact of AI chatbots on small business customer support in 2026"*) and it will:

1. Break the topic into focused sub-questions
2. Research each one using live web search **and** any documents you've uploaded
3. Fact-check every claim against the actual source text before it's allowed into the report
4. Write a structured draft with inline citations
5. Polish it into a final report with an executive summary, key takeaways, and a numbered sources list

Every claim in the final report is traceable back to a real source — either a web page or an exact page in one of your uploaded documents.

---

## Architecture

```
                         ┌───────────────┐
                         │     User      │
                         └───────┬───────┘
                                 ↓
                         ┌───────────────┐
                         │ Planner Agent │
                         └───────┬───────┘
                                 ↓
              ┌──────────────────┼──────────────────┐
              ↓                                      ↓
       Web Search (DDGS)                   Document RAG (uploaded PDFs/DOCX/TXT)
              │                                      │
              └──────────────────┬───────────────────┘
                                 ↓
                        ┌────────────────┐
                        │ Research Agent │  → structured findings, each tied to a source
                        └───────┬────────┘
                                ↓
                        ┌────────────────┐
                        │ Fact Checker   │  → verifies each claim against its cited source
                        └───────┬────────┘
                                ↓
                        ┌────────────────┐
                        │  Writer Agent  │  → drafts report with inline [n] citations
                        └───────┬────────┘
                                ↓
                        ┌────────────────┐
                        │  Editor Agent  │  → polishes, adds summary + key takeaways
                        └───────┬────────┘
                                ↓
                     ┌─────────────────────┐
                     │ Report + Citations  │
                     └─────────────────────┘
```

An independent **Judge Agent** scores report relevance separately, used only by the Evaluation Dashboard — it's not part of the report-writing chain, so it can't grade its own work.

---

## Key features

- **Multi-agent pipeline** — Planner → Researcher → Fact Checker → Writer → Editor, each with a single clear responsibility
- **RAG (Retrieval-Augmented Generation)** — upload PDF/DOCX/TXT files; the Research Agent pulls findings from them alongside web results, citing exact page numbers for PDFs
- **Structured output** — agents return validated Pydantic objects (`Finding`, `ResearchOutput`, `FactCheckOutput`), not free-text strings, using LangChain's `with_structured_output`
- **Inline citations** — every factual claim ends with a `[n]` marker linking to a deduplicated, numbered source list at the end of the report
- **Hallucination detection** — the Fact Checker Agent independently verifies each finding against the actual text of the source it cited, and drops anything unsupported before it reaches the Writer
- **Evaluation Dashboard** — run the full pipeline across multiple test topics and get:
  - Relevance score (1–5) from an independent LLM judge
  - Fact-check pass rate (faithfulness)
  - Citation accuracy (valid vs. invalid `[n]` references)
  - Latency, token usage, and LLM call count per run
  - Exportable as CSV
- **Lightweight vector store** — document embeddings are stored as a plain NumPy array with cosine similarity search, no external vector DB required

---

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini API (`gemini-2.5-flash-lite` / `gemini-3-flash-preview`) |
| Embeddings | `gemini-embedding-001` |
| Orchestration | LangChain (`langchain-google-genai`, structured output) |
| Web search | `ddgs` (DuckDuckGo Search) |
| Document parsing | `pypdf`, `python-docx` |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Vector search | NumPy (cosine similarity, in-memory) |
| UI | Streamlit |
| Data/eval | Pandas |

---

## Project structure

The codebase is split into focused modules rather than one large script:

```
├── app.py            # Streamlit UI only — sidebar, main run flow, evaluation dashboard
├── schemas.py         # Pydantic models shared across the pipeline
├── search_tools.py    # Web search (DDGS) + prompt formatting
├── rag.py              # Document upload, chunking, embedding, vector search
├── agents.py           # Planner, Research, Fact Checker, Writer, Editor, Judge agents
└── pipeline.py          # run_pipeline() — orchestrates all agents into one run
```

`app.py` contains no business logic — it only renders UI and calls `run_pipeline()`. This means the same pipeline function powers both the single-topic "Run Agent Team" button and the batch Evaluation Dashboard, so there's one implementation to maintain, not two.

---

## Setup

**1. Clone and enter the project folder**

```bash
git clone https://github.com/im-nazim/multi-agent-report-team.git
cd multi-agent-report-team
```

**2. Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

**3. Install dependencies**

```bash
pip install streamlit langchain langchain-google-genai langchain-text-splitters ddgs pydantic pypdf python-docx numpy pandas
```

**4. Get a free Gemini API key**

Visit [aistudio.google.com](https://aistudio.google.com) to generate one.

**5. Run the app**

```bash
streamlit run app.py
```

Paste your API key into the sidebar and enter a topic to get started.

---

## Using RAG (document upload)

1. In the sidebar, upload one or more PDF, DOCX, or TXT files under **Knowledge Base (RAG)**
2. Click **Index Documents** — this extracts, chunks, and embeds the text (shown as a chunk count once done)
3. Run a topic related to the document's content — the Research Agent will pull findings from it and cite the exact page (for PDFs) in the final Sources list

Indexing is manual (button-triggered) rather than automatic, to avoid re-embedding on every UI interaction and burning API quota unnecessarily.

---

## Using the Evaluation Dashboard

Expand **📊 Evaluation Dashboard** at the bottom of the app, enter one or more test topics (one per line), and click **Run Evaluation**. Each topic runs the full pipeline independently and is scored on relevance, faithfulness, citation accuracy, latency, and token cost. Results can be exported as CSV for tracking regressions across changes.

Note: each test topic runs the *entire* multi-agent pipeline, so evaluation cost/time scales linearly with the number of topics — start with 3–5 before scaling up.

---

## Known limitations

- **Free-tier rate limits**: a single report run makes roughly 12–15 LLM calls (Planner + Research × N subtopics + Fact Check × N + Writer + Editor). Running the Evaluation Dashboard across several topics can hit Gemini's free-tier rate limits quickly — consider `gemini-2.5-flash-lite`, which currently has the most generous free allowance.
- **Web search reliability**: `ddgs` scrapes DuckDuckGo rather than using an official API, so it can occasionally rate-limit or return thin results. A production version would swap this for a dedicated search API (e.g. Tavily, Serper).
- **DOCX/TXT citations** don't include page numbers (only PDFs do), since those formats don't have a native page concept.

---

## Roadmap / possible next steps

- Swap DuckDuckGo scraping for a dedicated search API (Tavily/Serper) for reliability
- Add conversational memory ("make the report shorter", "add more on section 3")
- Persist the knowledge base with a real vector DB (Chroma/FAISS) instead of in-memory NumPy, for larger document sets
- FastAPI backend + Docker for production deployment
- Automated test suite + CI

---

## Live demo

Deployed on Streamlit Community Cloud: *(add your deployment link here)*

## Author

Muhammad Nazim — [GitHub](https://github.com/im-nazim) · muhammadnazim.workspace@gmail.com