# AI Research & Report Team — Multi-Agent System

A 4-agent pipeline built with LangChain + Google Gemini + Streamlit:

**Planner → Researcher → Writer → Editor**

Give it a topic, and it plans sub-questions, searches the web for each,
drafts a report, then polishes it — all visible live in the UI.

---

## 1. Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Gemini API key from https://aistudio.google.com/apikey
(enter it in the sidebar when the app opens — no `.env` needed for local testing).

---

## 2. Deploy it for free (so clients can test it in a browser)

**Streamlit Community Cloud (recommended — free, 5 min setup):**

1. Push this folder to a new GitHub repo (public or private).
2. Go to https://share.streamlit.io → "New app" → connect your repo.
3. Set the main file to `app.py` → Deploy.
4. You'll get a live URL like `https://yourname-agent-team.streamlit.app`.

That URL is what you send to Fiverr clients — they open it, paste an API key
(theirs or a limited one you provide for demo), and run it themselves.

**Alternative:** Render or Railway also host Streamlit apps free-tier — useful if
you want a custom domain later.

---

## 3. Using this for your Fiverr gig

**Gig title ideas:**
- "I will build a custom multi-agent AI research and automation system"
- "I will build an AI agent team using LangChain and Gemini for your business"

**Gig description pitch:**
> I'll build you a multi-agent AI system where specialized agents (planner,
> researcher, writer, reviewer, etc.) collaborate to complete a task —
> content research, report generation, customer support triage, or a custom
> workflow for your business. Delivered as a working web app you can test
> immediately.

**Demo strategy:**
- Deploy this exact app and link it in your gig gallery/portfolio — let people
  try it live before they buy.
- Record a 30–60 sec screen recording of it running (Loom is free) — shows the
  "agents talking to each other" trace, which is the most impressive part.

**For paid orders — customize per client:**
- Swap the 4 agents for whatever the client's task needs (e.g., support ticket
  triage, product description generator, lead-qualification bot).
- Keep the same trace-UI pattern — clients love *seeing* the reasoning, not just
  getting a final answer.

---

## 4. Known limitations to mention to clients

- Uses DuckDuckGo's free search — no API key needed, but results can be rate
  limited on very heavy use. For production-scale, swap in a paid search API
  (Tavily, SerpAPI) — a good $$ upsell.
- Gemini free tier has request-per-minute limits — fine for demos, mention
  paid tier for production client use.
