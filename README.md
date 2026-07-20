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

## 2. Deploy it for free (so it's accessible from a browser)

**Streamlit Community Cloud (recommended — free, 5 min setup):**

1. Push this folder to a GitHub repo (public or private).
2. Go to https://share.streamlit.io → "New app" → connect your repo.
3. Set the main file to `app.py` → Deploy.
4. You'll get a live URL like `https://yourname-agent-team.streamlit.app`.

**Alternative:** Render or Railway also host Streamlit apps on a free tier — useful
if you want a custom domain later.

---

## 3. Project structure

```
app.py            # Main Streamlit app + agent logic
requirements.txt  # Python dependencies
.gitignore        # Excludes venv, cache, and secrets from git
```

---

## 4. Known limitations

- Uses DuckDuckGo's free search — no API key needed, but results can be rate
  limited under heavy use. For production-scale use, swap in a paid search API
  (Tavily, SerpAPI).
- Gemini's free tier has request-per-minute and per-day limits. Google updates
  which specific model names are free-tier eligible fairly often — check
  https://ai.google.dev/gemini-api/docs/pricing if a model returns a 404 or
  quota error, and update the model list in `app.py` accordingly.
