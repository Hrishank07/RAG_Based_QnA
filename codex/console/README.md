# Codex Console (Preview)

This directory houses a lightweight Streamlit app that previews how the Codex admin
console will look and feel. It offers:

- A “Ask Codex” panel that hits the deployed `/query` endpoint (or falls back to mock data).
- Ingestion health cards + tables that visualize document flow at a glance.
- A narrated pipeline walkthrough so stakeholders can grok the architecture.

## Run locally

```bash
pip install -r requirements-dev.txt
export CODEX_API_URL="https://your-api-id.execute-api.region.amazonaws.com/prod"
streamlit run codex/console/app.py
```

Omitting `CODEX_API_URL` keeps the app in mock mode with sample answers and stats—useful for
quick demos while the backend is still bootstrapping.
