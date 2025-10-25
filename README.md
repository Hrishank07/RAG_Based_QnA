# Codex RAG Q&A

Codex is a retrieval-augmented generation (RAG) stack that delivers low-latency, grounded answers over private documents. The repository is structured as a mono-repo containing infrastructure-as-code, application Lambdas, an OpenAPI specification, evaluation tooling, and the future administrative console.

## Repository Layout

```
codex/
  infra/        # AWS CDK application
  lambdas/      # Ingestion and query Lambda sources
  api/          # OpenAPI specification for Codex APIs
  eval/         # Golden set and evaluation harness
  console/      # Placeholder for the web console
```

## Getting Started

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements-dev.txt`.
3. Run the test suite via `pytest`.
4. Synthesize the infrastructure with `cdk synth --app "python -m codex.infra.app"`.

CI is configured via GitHub Actions to run linting, unit tests, and `cdk synth` on every pull request. Deployments to the `main` branch are gated behind a manual approval step.

## Preview console

Want a sneak peek into the future admin experience? Launch the Streamlit prototype:

```bash
pip install -r requirements-dev.txt
streamlit run codex/console/app.py
```

Set `CODEX_API_URL` to wire the “Ask Codex” panel to your deployed `/query` endpoint, otherwise the app falls back to curated mock data so you can still demo the pipeline story end-to-end.
