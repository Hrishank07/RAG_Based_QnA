"""Lightweight Streamlit console that previews Codex activity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Sequence

import requests
import streamlit as st

API_URL = os.getenv("CODEX_API_URL", "").rstrip("/")

st.set_page_config(
    page_title="Codex Console (Preview)",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "About": "RAG pipeline preview console for Codex. Switch CODEX_API_URL to wire it up.",
    },
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float


def call_query_api(query: str) -> dict:
    """Fire the deployed /query endpoint if available, otherwise return a mock payload."""
    if not API_URL:
        return _mock_answer(query)

    try:
        response = requests.post(
            f"{API_URL.rstrip('/')}/query",
            json={"query": query},
            timeout=20,
        )
        response.raise_for_status()
        body = response.json()
        return {
            "answer": body.get("answer", "No answer returned"),
            "chunks": [
                Chunk(
                    chunk_id=chunk.get("chunk_id", ""),
                    doc_id=chunk.get("doc_id", ""),
                    text=chunk.get("text", ""),
                    score=chunk.get("score", 0.0),
                )
                for chunk in body.get("chunks", [])
            ],
            "source": "live-api",
        }
    except requests.RequestException as exc:
        return {
            "answer": f"Request failed: {exc}",
            "chunks": [],
            "source": "error",
        }


def _mock_answer(query: str) -> dict:
    """Fallback data when no backend is wired yet."""
    canned_chunks = [
        Chunk(
            chunk_id="00001",
            doc_id="security_playbook.pdf",
            text="Codex stores uploads inside a versioned S3 bucket named CodexDocs with SSL enforced.",
            score=0.92,
        ),
        Chunk(
            chunk_id="00007",
            doc_id="architecture.md",
            text="Chunk metadata lives in DynamoDB (Documents + Chunks tables) while vectors flow into OpenSearch Serverless.",
            score=0.87,
        ),
    ]
    answer = (
        "Codex ingests PDFs into a versioned S3 bucket, tracks document versions in DynamoDB, "
        "and fans chunks out to OpenSearch Serverless for hybrid retrieval. "
        "Open the admin console to monitor chunk counts and Textract usage."
    )
    return {"answer": answer, "chunks": canned_chunks, "source": "mock"}


def sample_ingestion_activity() -> dict:
    """Generate lightweight ingestion stats for the overview cards."""
    now = datetime.now(timezone.utc)
    documents = [
        {
            "doc_id": "employee-handbook.pdf",
            "status": "INDEXED",
            "chunks": 42,
            "updated_at": now - timedelta(minutes=3),
        },
        {
            "doc_id": "benefits.html",
            "status": "INDEXED",
            "chunks": 18,
            "updated_at": now - timedelta(minutes=12),
        },
        {
            "doc_id": "it-policy.docx",
            "status": "PROCESSING",
            "chunks": 0,
            "updated_at": now - timedelta(minutes=1),
        },
    ]
    metrics = {
        "docs_today": 6,
        "chunks_today": 214,
        "avg_latency_ms": 940,
        "token_usage": 138_000,
    }
    return {"documents": documents, "metrics": metrics}


def display_metrics(metrics: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Docs ingested (24h)", metrics["docs_today"])
    col2.metric("Chunks minted", metrics["chunks_today"])
    col3.metric("p50 ingest latency", f'{metrics["avg_latency_ms"]} ms')
    col4.metric("Tokens embedded", f'{metrics["token_usage"]:,}')


def display_document_table(documents: Sequence[dict]) -> None:
    st.dataframe(
        [
            {
                "Document": doc["doc_id"],
                "Status": doc["status"],
                "Chunks": doc["chunks"],
                "Updated": doc["updated_at"].strftime("%H:%M:%S"),
            }
            for doc in documents
        ],
        hide_index=True,
        use_container_width=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Environment")
        if API_URL:
            st.success(f"Connected to {API_URL}")
        else:
            st.info("CODEX_API_URL not set — running in mock mode.")
        st.markdown(
            """
            **Tips**
            - Point CODEX_API_URL to the deployed API Gateway base URL.
            - Keep this console open while uploading docs to watch status changes.
            - Use the feedback area below to capture operator notes.
            """
        )
        st.divider()
        note = st.text_area("Operator notes", placeholder="Record ingestion outcomes…")
        if st.button("Log note"):
            st.toast("Note captured (local preview only).")
            st.session_state["last_note"] = note


render_sidebar()

st.title("Codex Console (Preview)")
st.caption("Sneak peek into ingestion + retrieval health while the full admin console is being built.")

activity = sample_ingestion_activity()
display_metrics(activity["metrics"])

ask_tab, ingestion_tab, pipeline_tab = st.tabs(["Ask Codex", "Ingestion Feed", "Pipeline Anatomy"])

with ask_tab:
    st.subheader("Grounded answers in one place")
    user_query = st.text_area("Ask a question", placeholder="How does Codex store uploaded PDFs?")
    if st.button("Run query", type="primary"):
        if not user_query.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving context and synthesizing response…"):
                result = call_query_api(user_query.strip())
            st.markdown(f"**Answer ({result['source']}):** {result['answer']}")
            if result["chunks"]:
                st.markdown("**Citations**")
                for chunk in result["chunks"]:
                    st.write(f"• `{chunk.doc_id}` · Chunk {chunk.chunk_id} · score {chunk.score:.2f}")
                    st.caption(chunk.text[:280] + ("…" if len(chunk.text) > 280 else ""))
            else:
                st.info("No supporting chunks returned.")

with ingestion_tab:
    st.subheader("Recent ingestion events")
    display_document_table(activity["documents"])
    st.markdown("Use this panel to spot documents stuck in PROCESSING or FAILED.")

with pipeline_tab:
    st.subheader("How Codex answers questions")
    stages = [
        ("Upload", "Documents land inside a versioned, encrypted S3 bucket with SSL enforcement."),
        ("Understanding", "Textract + tokenizer extract text, then Bedrock Titan embeds each chunk."),
        ("Indexing", "Chunks land in DynamoDB for lineage and OpenSearch Serverless (BM25 + vector)."),
        ("Retrieval", "Query Lambda fans out to vector + keyword search, dedupes, and builds citations."),
        ("Generation", "Claude 3 Haiku (or fallback) writes grounded answers with numbered evidence."),
    ]
    for idx, (title, blurb) in enumerate(stages, start=1):
        st.markdown(f"**{idx}. {title}** — {blurb}")
