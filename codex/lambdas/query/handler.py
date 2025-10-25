"""Query Lambda that orchestrates retrieval and response synthesis."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import boto3

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

BEDROCK = boto3.client("bedrock-runtime")
DYNAMODB = boto3.resource("dynamodb")
CHUNKS_TABLE = DYNAMODB.Table(os.environ["CHUNKS_TABLE"])
VECTOR_COLLECTION_ID = os.environ["VECTOR_COLLECTION_ID"]
SEARCH_COLLECTION_ID = os.environ["SEARCH_COLLECTION_ID"]

RESPONSE_MODEL_ID = os.environ.get("RESPONSE_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
TOP_K = int(os.environ.get("TOP_K", "5"))


@dataclass
class RetrievedChunk:
    """Container for chunk metadata returned to the caller."""

    chunk_id: str
    doc_id: str
    text: str
    score: float


def lambda_handler(event: Dict[str, Any], _context) -> Dict[str, Any]:
    """Handle API Gateway requests for question answering."""
    LOGGER.info("Received query event: %s", json.dumps(event))
    body = event.get("body") or {}
    if isinstance(body, str):
        body = json.loads(body or "{}")

    query = body.get("query")
    if not query:
        return {"statusCode": 400, "body": json.dumps({"error": "Missing query"})}

    retrieved = retrieve(query)
    answer = synthesize_answer(query=query, context_chunks=retrieved)
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "answer": answer,
                "chunks": [chunk.__dict__ for chunk in retrieved],
            }
        ),
    }


def retrieve(query: str) -> List[RetrievedChunk]:
    """Retrieve relevant chunks from OpenSearch Serverless."""
    # Placeholder retrieval: in production this would use signed HTTP calls to
    # the vector and search collections. Here we fall back to DynamoDB scan for
    # bootstrap and unit test friendliness.
    LOGGER.info(
        "Retrieving top %s chunks for query '%s' using collections %s/%s",
        TOP_K,
        query,
        SEARCH_COLLECTION_ID,
        VECTOR_COLLECTION_ID,
    )

    response = CHUNKS_TABLE.scan(Limit=TOP_K)
    items = response.get("Items", [])
    return [
        RetrievedChunk(
            chunk_id=item["chunk_id"],
            doc_id=item["doc_id"],
            text=item.get("text", ""),
            score=1.0,
        )
        for item in items
    ]


def synthesize_answer(*, query: str, context_chunks: Sequence[RetrievedChunk]) -> str:
    """Generate an answer grounded in retrieved context using Bedrock."""
    context = "\n\n".join(f"Chunk {chunk.chunk_id}: {chunk.text}" for chunk in context_chunks)
    prompt = (
        "You are an assistant that answers questions using the provided context.\n"
        "If the context is insufficient, say you do not know.\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    response = BEDROCK.invoke_model(
        body=json.dumps({"input": prompt, "anthropic_version": "bedrock-2023-05-31"}),
        modelId=RESPONSE_MODEL_ID,
        accept="application/json",
        contentType="application/json",
    )
    body = json.loads(response["body"].read())
    return body.get("output_text") or body.get("completion") or ""
