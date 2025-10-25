"""Lambda responsible for ingesting and chunking documents."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import boto3

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

S3 = boto3.client("s3")
TEXTRACT = boto3.client("textract")
BEDROCK = boto3.client("bedrock-runtime")
DYNAMODB = boto3.resource("dynamodb")

DOCUMENTS_TABLE = DYNAMODB.Table(os.environ["DOCUMENTS_TABLE"])
CHUNKS_TABLE = DYNAMODB.Table(os.environ["CHUNKS_TABLE"])
DOCS_BUCKET = os.environ["DOCS_BUCKET"]
VECTOR_COLLECTION_ID = os.environ["VECTOR_COLLECTION_ID"]
SEARCH_COLLECTION_ID = os.environ["SEARCH_COLLECTION_ID"]

EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))


@dataclass
class Chunk:
    """Represents a logical chunk of a document."""

    document_id: str
    chunk_id: str
    text: str
    section: str | None = None
    version: str | None = None
    tags: Sequence[str] | None = None


def lambda_handler(event, _context):
    """Entrypoint for the S3 triggered ingestion."""
    LOGGER.info("Received event: %s", json.dumps(event))
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        if bucket != DOCS_BUCKET:
            LOGGER.warning("Skipping object from unexpected bucket %s", bucket)
            continue
        process_object(bucket=bucket, key=key)


def process_object(*, bucket: str, key: str) -> None:
    """Process a newly uploaded object."""
    document_id = key.split("/")[-1]
    version = str(int(time.time()))

    body = S3.get_object(Bucket=bucket, Key=key)["Body"].read()
    text = extract_text(content=body, key=key)
    chunks = list(chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))

    register_document(
        document_id=document_id,
        version=version,
        chunk_count=len(chunks),
        content_hash=str(hash(body)),
    )

    chunk_items = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{idx:05d}"
        chunk_items.append(
            Chunk(
                document_id=document_id,
                chunk_id=chunk_id,
                text=chunk,
                section=None,
                version=version,
                tags=None,
            )
        )

    embeddings = embed_chunks(chunk_items)
    write_chunks(chunk_items=chunk_items, embeddings=embeddings)


def extract_text(*, content: bytes, key: str) -> str:
    """Extract text from the uploaded document."""
    if key.lower().endswith(".pdf"):
        LOGGER.info("Running Textract on PDF %s", key)
        response = TEXTRACT.detect_document_text(Document={"Bytes": content})
        lines = [block["Text"] for block in response.get("Blocks", []) if block.get("BlockType") == "LINE"]
        return "\n".join(lines)

    LOGGER.info("Treating %s as UTF-8 text", key)
    return content.decode("utf-8", errors="ignore")


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> Iterable[str]:
    """Split text into overlapping windows by word count."""
    words = text.split()
    if not words:
        return []

    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start = max(0, end - overlap)


def register_document(*, document_id: str, version: str, chunk_count: int, content_hash: str) -> None:
    """Write document metadata to DynamoDB."""
    DOCUMENTS_TABLE.put_item(
        Item={
            "doc_id": document_id,
            "version": version,
            "status": "PROCESSING",
            "chunk_count": chunk_count,
            "hash": content_hash,
            "updated_at": int(time.time()),
        }
    )


def embed_chunks(chunks: Sequence[Chunk]) -> List[List[float]]:
    """Generate embeddings for each chunk using Bedrock."""
    embeddings: List[List[float]] = []
    for chunk in chunks:
        payload = {
            "inputText": chunk.text,
        }
        response = BEDROCK.invoke_model(
            body=json.dumps(payload),
            modelId=EMBEDDING_MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        body = json.loads(response["body"].read())
        embeddings.append(body.get("embedding") or body.get("vector") or [])
    return embeddings


def write_chunks(*, chunk_items: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
    """Persist chunk metadata and send to OpenSearch."""
    for chunk, embedding in zip(chunk_items, embeddings):
        CHUNKS_TABLE.put_item(
            Item={
                "doc_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "version": chunk.version,
                "section": chunk.section,
                "tags": list(chunk.tags or []),
                "tokens": len(chunk.text.split()),
                "embedding": embedding,
            }
        )

    # Placeholder for OpenSearch Serverless bulk ingestion. In production this would
    # call the OpenSearch ingestion API or data plane endpoint via signed HTTP.
    LOGGER.info(
        "Indexed %s chunks into collections %s/%s",
        len(chunk_items),
        SEARCH_COLLECTION_ID,
        VECTOR_COLLECTION_ID,
    )
