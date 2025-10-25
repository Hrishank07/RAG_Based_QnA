"""Lambda responsible for ingesting and chunking documents."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError, ClientError
from botocore.session import Session

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
SEARCH_COLLECTION_ENDPOINT = os.environ.get("SEARCH_COLLECTION_ENDPOINT")
VECTOR_COLLECTION_ENDPOINT = os.environ.get("VECTOR_COLLECTION_ENDPOINT")
SEARCH_INDEX_NAME = os.environ.get("SEARCH_INDEX_NAME", "codex-chunks-bm25")
VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "codex-chunks-vector")

EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))
EMBEDDING_MAX_RETRIES = int(os.environ.get("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_RETRY_BACKOFF = float(os.environ.get("EMBEDDING_RETRY_BACKOFF", "1.5"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
OPENSEARCH_BATCH_SIZE = int(os.environ.get("OPENSEARCH_BATCH_SIZE", "50"))

ASYNC_TEXTRACT_THRESHOLD = int(os.environ.get("ASYNC_TEXTRACT_THRESHOLD", str(5 * 1024 * 1024)))
TEXTRACT_POLL_DELAY = int(os.environ.get("TEXTRACT_POLL_DELAY", "5"))
TEXTRACT_MAX_POLLS = int(os.environ.get("TEXTRACT_MAX_POLLS", "60"))


@dataclass
class Chunk:
    """Represents a logical chunk of a document."""

    document_id: str
    chunk_id: str
    text: str
    token_count: int
    section: str | None = None
    version: str | None = None
    tags: Sequence[str] | None = None


class OpenSearchIndexer:
    """Signed HTTP helper that keeps BM25 and vector indexes in sync."""

    def __init__(
        self,
        *,
        search_endpoint: str,
        vector_endpoint: str,
        search_index: str,
        vector_index: str,
        embedding_dimension: int,
        batch_size: int,
    ) -> None:
        botocore_session = Session()
        region = (
            botocore_session.region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        credentials = botocore_session.get_credentials()
        if credentials is None:
            raise RuntimeError("Missing AWS credentials for OpenSearch signing.")

        self.search_endpoint = search_endpoint.rstrip("/")
        self.vector_endpoint = vector_endpoint.rstrip("/")
        self.search_index = search_index
        self.vector_index = vector_index
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self._region = region
        self._credentials = credentials.get_frozen_credentials()
        self._http = requests.Session()
        self._indexes_ensured = False

    def ensure_indexes(self) -> None:
        if self._indexes_ensured:
            return
        self._ensure_index(
            endpoint=self.search_endpoint,
            index_name=self.search_index,
            body={
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "version": {"type": "keyword"},
                        "section": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "text": {"type": "text"},
                        "tokens": {"type": "integer"},
                        "updated_at": {"type": "date"},
                    }
                }
            },
        )
        self._ensure_index(
            endpoint=self.vector_endpoint,
            index_name=self.vector_index,
            body={
                "settings": {
                    "index.knn": True,
                },
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "version": {"type": "keyword"},
                        "section": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "text": {"type": "text"},
                        "tokens": {"type": "integer"},
                        "updated_at": {"type": "date"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.embedding_dimension,
                            "method": {
                                "name": "hnsw",
                                "engine": "nmslib",
                                "space_type": "cosinesimil",
                            },
                        },
                    }
                },
            },
        )
        self._indexes_ensured = True

    def bulk_index(self, chunk_items: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunk_items:
            return
        if len(chunk_items) != len(embeddings):
            raise ValueError("Chunk items and embeddings length mismatch.")
        self.ensure_indexes()

        for start in range(0, len(chunk_items), self.batch_size):
            end = min(len(chunk_items), start + self.batch_size)
            subset = chunk_items[start:end]
            subset_embeddings = embeddings[start:end]
            search_body = []
            vector_body = []

            for chunk, embed in zip(subset, subset_embeddings):
                document_key = f"{chunk.document_id}#{chunk.chunk_id}"
                doc_payload = {
                    "doc_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "version": chunk.version,
                    "section": chunk.section,
                    "tags": list(chunk.tags or []),
                    "text": chunk.text,
                    "tokens": chunk.token_count,
                    "updated_at": int(time.time()),
                }
                search_body.append(json.dumps({"index": {"_index": self.search_index, "_id": document_key}}))
                search_body.append(json.dumps(doc_payload))

                vector_doc = dict(doc_payload)
                vector_doc["embedding"] = embed
                vector_body.append(json.dumps({"index": {"_index": self.vector_index, "_id": document_key}}))
                vector_body.append(json.dumps(vector_doc))

            self._bulk_request(self.search_endpoint, "\n".join(search_body) + "\n")
            self._bulk_request(self.vector_endpoint, "\n".join(vector_body) + "\n")

    def _ensure_index(self, *, endpoint: str, index_name: str, body: dict) -> None:
        head = self._request(
            endpoint=endpoint,
            method="HEAD",
            path=f"/{index_name}",
            expected_status_codes=(200, 404),
        )
        if head.status_code == 200:
            return
        LOGGER.info("Creating OpenSearch index %s at %s", index_name, endpoint)
        self._request(
            endpoint=endpoint,
            method="PUT",
            path=f"/{index_name}",
            body=json.dumps(body),
            headers={"content-type": "application/json"},
        )

    def _bulk_request(self, endpoint: str, payload: str) -> None:
        self._request(
            endpoint=endpoint,
            method="POST",
            path="/_bulk",
            body=payload,
            headers={"content-type": "application/x-ndjson"},
        )

    def _request(
        self,
        *,
        endpoint: str,
        method: str,
        path: str,
        body: str | None = None,
        headers: dict[str, str] | None = None,
        expected_status_codes: tuple[int, ...] = (200, 201, 202),
    ) -> requests.Response:
        url = f"{endpoint}{path}"
        request = AWSRequest(method=method, url=url, data=body, headers=headers or {})
        SigV4Auth(self._credentials, "aoss", self._region).add_auth(request)
        prepared = request.prepare()
        response = self._http.send(prepared)
        if response.status_code not in expected_status_codes:
            LOGGER.error(
                "OpenSearch request failed: %s %s %s %s",
                method,
                url,
                response.status_code,
                response.text,
            )
            response.raise_for_status()
        return response


INDEXER: OpenSearchIndexer | None = None
if SEARCH_COLLECTION_ENDPOINT and VECTOR_COLLECTION_ENDPOINT:
    try:
        INDEXER = OpenSearchIndexer(
            search_endpoint=SEARCH_COLLECTION_ENDPOINT,
            vector_endpoint=VECTOR_COLLECTION_ENDPOINT,
            search_index=SEARCH_INDEX_NAME,
            vector_index=VECTOR_INDEX_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
            batch_size=OPENSEARCH_BATCH_SIZE,
        )
    except Exception as exc:  # pragma: no cover - initialization occurs in AWS
        LOGGER.warning("Failed to initialize OpenSearch indexer: %s", exc)
else:
    LOGGER.info("OpenSearch endpoints unavailable; skipping index initialization.")


def lambda_handler(event, _context):
    """Entrypoint for the S3 triggered ingestion."""
    LOGGER.info("Received event: %s", json.dumps(event))
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        if bucket != DOCS_BUCKET:
            LOGGER.warning("Skipping object from unexpected bucket %s", bucket)
            continue
        try:
            process_object(bucket=bucket, key=key)
        except Exception:
            LOGGER.exception("Failed to ingest %s/%s", bucket, key)
            raise


def process_object(*, bucket: str, key: str) -> None:
    """Process a newly uploaded object."""
    document_id = key.split("/")[-1]
    version = str(int(time.time()))
    start_time = time.time()

    body = S3.get_object(Bucket=bucket, Key=key)["Body"].read()
    register_document(
        document_id=document_id,
        version=version,
        chunk_count=0,
        content_hash=str(hash(body)),
    )

    try:
        text = extract_text(bucket=bucket, key=key, content=body)
        chunks = list(chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))

        chunk_items = []
        token_total = 0
        for idx, chunk in enumerate(chunks):
            tokens = len(chunk.split())
            token_total += tokens
            chunk_items.append(
                Chunk(
                    document_id=document_id,
                    chunk_id=f"{idx:05d}",
                    text=chunk,
                    token_count=tokens,
                    section=None,
                    version=version,
                    tags=None,
                )
            )

        embeddings = embed_chunks(chunk_items)
        write_chunks(chunk_items=chunk_items, embeddings=embeddings)
        update_document_status(
            document_id=document_id,
            version=version,
            status="INDEXED",
            chunk_count=len(chunk_items),
        )
        record_metrics(
            document_id=document_id,
            chunk_count=len(chunk_items),
            token_total=token_total,
            duration=time.time() - start_time,
            status="success",
        )
    except Exception as exc:
        update_document_status(
            document_id=document_id,
            version=version,
            status="FAILED",
            error=str(exc),
        )
        record_metrics(
            document_id=document_id,
            chunk_count=0,
            token_total=0,
            duration=time.time() - start_time,
            status="failed",
        )
        raise


def extract_text(*, bucket: str, key: str, content: bytes) -> str:
    """Extract text from the uploaded document."""
    if not key.lower().endswith(".pdf"):
        LOGGER.info("Treating %s as UTF-8 text", key)
        return content.decode("utf-8", errors="ignore")

    if len(content) <= ASYNC_TEXTRACT_THRESHOLD:
        LOGGER.info("Running synchronous Textract on %s", key)
        response = TEXTRACT.detect_document_text(Document={"Bytes": content})
        return _textract_lines(response.get("Blocks", []))

    LOGGER.info("Running asynchronous Textract on %s", key)
    job = TEXTRACT.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
    )
    job_id = job["JobId"]
    attempts = 0
    while attempts < TEXTRACT_MAX_POLLS:
        response = TEXTRACT.get_document_text_detection(JobId=job_id, MaxResults=1000)
        status = response.get("JobStatus")
        if status == "SUCCEEDED":
            return _drain_textract_results(job_id, first_page=response)
        if status == "FAILED":
            raise RuntimeError(f"Textract failed for {key}")
        attempts += 1
        time.sleep(TEXTRACT_POLL_DELAY)
    raise TimeoutError(f"Textract timed out for {key}")


def _textract_lines(blocks: Sequence[dict]) -> str:
    lines = [block["Text"] for block in blocks if block.get("BlockType") == "LINE"]
    return "\n".join(lines)


def _drain_textract_results(job_id: str, first_page: dict) -> str:
    lines: List[str] = []
    first_lines = _textract_lines(first_page.get("Blocks", []))
    if first_lines:
        lines.extend(first_lines.splitlines())
    next_token = first_page.get("NextToken")
    while next_token:
        page = TEXTRACT.get_document_text_detection(JobId=job_id, MaxResults=1000, NextToken=next_token)
        page_lines = _textract_lines(page.get("Blocks", []))
        if page_lines:
            lines.extend(page_lines.splitlines())
        next_token = page.get("NextToken")
    return "\n".join(lines)


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


def update_document_status(
    *,
    document_id: str,
    version: str,
    status: str,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    """Update status fields for a document record."""
    expression = ["#status = :status", "updated_at = :updated_at"]
    values = {
        ":status": status,
        ":updated_at": int(time.time()),
    }
    names = {"#status": "status"}

    if chunk_count is not None:
        expression.append("chunk_count = :chunk_count")
        values[":chunk_count"] = chunk_count
    if error:
        expression.append("error = :error")
        values[":error"] = error[:512]

    DOCUMENTS_TABLE.update_item(
        Key={"doc_id": document_id, "version": version},
        UpdateExpression="SET " + ", ".join(expression),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def embed_chunks(chunks: Sequence[Chunk]) -> List[List[float]]:
    """Generate embeddings for each chunk using Bedrock with retries."""
    embeddings: List[List[float]] = []
    for chunk in chunks:
        embeddings.append(_embed_with_retries(chunk.text))
    return embeddings


def _embed_with_retries(text: str) -> List[float]:
    delay = EMBEDDING_RETRY_BACKOFF
    last_error: Exception | None = None
    for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
        try:
            payload = {"inputText": text}
            response = BEDROCK.invoke_model(
                body=json.dumps(payload),
                modelId=EMBEDDING_MODEL_ID,
                accept="application/json",
                contentType="application/json",
            )
            body = json.loads(response["body"].read())
            vector = body.get("embedding") or body.get("vector")
            if not vector:
                raise RuntimeError("Bedrock response missing embedding vector.")
            return vector
        except (BotoCoreError, ClientError, RuntimeError) as exc:
            last_error = exc
            LOGGER.warning("Embedding attempt %s failed: %s", attempt, exc)
            if attempt == EMBEDDING_MAX_RETRIES:
                break
            time.sleep(delay)
            delay *= 2
    if last_error:
        raise last_error
    raise RuntimeError("Embedding failed without exception.")


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
                "tokens": chunk.token_count,
                "embedding": list(embedding),
            }
        )

    if INDEXER is None:
        LOGGER.warning(
            "OpenSearch indexer unavailable; %s chunks not indexed (collections %s/%s).",
            len(chunk_items),
            SEARCH_COLLECTION_ID,
            VECTOR_COLLECTION_ID,
        )
        return

    INDEXER.bulk_index(chunk_items, embeddings)
    LOGGER.info(
        "Indexed %s chunks into collections %s/%s",
        len(chunk_items),
        SEARCH_COLLECTION_ID,
        VECTOR_COLLECTION_ID,
    )


def record_metrics(
    *,
    document_id: str,
    chunk_count: int,
    token_total: int,
    duration: float,
    status: str,
) -> None:
    """Emit a lightweight structured metrics log."""
    LOGGER.info(
        "METRIC|ingest|%s",
        json.dumps(
            {
                "doc_id": document_id,
                "chunk_count": chunk_count,
                "token_total": token_total,
                "duration_ms": int(duration * 1000),
                "status": status,
                "timestamp": int(time.time()),
            }
        ),
    )
