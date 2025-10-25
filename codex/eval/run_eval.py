"""Simple evaluation harness for the Codex API."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, List

import requests


def load_cases(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def run_case(*, session: requests.Session, endpoint: str, case: dict) -> dict:
    start = time.perf_counter()
    response = session.post(endpoint, json={"query": case["question"]}, timeout=30)
    latency = time.perf_counter() - start
    payload = response.json()
    return {
        "latency": latency,
        "answer": payload.get("answer", ""),
        "chunks": payload.get("chunks", []),
        "case": case,
    }


def percentile(data: List[float], q: float) -> float:
    """Return the q percentile (0-1) for the provided data."""
    if not data:
        return 0.0
    values = sorted(data)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return (1 - weight) * values[lower] + weight * values[upper]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Codex golden set.")
    parser.add_argument("endpoint", help="Codex /query endpoint")
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=Path(__file__).with_name("golden_set.jsonl"),
    )
    args = parser.parse_args()

    session = requests.Session()
    results: List[dict] = []
    for case in load_cases(args.golden_set):
        results.append(run_case(session=session, endpoint=args.endpoint, case=case))

    latencies = [result["latency"] for result in results]
    print(f"p50 latency: {percentile(latencies, 0.50):.3f}s")
    print(f"p95 latency: {percentile(latencies, 0.95):.3f}s")

    grounded = 0
    for result in results:
        expected = result["case"]["expected_answer"].lower()
        if expected in result["answer"].lower():
            grounded += 1
    hallucination_rate = 1 - (grounded / max(1, len(results)))
    print(f"Hallucination rate: {hallucination_rate:.2%}")


if __name__ == "__main__":
    main()
