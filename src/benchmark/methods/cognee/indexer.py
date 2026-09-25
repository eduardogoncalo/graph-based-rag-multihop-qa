from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmark.methods.cognee.adapter import CogneeAdapter


@dataclass(frozen=True)
class CogneeIndexRecord:
    document_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] | None = None

    def to_payload(self) -> str:
        metadata = dict(self.metadata or {})
        metadata.update({"document_id": self.document_id, "title": self.title})
        return (
            f"document_id: {self.document_id}\n"
            f"title: {self.title or ''}\n"
            f"metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n\n"
            f"{self.text}"
        )


async def index_records_async(
    *,
    adapter: CogneeAdapter,
    records: Iterable[CogneeIndexRecord],
    chunk_size: int | None = None,
) -> dict[str, Any]:
    payloads = [record.to_payload() for record in records]
    started = time.perf_counter()
    kwargs: dict[str, Any] = {}
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    result = await adapter.remember_async(payloads, **kwargs)
    return {
        "status": "completed",
        "records_requested": len(payloads),
        "latency_seconds": time.perf_counter() - started,
        "raw_result": result,
    }


def load_document_records(path: str | Path, *, limit: int | None = None) -> list[CogneeIndexRecord]:
    records: list[CogneeIndexRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                CogneeIndexRecord(
                    document_id=row["document_id"],
                    title=row.get("title"),
                    text=row["text"],
                    metadata=row.get("metadata") or {},
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records
