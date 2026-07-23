"""Persistent memory store: SQLite-backed vectors + metadata lookups.

Records hold text, JSON metadata, and an embedding vector; semantic recall is
cosine similarity over those vectors. The implementation is dependency-free —
vectors are stored as JSON and scored in Python — so it works out of the box;
for larger stores the same schema can be backed by the ``sqlite-vec`` extension
without changing the interface.

The embedder is injectable. The default :class:`HashingEmbedder` is deterministic
and needs no model (good for tests and offline use); pass the Phase-2
:class:`~omniord.router.router.Router` (whose ``embed`` matches the protocol) for
real embeddings.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

_WORD = re.compile(r"[a-z0-9]{2,}")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """A deterministic, model-free bag-of-words hashing embedder."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vec[int(digest, 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryRecord(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float


class MemoryStore:
    """A persistent semantic memory backed by SQLite."""

    def __init__(self, path: Path, *, embedder: Embedder | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                metadata   TEXT NOT NULL,
                embedding  TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    # ---------------- writes ----------------

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        embedding = (await self.embedder.embed([text]))[0]
        record_id = uuid.uuid4().hex[:16]
        created_at = time.time()
        await asyncio.to_thread(
            self._insert, record_id, text, metadata or {}, embedding, created_at
        )
        return record_id

    def _insert(
        self,
        record_id: str,
        text: str,
        metadata: dict[str, Any],
        embedding: list[float],
        created_at: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, text, metadata, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (record_id, text, json.dumps(metadata), json.dumps(embedding), created_at),
            )
            self._conn.commit()

    def delete(self, record_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ---------------- reads ----------------

    async def search(
        self, query: str, k: int = 5, *, where: dict[str, Any] | None = None
    ) -> list[tuple[MemoryRecord, float]]:
        """Return up to ``k`` records most similar to ``query`` (with scores)."""
        query_vec = (await self.embedder.embed([query]))[0]
        rows = await asyncio.to_thread(self._fetch_all)
        scored: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            metadata = json.loads(row["metadata"])
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            similarity = _cosine(query_vec, json.loads(row["embedding"]))
            if similarity <= 0.0:
                continue
            scored.append((self._to_record(row, metadata), similarity))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def get(self, record_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (record_id,)
            ).fetchone()
        return self._to_record(row, json.loads(row["metadata"])) if row else None

    def all(self) -> list[MemoryRecord]:
        rows = self._fetch_all()
        return [self._to_record(row, json.loads(row["metadata"])) for row in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def _fetch_all(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM memories").fetchall()

    @staticmethod
    def _to_record(row: sqlite3.Row, metadata: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            text=row["text"],
            metadata=metadata,
            created_at=row["created_at"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
