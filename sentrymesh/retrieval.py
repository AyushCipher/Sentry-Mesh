"""Precedent retrieval over past *resolved* cases.

Two interchangeable backends behind one interface:

``pgvector``
    A PostgreSQL table with a ``vector`` column and an ivfflat index, queried
    with the ``<=>`` cosine-distance operator.  Used when ``DATABASE_URL``
    points at a reachable database that has the ``vector`` extension available.

``local_cosine``
    An in-process NumPy cosine search over the same corpus and the same
    embeddings.  Used when no such database is reachable.  Results are
    numerically equivalent at this corpus size (a few dozen precedents); the
    difference is operational, not semantic.

Whichever backend answered is recorded on every tool result and in the harness
report, so a reader is never left guessing which path produced a number.

EMBEDDINGS ARE A DOCUMENTED MOCK.  ``embed()`` is a deterministic hashed
bag-of-words projection, not a learned text embedding.  It captures lexical
overlap between case summaries and nothing deeper - two cases describing the
same fraud pattern in different words will not match well.  Replacing it with a
real embedding model is a one-function change; the honest consequence of not
having done so is stated in the README's Known Limitations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentrymesh.config import PRECEDENT_CORPUS, SETTINGS

EMBED_DIM = 256
_TOKEN = re.compile(r"[a-z0-9_]+")


def embed(text: str) -> List[float]:
    """Deterministic hashed bag-of-words vector (MOCK - see module docstring)."""
    vec = [0.0] * EMBED_DIM
    tokens = _TOKEN.findall((text or "").lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % EMBED_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_corpus(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    corpus_path = path or PRECEDENT_CORPUS
    if not corpus_path.exists():
        return []
    with corpus_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("precedents", data) if isinstance(data, dict) else data


def precedent_text(row: Dict[str, Any]) -> str:
    return " ".join(
        str(x)
        for x in (
            row.get("category"),
            row.get("summary"),
            " ".join(row.get("signals", [])),
            row.get("resolution_notes"),
        )
        if x
    )


class LocalCosineIndex:
    """In-process cosine search over the precedent corpus."""

    name = "local_cosine"

    def __init__(self, corpus: List[Dict[str, Any]]) -> None:
        self.corpus = corpus
        self._vectors = [embed(precedent_text(row)) for row in corpus]

    async def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        qv = embed(query)
        scored = [
            (cosine(qv, vec), row) for vec, row in zip(self._vectors, self.corpus)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {**row, "similarity": round(float(score), 4), "retrieval_backend": self.name}
            for score, row in scored[:k]
        ]

    async def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        for row in self.corpus:
            if row.get("case_id") == case_id:
                return {**row, "retrieval_backend": self.name}
        return None

    async def close(self) -> None:
        return None


class PgVectorIndex:
    """pgvector-backed cosine search."""

    name = "pgvector"

    def __init__(self, pool: Any, corpus: List[Dict[str, Any]]) -> None:
        self._pool = pool
        self.corpus = corpus

    @classmethod
    async def create(cls, dsn: str, corpus: List[Dict[str, Any]]) -> "PgVectorIndex":
        import asyncpg

        pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4, timeout=5, command_timeout=10)
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS resolved_case_precedents (
                    case_id           TEXT PRIMARY KEY,
                    category          TEXT NOT NULL,
                    summary           TEXT NOT NULL,
                    signals           JSONB NOT NULL DEFAULT '[]'::jsonb,
                    resolution        TEXT NOT NULL,
                    resolution_notes  TEXT NOT NULL DEFAULT '',
                    resolved_at       TEXT NOT NULL DEFAULT '',
                    reviewer          TEXT NOT NULL DEFAULT '',
                    embedding         vector({EMBED_DIM}) NOT NULL
                )
                """
            )
            for row in corpus:
                await conn.execute(
                    """
                    INSERT INTO resolved_case_precedents
                        (case_id, category, summary, signals, resolution, resolution_notes, resolved_at, reviewer, embedding)
                    VALUES ($1,$2,$3,$4::jsonb,$5,$6,$7,$8,$9::vector)
                    ON CONFLICT (case_id) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        embedding = EXCLUDED.embedding
                    """,
                    row.get("case_id"),
                    row.get("category", ""),
                    row.get("summary", ""),
                    json.dumps(row.get("signals", [])),
                    row.get("resolution", ""),
                    row.get("resolution_notes", ""),
                    str(row.get("resolved_at", "")),
                    row.get("reviewer", ""),
                    _vector_literal(embed(precedent_text(row))),
                )
            # ivfflat needs rows present before it can be built usefully.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS resolved_case_precedents_embedding_idx "
                "ON resolved_case_precedents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 8)"
            )
        return cls(pool, corpus)

    async def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        literal = _vector_literal(embed(query))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT case_id, category, summary, signals, resolution, resolution_notes,
                       resolved_at, reviewer,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM resolved_case_precedents
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                literal,
                k,
            )
        return [
            {
                "case_id": r["case_id"],
                "category": r["category"],
                "summary": r["summary"],
                "signals": json.loads(r["signals"]) if isinstance(r["signals"], str) else r["signals"],
                "resolution": r["resolution"],
                "resolution_notes": r["resolution_notes"],
                "resolved_at": r["resolved_at"],
                "reviewer": r["reviewer"],
                "similarity": round(float(r["similarity"]), 4),
                "retrieval_backend": self.name,
            }
            for r in rows
        ]

    async def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT case_id, category, summary, signals, resolution, resolution_notes, "
                "resolved_at, reviewer FROM resolved_case_precedents WHERE case_id = $1",
                case_id,
            )
        if row is None:
            return None
        out = dict(row)
        if isinstance(out.get("signals"), str):
            out["signals"] = json.loads(out["signals"])
        out["retrieval_backend"] = self.name
        return out

    async def close(self) -> None:
        await self._pool.close()


def _vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


async def build_index(corpus: Optional[List[Dict[str, Any]]] = None) -> Any:
    """Return a pgvector index if one is reachable, else the local index."""
    rows = corpus if corpus is not None else load_corpus()
    dsn = SETTINGS.database_url
    if dsn:
        try:
            return await PgVectorIndex.create(dsn, rows)
        except Exception as exc:  # noqa: BLE001 - documented downgrade
            if SETTINGS.require_pgvector:
                raise RuntimeError(
                    f"SENTRYMESH_REQUIRE_PGVECTOR=1 but pgvector at {dsn} is unusable: {exc}"
                ) from exc
    elif SETTINGS.require_pgvector:
        raise RuntimeError("SENTRYMESH_REQUIRE_PGVECTOR=1 but DATABASE_URL is empty")
    return LocalCosineIndex(rows)
