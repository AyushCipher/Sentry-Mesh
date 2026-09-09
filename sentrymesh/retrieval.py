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

EMBEDDINGS ARE A DOCUMENTED MOCK.  ``embed()`` is hashed TF-IDF - a
deterministic lexical projection, not a learned text embedding.  IDF weights are
derived from the precedent corpus itself, which is what stops ubiquitous tokens
("account", "transaction", "usd") from dominating the cosine and reducing
retrieval to noise.  The first implementation here was unweighted hashed
bag-of-words and it was measurably useless: it returned high fraud-precedent
similarity for spotlessly clean cases, which would have shown up in the eval as
a false-positive problem in the *supervisor* when the fault was in retrieval.

What it still cannot do: match two cases that describe the same fraud pattern in
different vocabulary.  Lexical overlap is the whole mechanism.  Swapping in a
real embedding model is a change to this one function; the honest consequence of
not having done so is stated in the README's Known Limitations and interrogated
in the harness report.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentrymesh.config import PRECEDENT_CORPUS, SETTINGS

EMBED_DIM = 256
_TOKEN = re.compile(r"[a-z0-9_]+")

# Tokens carried by every case description regardless of risk. Kept explicit
# rather than folded into the IDF cutoff so a reader can see what was dropped.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "of", "on", "in", "to", "for", "with", "at", "by", "from",
        "is", "was", "were", "be", "been", "it", "its", "that", "this", "as", "but", "or",
        "not", "no", "than", "then", "over", "under", "into", "out", "up", "usd", "amount",
        "account", "accounts", "transaction", "transactions", "case", "cases", "days", "day",
    }
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1]


def build_idf(documents: List[str]) -> Dict[str, float]:
    """Smoothed inverse document frequency over the precedent corpus."""
    n = max(len(documents), 1)
    df: Counter = Counter()
    for doc in documents:
        df.update(set(tokenize(doc)))
    return {token: math.log((n + 1) / (count + 1)) + 1.0 for token, count in df.items()}


def embed(text: str, idf: Optional[Dict[str, float]] = None) -> List[float]:
    """Deterministic hashed TF-IDF vector (MOCK - see module docstring)."""
    vec = [0.0] * EMBED_DIM
    counts = Counter(tokenize(text))
    if not counts:
        return vec
    # Unseen tokens get the weight of a token appearing in exactly one document,
    # so a query term absent from the corpus is treated as maximally specific
    # rather than silently discarded.
    default_idf = max(idf.values()) if idf else 1.0
    for token, tf in counts.items():
        weight = (1.0 + math.log(tf)) * ((idf or {}).get(token, default_idf))
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % EMBED_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign * weight
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
        self.idf = build_idf([precedent_text(row) for row in corpus])
        self._vectors = [embed(precedent_text(row), self.idf) for row in corpus]

    async def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        qv = embed(query, self.idf)
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

    def __init__(self, pool: Any, corpus: List[Dict[str, Any]], idf: Dict[str, float]) -> None:
        self._pool = pool
        self.corpus = corpus
        self.idf = idf

    @classmethod
    async def create(cls, dsn: str, corpus: List[Dict[str, Any]]) -> "PgVectorIndex":
        import asyncpg

        idf = build_idf([precedent_text(row) for row in corpus])
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
                    _vector_literal(embed(precedent_text(row), idf)),
                )
            # ivfflat needs rows present before it can be built usefully.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS resolved_case_precedents_embedding_idx "
                "ON resolved_case_precedents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 8)"
            )
        return cls(pool, corpus, idf)

    async def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        literal = _vector_literal(embed(query, self.idf))
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

class DenseSemanticIndex:
    """Dense semantic vector search using sentence-transformers when available."""

    name = "dense_semantic"

    def __init__(self, corpus: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> None:
        self.corpus = corpus
        self.model_name = model_name
        self._model = None
        self._local: Optional[LocalCosineIndex] = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            texts = [precedent_text(row) for row in corpus]
            self._vectors = self._model.encode(texts, normalize_embeddings=True).tolist()
        except Exception:
            self._local = LocalCosineIndex(corpus)

    async def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self._model is not None:
            qv = self._model.encode([query], normalize_embeddings=True)[0].tolist()
            scored = [
                (cosine(qv, vec), row) for vec, row in zip(self._vectors, self.corpus)
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [
                {**row, "similarity": round(float(score), 4), "retrieval_backend": self.name}
                for score, row in scored[:k]
            ]
        return await self._local.search(query, k) if self._local else []

    async def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        for row in self.corpus:
            if row.get("case_id") == case_id:
                return {**row, "retrieval_backend": self.name}
        return None

    async def close(self) -> None:
        return None


def _vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


async def build_index(corpus: Optional[List[Dict[str, Any]]] = None) -> Any:
    """Return a pgvector index if one is reachable, else dense semantic or local index."""
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

    if SETTINGS.dense_embeddings:
        try:
            return DenseSemanticIndex(rows)
        except Exception:
            pass
    return LocalCosineIndex(rows)

