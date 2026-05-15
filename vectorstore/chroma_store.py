from __future__ import annotations

import chromadb
import numpy as np

from chunking.models import CodeChunk
from vectorstore.models import SearchResult


class ChromaStore:
    def __init__(
        self,
        persist_dir: str = "vectors/chroma",
        collection: str = "code",
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col    = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Build / persist                                                      #
    # ------------------------------------------------------------------ #

    def add(self, chunks: list[CodeChunk], vectors: np.ndarray) -> None:
        """Add chunks and their vectors. Persistence is automatic."""
        self._col.add(
            ids        = [c.id for c in chunks],
            embeddings = vectors.tolist(),
            documents  = [c.text for c in chunks],
            metadatas  = [_chunk_metadata(c) for c in chunks],
        )

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        where: dict | None = None,   # e.g. {"language": "python"}
    ) -> list[SearchResult]:
        """Return top-K results, optionally filtered by metadata."""
        kwargs: dict = dict(
            query_embeddings = [query_vector.tolist()],
            n_results        = top_k,
            include          = ["distances", "metadatas", "documents"],
        )
        if where:
            kwargs["where"] = where

        res   = self._col.query(**kwargs)
        ids   = res["ids"][0]
        dists = res["distances"][0]
        metas = res["metadatas"][0]
        docs  = res["documents"][0]

        results = []
        for rank, (chunk_id, dist, meta, doc) in enumerate(
            zip(ids, dists, metas, docs), start=1
        ):
            score = 1.0 - dist   # chroma returns cosine distance; convert to similarity
            chunk = _metadata_to_chunk(chunk_id, meta, doc)
            results.append(SearchResult(chunk=chunk, score=score, rank=rank))
        return results

    @property
    def size(self) -> int:
        return self._col.count()


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _chunk_metadata(c: CodeChunk) -> dict:
    return {
        "repo":        c.repo,
        "language":    c.language,
        "file_path":   c.file_path,
        "symbol":      c.symbol,
        "symbol_type": c.symbol_type,
        "start_line":  c.start_line,
        "end_line":    c.end_line,
        "commit_hash": c.commit_hash,
        "docstring":   c.docstring or "",
    }


def _metadata_to_chunk(chunk_id: str, meta: dict, text: str) -> CodeChunk:
    return CodeChunk(
        id          = chunk_id,
        repo        = meta["repo"],
        language    = meta["language"],
        file_path   = meta["file_path"],
        symbol      = meta["symbol"],
        symbol_type = meta["symbol_type"],
        start_line  = meta["start_line"],
        end_line    = meta["end_line"],
        text        = text,
        docstring   = meta.get("docstring") or None,
        commit_hash = meta.get("commit_hash", ""),
    )
