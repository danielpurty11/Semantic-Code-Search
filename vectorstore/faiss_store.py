from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from chunking.models import CodeChunk
from vectorstore.models import SearchResult

INDEX_FILE  = "vectors/faiss.index"
CHUNKS_FILE = "vectors/faiss_chunks.pkl"


class FaissStore:
    def __init__(
        self,
        dimensions: int,
        index_path: str = INDEX_FILE,
        chunks_path: str = CHUNKS_FILE,
    ):
        self.dimensions  = dimensions
        self.index_path  = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[CodeChunk] = []

    # ------------------------------------------------------------------ #
    # Build / persist                                                      #
    # ------------------------------------------------------------------ #

    def add(self, chunks: list[CodeChunk], vectors: np.ndarray) -> None:
        """Add chunks and their L2-normalized vectors to the index."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimensions)

        self._index.add(vectors.astype("float32"))
        self._chunks.extend(chunks)

    def save(self) -> None:
        """Persist index and chunk metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.chunks_path.write_bytes(pickle.dumps(self._chunks))

    def load(self) -> None:
        """Load a previously saved index from disk."""
        self._index  = faiss.read_index(str(self.index_path))
        self._chunks = pickle.loads(self.chunks_path.read_bytes())

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Return top-K results ordered by cosine similarity (highest first)."""
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            results.append(SearchResult(
                chunk=self._chunks[idx],
                score=float(score),
                rank=rank,
            ))
        return results

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
