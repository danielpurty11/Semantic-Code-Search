# Step 5: Vector Store

**Week 3 | Goal:** Store embedding vectors and search them by semantic similarity.

---

## What This Component Does

Takes the `(chunk, vector)` pairs from the embedding engine and:
1. **Stores** them in an index
2. **Searches** the index given a query vector — returns the top-K most similar chunks

```
embed_all_chunks()
    → list of (CodeChunk, vector)
    → VectorStore.add(chunks, vectors)

user query: "validate JWT token"
    → embedder.embed_query(query)   → query_vector
    → VectorStore.search(query_vector, top_k=10)
    → list[SearchResult]
```

---

## Concepts to Learn

| Concept | Description |
|---|---|
| ANN (Approximate Nearest Neighbor) | Find the K closest vectors without checking every one |
| FAISS | Facebook's fast in-process vector search library |
| ChromaDB | Embedded vector database with metadata filtering |
| Inner product search | Fast similarity with L2-normalized vectors (== cosine similarity) |
| Flat index | Exact brute-force search — simple and correct for small corpora |
| IVF index | Inverted file index — faster for large corpora, slight accuracy trade-off |

---

## The SearchResult Schema

```python
# vectorstore/models.py
from dataclasses import dataclass
from chunking.models import CodeChunk

@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float      # cosine similarity: 1.0 = identical, 0.0 = unrelated
    rank: int         # 1-based position in result list
```

---

## Option A: FAISS Store

FAISS is an in-process library — no server, no network, just a file on disk.
Best for: local development, single-machine indexing, fast iteration.

```python
# vectorstore/faiss_store.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from chunking.models import CodeChunk
from vectorstore.models import SearchResult

INDEX_FILE  = "vectors/faiss.index"
CHUNKS_FILE = "vectors/faiss_chunks.pkl"


class FaissStore:
    def __init__(self, dimensions: int, index_path: str = INDEX_FILE,
                 chunks_path: str = CHUNKS_FILE):
        self.dimensions  = dimensions
        self.index_path  = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[CodeChunk] = []

    # ------------------------------------------------------------------ #
    # Build / persist                                                      #
    # ------------------------------------------------------------------ #

    def add(self, chunks: list[CodeChunk], vectors: np.ndarray) -> None:
        """Add chunks and their vectors to the index."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimensions)  # inner product

        self._index.add(vectors.astype("float32"))
        self._chunks.extend(chunks)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.chunks_path.write_bytes(pickle.dumps(self._chunks))

    def load(self) -> None:
        self._index  = faiss.read_index(str(self.index_path))
        self._chunks = pickle.loads(self.chunks_path.read_bytes())

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Return top-K results for a query vector."""
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
```

---

## Option B: ChromaDB Store

ChromaDB is an embedded vector database with built-in persistence and metadata filtering.
Best for: filtering by language/repo/symbol_type without a separate ranking step.

```python
# vectorstore/chroma_store.py
import chromadb
import numpy as np
from chunking.models import CodeChunk
from vectorstore.models import SearchResult


class ChromaStore:
    def __init__(self, persist_dir: str = "vectors/chroma", collection: str = "code"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col    = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Build / persist                                                      #
    # ------------------------------------------------------------------ #

    def add(self, chunks: list[CodeChunk], vectors: np.ndarray) -> None:
        self._col.add(
            ids         = [c.id for c in chunks],
            embeddings  = vectors.tolist(),
            documents   = [c.text for c in chunks],
            metadatas   = [_chunk_metadata(c) for c in chunks],
        )

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        where: dict | None = None,    # e.g. {"language": "python"}
    ) -> list[SearchResult]:
        kwargs = dict(
            query_embeddings = [query_vector.tolist()],
            n_results        = top_k,
            include          = ["distances", "metadatas", "documents"],
        )
        if where:
            kwargs["where"] = where

        res  = self._col.query(**kwargs)
        ids  = res["ids"][0]
        dists = res["distances"][0]
        metas = res["metadatas"][0]
        docs  = res["documents"][0]

        results = []
        for rank, (chunk_id, dist, meta, doc) in enumerate(
            zip(ids, dists, metas, docs), start=1
        ):
            score = 1.0 - dist   # chroma returns cosine distance
            chunk = _metadata_to_chunk(chunk_id, meta, doc)
            results.append(SearchResult(chunk=chunk, score=score, rank=rank))
        return results

    @property
    def size(self) -> int:
        return self._col.count()


# ------------------------------------------------------------------ #
# Helpers                                                              #
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
    from chunking.models import CodeChunk
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
```

---

## End-to-End Search Example

```python
from scanner.repo_scanner import scan_repository
from chunking.pipeline import file_to_chunks
from embeddings.encoder import CodeEmbedder
from embeddings.pipeline import embed_all_chunks
from vectorstore.faiss_store import FaissStore

# 1. Scan + chunk
repo = "sample_repo"
all_chunks = []
for record in scan_repository(repo):
    all_chunks.extend(file_to_chunks(record, repo))

# 2. Embed
embedder = CodeEmbedder()
chunks, matrix = embed_all_chunks(all_chunks, embedder)

# 3. Store
store = FaissStore(dimensions=embedder.dimensions)
store.add(chunks, matrix)
store.save()

# 4. Search
query_vec = embedder.embed_query("validate JWT token")
results = store.search(query_vec, top_k=5)

for r in results:
    print(f"[{r.rank}] {r.chunk.symbol}  score={r.score:.3f}  {r.chunk.file_path}")
```

Expected output:
```
[1] validate_jwt_token  score=0.921  sample_repo/services/auth.py
[2] create_jwt_token    score=0.743  sample_repo/services/auth.py
[3] login               score=0.612  sample_repo/frontend/app.js
```

---

## FAISS vs ChromaDB

| | FAISS | ChromaDB |
|---|---|---|
| Setup | In-process, no server | In-process, file-based |
| Persistence | Manual (`save()`/`load()`) | Automatic |
| Metadata filtering | No (filter after search) | Yes (`where={"language": "python"}`) |
| Speed | Fastest | Fast |
| Best for | Benchmarking, offline indexing | Production MVP |

---

## Exercises

1. Index `sample_repo/` and search for `"payment retry logic"` — which function ranks first?
2. Add a `delete(chunk_id)` method to `FaissStore` (hint: FAISS doesn't support deletion natively — think about how you'd rebuild the index)
3. Use ChromaDB's `where` filter to restrict search to `"python"` only — compare results to unfiltered
4. Benchmark: how long does `search()` take for 100 chunks vs 10,000?

---

## What You Learned

- How ANN search works and why it's needed at scale
- FAISS `IndexFlatIP` for exact inner-product search on normalized vectors
- ChromaDB for persistent, filterable vector storage
- The full pipeline from raw file → searchable vector index

---

## Next Step

[06-ranking-engine.md](06-ranking-engine.md) — Re-rank results using signals beyond vector similarity.
