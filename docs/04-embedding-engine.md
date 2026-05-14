# Step 4: Embedding Engine

**Week 2 | Goal:** Convert `CodeChunk` objects into dense vector embeddings, with batching and caching.

---

## What This Component Does

Takes the text from a `CodeChunk` and produces a **vector** — a list of floating-point numbers
that encodes its semantic meaning. Similar code produces similar vectors.

```
CodeChunk.text
    │
    ▼
Embedding Model
    │
    ▼
[0.12, -0.94, 0.31, ..., 0.07]   ← 384 or 768 numbers
```

These vectors are stored in a vector database. At query time, the user's natural language
query goes through the same model, and the nearest vectors are returned as results.

---

## Concepts to Learn

| Concept | Description |
|---|---|
| Embedding | A fixed-size vector that encodes semantic meaning |
| Cosine similarity | How closeness between vectors is measured |
| Batch inference | Processing many texts in one forward pass for speed |
| Embedding cache | Avoid recomputing vectors for unchanged chunks |
| sentence-transformers | The library used to load embedding models |
| Model dimensions | Different models output different vector sizes (384, 768, 1536) |

---

## Choosing a Model

| Model | Dims | Speed | Quality | Best For |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Development, small repos |
| `all-mpnet-base-v2` | 768 | Medium | Better | Production MVP |
| `microsoft/codebert-base` | 768 | Medium | Best for code | Code-specific search |
| `text-embedding-3-small` (OpenAI) | 1536 | API call | Excellent | When cost is acceptable |

**Default choice for this project:** `all-MiniLM-L6-v2`
- No API key needed
- Runs locally
- Fast enough for a laptop
- Good enough for semantic search

---

## Part A: The Encoder

```python
# embeddings/encoder.py
from sentence_transformers import SentenceTransformer
from chunking.models import CodeChunk
import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


class CodeEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed_chunks(self, chunks: list[CodeChunk]) -> np.ndarray:
        """
        Embed a list of chunks. Returns an array of shape (N, dimensions).
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed raw texts in batches.
        Returns shape (N, dimensions).
        """
        return self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize for cosine similarity
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query. Returns shape (dimensions,).
        """
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
```

### Why `normalize_embeddings=True`?

After normalization, all vectors have length 1.
This means **dot product == cosine similarity** — cheaper and equivalent.

Without normalization you would need to compute:

```
cosine_similarity = dot(a, b) / (|a| * |b|)
```

With normalization, since `|a| == |b| == 1`:

```
cosine_similarity = dot(a, b)
```

Most vector databases (FAISS, ChromaDB) assume normalized vectors for inner-product search.

---

## Part B: Embedding Cache

Re-embedding unchanged code is wasteful.
If a chunk's `commit_hash` is the same as last time, its embedding is still valid.

```python
# embeddings/cache.py
import json
import hashlib
from pathlib import Path
import numpy as np

CACHE_DIR = Path(".embedding_cache")


def _cache_key(chunk_id: str, model_name: str) -> str:
    return hashlib.md5(f"{chunk_id}:{model_name}".encode()).hexdigest()


def load_cached(chunk_id: str, model_name: str) -> np.ndarray | None:
    key = _cache_key(chunk_id, model_name)
    path = CACHE_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_cached(chunk_id: str, model_name: str, vector: np.ndarray) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(chunk_id, model_name)
    np.save(CACHE_DIR / f"{key}.npy", vector)


def embed_with_cache(
    chunks: list,
    embedder,
) -> list[tuple]:
    """
    Embed chunks, using cache where available.
    Returns list of (chunk, vector) pairs.
    """
    results = []
    to_embed = []

    for chunk in chunks:
        cached = load_cached(chunk.id, embedder.model_name)
        if cached is not None:
            results.append((chunk, cached))
        else:
            to_embed.append(chunk)

    if to_embed:
        vectors = embedder.embed_chunks(to_embed)
        for chunk, vector in zip(to_embed, vectors):
            save_cached(chunk.id, embedder.model_name, vector)
            results.append((chunk, vector))

    return results
```

---

## Part C: Full Indexing Pipeline

```python
# embeddings/pipeline.py
from chunking.models import CodeChunk
from embeddings.encoder import CodeEmbedder
from embeddings.cache import embed_with_cache
import numpy as np


def embed_all_chunks(
    chunks: list[CodeChunk],
    embedder: CodeEmbedder,
    use_cache: bool = True,
) -> tuple[list[CodeChunk], np.ndarray]:
    """
    Embed all chunks.
    Returns (chunks, matrix) where matrix[i] is the vector for chunks[i].
    """
    if use_cache:
        pairs = embed_with_cache(chunks, embedder)
    else:
        vectors = embedder.embed_chunks(chunks)
        pairs = list(zip(chunks, vectors))

    ordered_chunks = [pair[0] for pair in pairs]
    matrix = np.stack([pair[1] for pair in pairs])

    return ordered_chunks, matrix
```

---

## What the Output Looks Like

```python
from embeddings.encoder import CodeEmbedder
from parsers.python_parser import parse_python_file
from chunking.symbol_chunker import symbol_to_chunks

embedder = CodeEmbedder()
symbols = parse_python_file("sample_repo/services/auth.py")

for sym in symbols:
    chunks = symbol_to_chunks(sym, "sample_repo")
    vectors = embedder.embed_chunks(chunks)
    for chunk, vec in zip(chunks, vectors):
        print(f"{chunk.symbol}: shape={vec.shape}, norm={vec @ vec:.4f}")
```

Expected output:
```
validate_jwt_token: shape=(384,), norm=1.0000
create_jwt_token:   shape=(384,), norm=1.0000
```

Every vector has exactly 384 numbers and is L2-normalized (norm = 1.0).

---

## How Cosine Similarity Works

Two vectors are "close" if they point in the same direction in 384-dimensional space.

```
query:  "validate JWT token"        → vector q
chunk:  validate_jwt_token source   → vector c

similarity = dot(q, c)   (since both are normalized)
           = cos(angle between them)
           = 1.0  → identical meaning
           = 0.0  → completely unrelated
           = -1.0 → opposite meaning (rare in practice)
```

Typical similarity scores:
- **> 0.85** — Strong match (likely the right function)
- **0.70–0.85** — Related code
- **< 0.70** — Weak or incidental match

---

## Install

```bash
pip install sentence-transformers numpy
```

The first run downloads the model weights (~90MB for MiniLM).
They are cached in `~/.cache/torch/sentence_transformers/`.

---

## Exercises

1. Embed `validate_jwt_token` and `create_jwt_token`. Print `dot(v1, v2)`. Are they similar?
2. Embed the query `"check if a token is expired"`. Which function is closer?
3. Try `all-mpnet-base-v2` (768 dims) — does search quality improve?
4. Time how long it takes to embed 100 chunks vs 1000 chunks. Does batching help?
5. Delete `.embedding_cache/` and re-run. Then run again. Measure the time difference.

---

## What You Learned

- How embedding models convert text into vectors
- Why L2-normalization enables cosine similarity via dot product
- Batch inference for throughput
- File-based caching to skip unchanged chunks
- The shape and scale of real embedding vectors

---

## Next Step

[05-vector-store.md](05-vector-store.md) — Store vectors in FAISS and ChromaDB, then search them.
