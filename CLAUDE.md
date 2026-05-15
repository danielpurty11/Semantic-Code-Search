# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests (no venv needed — system Python has all required packages)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/unit/test_chunking.py -v

# Run a single test
python -m pytest tests/unit/test_chunking.py::test_code_chunk_defaults -v

# Run the CLI (once cli/main.py is implemented)
python main.py
```

**Note:** `.venv/` exists but is broken on this platform (missing C extensions). Run tests directly with the system Python. `tree_sitter_languages`, `numpy`, `sentence-transformers`, `faiss`, and `chromadb` are not installable here — all are stubbed out via `sys.modules` injection in `tests/unit/conftest.py` and the individual test files, so the full suite runs without them.

No linting or formatting tools are configured.

## Architecture

The system is a semantic code search engine. Data flows through this pipeline:

```
scan_repository()  →  parse_file()  →  file_to_chunks()  →  embed_all_chunks()  →  FaissStore/ChromaStore  →  rank  →  api/cli
   (scanner)           (parsers)        (chunking)           (embeddings)            (vectorstore)           (stub)  (stub)
```

**Implemented modules:** `scanner`, `parsers`, `chunking`, `embeddings`, `vectorstore`  
**Stubs (empty files):** `ranking`, `graph`, `api`, `cli`, `workers`, `scheduler`, `observability`

### Key dataclasses

| Dataclass | Module | Role |
|---|---|---|
| `FileRecord` | `scanner.repo_scanner` | File path, language, size, commit hash |
| `CodeSymbol` | `parsers.python_parser` | Symbol name/type, source, docstring, imports, decorators |
| `CodeChunk` | `chunking.models` | All of the above + `text` field sent to the embedding model |
| `SearchResult` | `vectorstore.models` | Chunk + cosine similarity score + 1-based rank |

### Module notes

- **`scanner/repo_scanner.py`** — walks a repo, respects `.gitignore`, detects language by extension, attaches git commit hash to each `FileRecord`
- **`scanner/incremental.py`** — `IncrementalTracker` uses SQLite to skip files whose mtime hasn't changed since last index
- **`parsers/dispatcher.py`** — routes to `parse_python_file()` or `parse_with_treesitter()` based on language; tree-sitter import is lazy to avoid hard failure when the package is absent
- **`chunking/symbol_chunker.py`** — one chunk per symbol; functions over 60 lines are split into overlapping windows (overlap=10); imports `CodeSymbol` directly from `parsers.python_parser`, not via the dispatcher
- **`chunking/sliding_window.py`** — fallback for files that fail parsing; produces `symbol_type="module"` chunks
- **`chunking/pipeline.py`** — `file_to_chunks(record, repo)` is the main entry point: parses then chunks, falls back to sliding window if no symbols
- **`embeddings/encoder.py`** — `CodeEmbedder` wraps `SentenceTransformer`; `embed_chunks()` / `embed_query()` with L2 normalization (dot product == cosine similarity)
- **`embeddings/cache.py`** — file-based `.npy` cache keyed by `md5(chunk_id:model_name)`; `embed_with_cache()` skips chunks whose vectors are already stored
- **`embeddings/pipeline.py`** — `embed_all_chunks(chunks, embedder)` returns `(chunks, matrix)` ready for the vector store
- **`vectorstore/faiss_store.py`** — `FaissStore`: in-process `IndexFlatIP`, manual `save()`/`load()` via pickle; no metadata filtering
- **`vectorstore/chroma_store.py`** — `ChromaStore`: persistent ChromaDB, auto-saved, supports `where={"language": "python"}` filters; converts cosine distance → similarity with `score = 1 - dist`
- **`config.py`** — Pydantic `BaseSettings`; copy `.env.example` to `.env` to set `REPO_PATH`, `EMBEDDING_MODEL`, etc.

### Test stubbing pattern

Packages unavailable in this environment are stubbed in two places:
- **`tests/unit/conftest.py`** — installs a `tree_sitter_languages` stub with a regex-based JS parser that produces real symbol names and line numbers from `sample_repo/frontend/app.js`
- **Individual test files** (`test_embeddings.py`, `test_vectorstore.py`) — inject `numpy`, `sentence_transformers`, `faiss`, `chromadb` stubs via `sys.modules` at the top of each file before any module import

When `tree_sitter_languages` is genuinely installed, `conftest.py` detects this and skips installing the stub.

## Sample repo

`sample_repo/` contains Python and JavaScript fixtures used by the test suite. Tests reference it via `Path(__file__).parent.parent.parent / "sample_repo"`.
