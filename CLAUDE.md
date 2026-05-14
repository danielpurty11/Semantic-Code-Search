# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/unit/test_chunking.py -v

# Run a single test
python -m pytest tests/unit/test_chunking.py::test_code_chunk_defaults -v

# Run the CLI (once cli/main.py is implemented)
python main.py
```

**Note:** `tree_sitter_languages` is not available in this environment. `tests/unit/test_parser.py` will fail with an ImportError unless the package is installed. The dispatcher uses a lazy import so all other modules work fine without it.

No linting or formatting tools are configured.

## Architecture

The system is a semantic code search engine. Data flows through this pipeline:

```
scan_repository()  →  parse_file()  →  file_to_chunks()  →  embed  →  vectorstore  →  rank  →  api/cli
   (scanner)           (parsers)        (chunking)        (stub)      (stub)        (stub)  (stub)
```

**Implemented modules:** `scanner`, `parsers`, `chunking`, incremental tracker  
**Stubs (empty files):** `embeddings`, `vectorstore`, `ranking`, `graph`, `api`, `cli`, `workers`, `scheduler`, `observability`

### Key dataclasses

| Dataclass | Module | Role |
|---|---|---|
| `FileRecord` | `scanner.repo_scanner` | File path, language, size, commit hash |
| `CodeSymbol` | `parsers.python_parser` | Symbol name/type, source, docstring, imports, decorators |
| `CodeChunk` | `chunking.models` | All of the above + `text` field sent to the embedding model |

### Module notes

- **`scanner/repo_scanner.py`** — walks a repo, respects `.gitignore`, detects language by extension, attaches git commit hash to each `FileRecord`
- **`scanner/incremental.py`** — `IncrementalTracker` uses SQLite to skip files whose mtime hasn't changed since last index
- **`parsers/dispatcher.py`** — routes to `parse_python_file()` or `parse_with_treesitter()` based on language; tree-sitter import is lazy to avoid hard failure when the package is absent
- **`chunking/symbol_chunker.py`** — one chunk per symbol; functions over 60 lines are split into overlapping windows (overlap=10); imports `CodeSymbol` directly from `parsers.python_parser`, not via the dispatcher
- **`chunking/sliding_window.py`** — fallback for files that fail parsing; produces `symbol_type="module"` chunks
- **`chunking/pipeline.py`** — `file_to_chunks(record, repo)` is the main entry point: parses then chunks, falls back to sliding window if no symbols
- **`config.py`** — Pydantic `BaseSettings`; copy `.env.example` to `.env` to set `REPO_PATH`, `EMBEDDING_MODEL`, etc.

## Sample repo

`sample_repo/` contains Python and JavaScript fixtures used by the test suite. Tests reference it via `Path(__file__).parent.parent.parent / "sample_repo"`.
