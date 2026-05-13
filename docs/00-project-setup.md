# Step 0: Project Setup

## What You'll Build

A CLI tool that lets you search a codebase using natural language:

```bash
codesearch "where is jwt token validation implemented?"
```

---

## Folder Structure

```
semantic-code-search/
├── api/              # FastAPI REST endpoints
├── cli/              # Typer CLI commands
├── parsers/          # Tree-sitter + AST parsing
├── embeddings/       # Model inference
├── vectorstore/      # FAISS / Qdrant wrapper
├── chunking/         # Code chunking logic
├── graph/            # Dependency graph
├── ranking/          # Result reranking
├── scheduler/        # Background indexing jobs
├── workers/          # Distributed workers (Phase 2)
├── observability/    # Metrics, tracing
├── tests/
├── deployments/      # Docker, k8s
└── docs/
```

---

## Environment Setup

### 1. Python Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install Core Dependencies

```bash
pip install \
  fastapi uvicorn \
  typer rich \
  tree-sitter tree-sitter-languages \
  sentence-transformers \
  faiss-cpu \
  chromadb \
  gitpython \
  watchdog \
  sqlalchemy \
  aiosqlite \
  asyncio aiofiles \
  pydantic
```

### 3. Project Init Files

```bash
touch semantic_code_search/__init__.py
touch semantic_code_search/config.py
```

### 4. config.py

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    repo_path: str = "./repo"
    db_path: str = "./index.db"
    vector_dir: str = "./vectors"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 50          # lines per chunk
    top_k: int = 10               # results per query

settings = Settings()
```

---

## Key Concepts to Learn First

| Concept | Why It Matters |
|---|---|
| Vector embeddings | Core of semantic search |
| Cosine similarity | How "closeness" is measured |
| AST (Abstract Syntax Tree) | How parsers understand code |
| FAISS / vector DB | Storing and retrieving embeddings |
| Tree-sitter | Universal code parser |

---

## Learning Resources

- [Tree-sitter docs](https://tree-sitter.github.io/tree-sitter/)
- [sentence-transformers docs](https://www.sbert.net/)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
- [FastAPI docs](https://fastapi.tiangolo.com/)
- [Typer docs](https://typer.tiangolo.com/)

---

## Next Step

[01-repo-scanner.md](01-repo-scanner.md) — Scan a Git repo and discover its files.
