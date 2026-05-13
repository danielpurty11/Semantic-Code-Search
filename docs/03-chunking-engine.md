# Step 3: Chunking Engine

**Week 2 | Goal:** Convert parsed symbols into embedding-ready text chunks with rich metadata.

---

## Why Chunking Matters

Embedding models have a token limit (~512 tokens for most models).
Large functions must be split. Small functions can be merged.
Each chunk needs enough context to be independently searchable.

---

## Concepts to Learn

| Concept | Description |
|---|---|
| Token limits | Most embedding models: 256–512 tokens |
| Context window | More context = better embeddings, but costlier |
| Overlapping chunks | Avoid cutting meaning at boundaries |
| Chunk metadata | Every chunk must carry enough info to reconstruct its origin |

---

## Chunk Schema

```python
# chunking/models.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CodeChunk:
    id: str                          # unique ID: repo:path:symbol:chunk_n
    repo: str
    language: str
    file_path: str
    symbol: str                      # function/class name
    symbol_type: str                 # "function" | "class" | "module"
    start_line: int
    end_line: int
    text: str                        # the text to be embedded
    docstring: Optional[str] = None
    imports: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    commit_hash: str = ""
```

---

## Chunking Strategy

### Strategy 1: Symbol-level chunks (preferred)

One chunk per function/class. Best for semantic accuracy.

```python
# chunking/symbol_chunker.py
from parsers.dispatcher import CodeSymbol
from chunking.models import CodeChunk
import hashlib

MAX_LINES = 60   # functions longer than this get split


def symbol_to_chunks(
    symbol: CodeSymbol,
    repo: str,
    commit_hash: str = "",
) -> list[CodeChunk]:
    lines = symbol.source.splitlines()

    if len(lines) <= MAX_LINES:
        return [_make_chunk(symbol, repo, commit_hash, lines, 0)]

    # Split into overlapping windows
    chunks = []
    window = MAX_LINES
    overlap = 10
    step = window - overlap

    for i, start in enumerate(range(0, len(lines), step)):
        chunk_lines = lines[start : start + window]
        if not chunk_lines:
            break
        chunks.append(_make_chunk(symbol, repo, commit_hash, chunk_lines, i, part=i))

    return chunks


def _make_chunk(
    symbol: CodeSymbol,
    repo: str,
    commit_hash: str,
    lines: list[str],
    part: int,
    part: int = 0,
) -> CodeChunk:
    text = _build_chunk_text(symbol, lines)
    chunk_id = hashlib.md5(
        f"{repo}:{symbol.file_path}:{symbol.symbol}:{part}".encode()
    ).hexdigest()[:16]

    return CodeChunk(
        id=chunk_id,
        repo=repo,
        language=symbol.language,
        file_path=symbol.file_path,
        symbol=symbol.symbol,
        symbol_type=symbol.type,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        text=text,
        docstring=symbol.docstring,
        imports=symbol.imports,
        decorators=symbol.decorators,
        commit_hash=commit_hash,
    )


def _build_chunk_text(symbol: CodeSymbol, lines: list[str]) -> str:
    """
    Build rich text for embedding.
    Include context clues so the model understands what this is.
    """
    parts = []
    if symbol.docstring:
        parts.append(f"# {symbol.docstring}")
    if symbol.imports:
        parts.append(f"# imports: {', '.join(symbol.imports[:5])}")
    parts.append(f"# file: {symbol.file_path}")
    parts.append("\n".join(lines))
    return "\n".join(parts)
```

---

### Strategy 2: Sliding Window (fallback for unparsed files)

For files that fail parsing (config files, unusual syntax):

```python
# chunking/sliding_window.py
from chunking.models import CodeChunk
import hashlib

def chunk_raw_file(
    file_path: str,
    repo: str,
    language: str,
    window: int = 40,
    overlap: int = 10,
) -> list[CodeChunk]:
    lines = open(file_path).readlines()
    step = window - overlap
    chunks = []

    for i, start in enumerate(range(0, len(lines), step)):
        chunk_lines = lines[start : start + window]
        if not chunk_lines:
            break
        text = "".join(chunk_lines)
        chunk_id = hashlib.md5(f"{file_path}:{i}".encode()).hexdigest()[:16]

        chunks.append(CodeChunk(
            id=chunk_id,
            repo=repo,
            language=language,
            file_path=file_path,
            symbol=f"raw_chunk_{i}",
            symbol_type="module",
            start_line=start + 1,
            end_line=start + len(chunk_lines),
            text=text,
        ))

    return chunks
```

---

## Main Pipeline: File → Chunks

```python
# chunking/pipeline.py
from scanner.repo_scanner import FileRecord
from parsers.dispatcher import parse_file
from chunking.symbol_chunker import symbol_to_chunks
from chunking.sliding_window import chunk_raw_file
from chunking.models import CodeChunk

def file_to_chunks(record: FileRecord, repo: str) -> list[CodeChunk]:
    symbols = parse_file(record.path, record.language)

    if symbols:
        chunks = []
        for symbol in symbols:
            chunks.extend(symbol_to_chunks(symbol, repo, record.last_commit))
        return chunks
    else:
        # fallback: sliding window
        return chunk_raw_file(
            record.path, repo, record.language
        )
```

---

## How Good Chunk Text Looks

The text passed to the embedding model should be self-contained:

```
# Validates a Bearer JWT token from the request header.
# imports: jwt, datetime, fastapi
# file: auth/middleware.py

def validate_jwt_token(token: str) -> dict:
    """Validates a Bearer JWT token from the request header."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
```

This makes the embedding capture:
- **what** the function does (docstring)
- **what** it uses (imports)
- **where** it lives (file path)
- **how** it works (code)

---

## Exercises

1. Print chunk count vs symbol count for a real repo
2. Add a `token_count` field using `tiktoken` to measure chunk size
3. Try embedding the same function with/without docstring — compare similarity scores

---

## What You Learned

- Why chunking is necessary for embedding
- Symbol-level vs sliding-window strategies
- How to craft embedding-friendly text
- Overlapping windows to preserve context at boundaries

---

## Next Step

[04-embedding-engine.md](04-embedding-engine.md) — Convert chunks into vectors.
