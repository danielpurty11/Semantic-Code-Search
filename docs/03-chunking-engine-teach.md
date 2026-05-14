# Step 3: Chunking Engine — Teaching Guide

This document walks through every concept in the chunking engine from first principles.
Read it top to bottom. Each section has an explanation, a worked example, and a question to test yourself.

---

## 1. Why Chunking Exists

### The Problem: Embedding Models Have a Token Limit

An embedding model converts text into a vector (a list of numbers).
That vector represents the *meaning* of the text.

But embedding models can only process a fixed number of tokens at once — typically **256–512 tokens**.
One token ≈ 4 characters, so 512 tokens ≈ ~2000 characters, roughly 40–60 lines of code.

A real codebase has functions that are much longer than that.

**What happens if you exceed the limit?**
The model silently truncates the input. The vector only represents the first N tokens.
The rest of the function is invisible to search.

### The Solution: Break Code Into Chunks

A *chunk* is a slice of code small enough to fit in the model's context window,
with enough surrounding context to be independently searchable.

```
file_to_chunks(file)
    → parse symbols (functions, classes)
    → for each symbol → 1 or more chunks
    → each chunk: text + metadata
```

**Key insight:** A chunk is not just raw code. It carries metadata that makes it
independently interpretable even without the rest of the file.

### Self-check
> Why can't you just embed an entire file as one unit?

---

## 2. The CodeChunk Schema

Every chunk is a `CodeChunk` dataclass defined in [chunking/models.py](../chunking/models.py).

```python
@dataclass
class CodeChunk:
    id: str           # unique ID: md5(repo:path:symbol:part)[:16]
    repo: str         # which repository this came from
    language: str     # "python", "javascript", etc.
    file_path: str    # absolute path on disk
    symbol: str       # function/class name, or "raw_chunk_N" for fallback
    symbol_type: str  # "function" | "class" | "module"
    start_line: int   # where in the file this chunk starts
    end_line: int     # where it ends
    text: str         # THE TEXT SENT TO THE EMBEDDING MODEL
    docstring: Optional[str] = None
    imports: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    commit_hash: str = ""
```

### Why each field matters

| Field | Why it's there |
|---|---|
| `id` | Deduplication — same symbol re-indexed should produce same ID |
| `repo` | Multi-repo systems need to know the source |
| `language` | Affects how results are ranked and displayed |
| `file_path` | "Jump to file" in the UI |
| `symbol` | "Jump to function" — also used as chunk identity |
| `symbol_type` | Lets you filter "show me only classes" |
| `start_line` / `end_line` | Deep-link to exact lines in an IDE |
| `text` | What the embedding model actually reads |
| `docstring` | Stored separately so you can display it without re-parsing |
| `imports` | Context for the embedding; also useful for dependency graphs |
| `commit_hash` | Invalidate stale chunks when the file changes |

### Self-check
> If you re-index the same function on a different day, should the `id` change?
> What if only the `commit_hash` changed?

---

## 3. Strategy 1 — Symbol-Level Chunking (Preferred)

Source: [chunking/symbol_chunker.py](../chunking/symbol_chunker.py)

### The Idea

One function or class = one chunk (unless it's too long).
This is the preferred strategy because a function is already a *semantic unit* —
it has a clear purpose that can be expressed as a query like "validate JWT token".

### The Code, Line by Line

```python
MAX_LINES = 60  # functions longer than this get split

def symbol_to_chunks(symbol, repo, commit_hash="") -> list[CodeChunk]:
    lines = symbol.source.splitlines()

    if len(lines) <= MAX_LINES:
        # Short function: one chunk, done
        return [_make_chunk(symbol, repo, commit_hash, lines, part=0)]

    # Long function: split into overlapping windows
    window  = MAX_LINES   # each chunk is at most 60 lines
    overlap = 10          # consecutive chunks share 10 lines
    step    = window - overlap  # = 50 — how far to advance each iteration

    chunks = []
    for i, start in enumerate(range(0, len(lines), step)):
        chunk_lines = lines[start : start + window]
        if not chunk_lines:
            break
        chunks.append(_make_chunk(symbol, repo, commit_hash, chunk_lines, part=i))

    return chunks
```

### Understanding the Overlapping Window

Imagine a 130-line function with `window=60, overlap=10, step=50`:

```
Lines:    1 ──────────────────────────── 130
Chunk 0:  [1   ──────── 60]
Chunk 1:             [51 ──────── 110]
Chunk 2:                       [101 ──── 130]
           ↑ overlap ↑
           lines 51–60 appear in BOTH chunk 0 and chunk 1
```

Why overlap?
- Without it, a logical block that spans lines 58–62 gets cut in half.
- With it, every line is guaranteed to appear in full context in at least one chunk.

### The Chunk ID

```python
chunk_id = hashlib.md5(
    f"{repo}:{symbol.file_path}:{symbol.symbol}:{part}".encode()
).hexdigest()[:16]
```

- MD5 is used for speed, not security. Collisions don't matter here.
- `[:16]` keeps IDs short (64-bit of entropy, enough for a codebase).
- The same input always produces the same ID — so re-indexing is idempotent.

### Self-check
> A 200-line function with window=60, overlap=10 — how many chunks does it produce?
> (Hint: step=50, so count how many `start` values exist in `range(0, 200, 50)`.)

---

## 4. Building Embedding-Friendly Text

The `text` field is what the embedding model sees. It is constructed in `_build_chunk_text`:

```python
def _build_chunk_text(symbol, lines) -> str:
    parts = []
    if symbol.docstring:
        parts.append(f"# {symbol.docstring}")        # what it does
    if symbol.imports:
        parts.append(f"# imports: {', '.join(symbol.imports[:5])}")  # what it uses
    parts.append(f"# file: {symbol.file_path}")       # where it lives
    parts.append("\n".join(lines))                    # how it works
    return "\n".join(parts)
```

### Example Output

For `validate_jwt_token` in `auth/middleware.py`:

```
# Validates a Bearer JWT token from the request header.
# imports: jwt, datetime, fastapi
# file: /repo/auth/middleware.py

def validate_jwt_token(token: str) -> dict:
    """Validates a Bearer JWT token from the request header."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
```

### Why This Format Works

Without the context header, the embedding vector is purely syntactic — the model
sees `jwt.decode(token, SECRET_KEY, ...)` but has no label for *what* the function is.

With the header, the vector captures all four dimensions:

| Layer | What it contributes |
|---|---|
| Docstring comment | The human intent |
| Imports comment | Domain signals (e.g. "jwt" → auth context) |
| File path | Structural location |
| Code lines | Implementation details |

When a user queries "validate JWT", the embedding of the query aligns with the embedding
of the chunk because *both* mention "validate", "JWT", and the same domain vocabulary.

### Self-check
> If you remove the docstring line from the chunk text, what kind of queries would
> fail to find this function? What kind would still succeed?

---

## 5. Strategy 2 — Sliding Window (Fallback)

Source: [chunking/sliding_window.py](../chunking/sliding_window.py)

Used when the parser returns no symbols — for config files, unusual syntax, or
languages with no parser.

```python
def chunk_raw_file(file_path, repo, language, window=40, overlap=10):
    lines = open(file_path).readlines()
    step = window - overlap

    for i, start in enumerate(range(0, len(lines), step)):
        chunk_lines = lines[start : start + window]
        if not chunk_lines:
            break
        text = "".join(chunk_lines)
        chunk_id = hashlib.md5(f"{file_path}:{i}".encode()).hexdigest()[:16]

        chunks.append(CodeChunk(
            id=chunk_id,
            symbol=f"raw_chunk_{i}",   # no symbol name — positional
            symbol_type="module",       # whole file level
            start_line=start + 1,
            end_line=start + len(chunk_lines),
            text=text,
            ...
        ))
```

### Differences from symbol chunker

| | Symbol chunker | Sliding window |
|---|---|---|
| Input | Parsed `CodeSymbol` | Raw file path |
| Granularity | One function/class | Fixed-size line windows |
| `symbol` field | Function name | `raw_chunk_0`, `raw_chunk_1`, ... |
| `symbol_type` | `"function"` / `"class"` | `"module"` |
| Metadata quality | Rich (docstring, imports, decorators) | Minimal |
| Search quality | High | Lower, but better than nothing |

### Self-check
> For a 100-line file with window=40, overlap=10 (step=30), how many chunks are produced?
> List the start line of each chunk.

---

## 6. The Pipeline: Putting It Together

Source: [chunking/pipeline.py](../chunking/pipeline.py)

```python
def file_to_chunks(record: FileRecord, repo: str) -> list[CodeChunk]:
    symbols = parse_file(record.path, record.language)

    if symbols:
        chunks = []
        for symbol in symbols:
            chunks.extend(symbol_to_chunks(symbol, repo, record.last_commit))
        return chunks
    else:
        # Fallback: no parser available or file couldn't be parsed
        return chunk_raw_file(record.path, repo, record.language)
```

### Data flow

```
FileRecord (path, language, last_commit)
    │
    ▼
parse_file()
    │
    ├─ [Python]      → parse_python_file()     → list[CodeSymbol]
    ├─ [JS/TS/...]   → parse_with_treesitter() → list[CodeSymbol]
    └─ [unknown/fail]→ []
         │
         ▼ (if symbols)
    symbol_to_chunks()   ← for each symbol
         │
         └─ [short] → 1 chunk
         └─ [long]  → N overlapping chunks
         │
         ▼ (if no symbols)
    chunk_raw_file()     ← sliding window fallback
         │
         ▼
    list[CodeChunk]   → ready for embedding
```

### The `last_commit` field

`FileRecord.last_commit` is passed through as `commit_hash` on every chunk.
Later, when a file is re-scanned, the indexer can check:

```
stored_chunk.commit_hash == current_commit_hash
```

If they match, the file hasn't changed — skip re-embedding.
If they differ, delete old chunks and re-index.

This makes the indexer incremental and fast on large repos.

### Self-check
> What happens if `parse_file` returns an empty list for a Python file
> (e.g. the file has a syntax error)? Is the file silently skipped or still chunked?

---

## 7. Concept Summary

| Concept | Where it appears | Key insight |
|---|---|---|
| Token limits | `MAX_LINES = 60` | Hard cap driven by embedding model constraints |
| Overlapping windows | `overlap = 10`, `step = window - overlap` | Prevents semantic content from being split across boundary |
| Chunk metadata | `CodeChunk` dataclass | Every chunk carries enough info to stand alone |
| Embedding-friendly text | `_build_chunk_text` | Docstring + imports + file + code = richer vector |
| Symbol-level chunking | `symbol_to_chunks` | Aligns with semantic units humans actually search for |
| Sliding window fallback | `chunk_raw_file` | Graceful degradation for unparseable files |
| Idempotent IDs | `md5(repo:path:symbol:part)` | Re-indexing doesn't create duplicates |
| Commit hash | `CodeChunk.commit_hash` | Enables incremental re-indexing |

---

## 8. Exercises

Work through these in order. Each one builds on the last.

**Exercise 1 — Explore chunk counts**

```python
from scanner.repo_scanner import scan_repository
from chunking.pipeline import file_to_chunks
from parsers.dispatcher import parse_file

repo_path = "sample_repo"
for record in scan_repository(repo_path):
    symbols = parse_file(record.path, record.language)
    chunks  = file_to_chunks(record, "sample_repo")
    print(f"{record.relative_path}: {len(symbols)} symbols → {len(chunks)} chunks")
```

What do you notice? Are there ever more chunks than symbols? When?

---

**Exercise 2 — Inspect chunk text**

```python
from parsers.python_parser import parse_python_file
from chunking.symbol_chunker import symbol_to_chunks

symbols = parse_python_file("sample_repo/services/auth.py")
for sym in symbols:
    chunks = symbol_to_chunks(sym, "sample_repo")
    for chunk in chunks:
        print("=== CHUNK:", chunk.id, "===")
        print(chunk.text)
        print()
```

Verify that the docstring, imports, and file path appear in the text.

---

**Exercise 3 — Simulate a long function**

```python
from parsers.python_parser import CodeSymbol
from chunking.symbol_chunker import symbol_to_chunks, MAX_LINES

long_source = "\n".join([f"    x_{i} = {i}" for i in range(MAX_LINES + 30)])
sym = CodeSymbol(
    symbol="big_function",
    type="function",
    language="python",
    file_path="fake.py",
    start_line=1,
    end_line=MAX_LINES + 30,
    source=long_source,
)
chunks = symbol_to_chunks(sym, "myrepo")
print(f"Lines: {len(long_source.splitlines())}, Chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"  chunk {i}: {len(c.text.splitlines())} lines in text")
```

Verify that consecutive chunks share ~10 lines (the overlap).

---

**Exercise 4 — Token counting (stretch)**

Install `tiktoken` and add a token count to each chunk:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# Use it:
chunks = symbol_to_chunks(sym, "myrepo")
for chunk in chunks:
    print(f"{chunk.symbol}: {count_tokens(chunk.text)} tokens")
```

Are any chunks close to or over 512 tokens? What would you do about those?

---

## 9. Answers to Self-Checks

**Section 1:** You can't embed a whole file because it would exceed the token limit.
The model would silently truncate, and the resulting vector would only represent the beginning.

**Section 2:** The `id` should NOT change if only `commit_hash` changes. The ID is based on
`repo:path:symbol:part`, not the commit. `commit_hash` is stored as metadata — it's used for
invalidation, not identity.

**Section 3:** With `window=60, overlap=10, step=50`, for a 200-line function:
`range(0, 200, 50)` → starts at `[0, 50, 100, 150]` → **4 chunks**.

**Section 5:** For a 100-line file, `window=40, overlap=10, step=30`:
`range(0, 100, 30)` → starts at `[0, 30, 60, 90]` → **4 chunks** at lines 1, 31, 61, 91.

**Section 6:** If `parse_file` returns `[]` for a Python file (e.g. syntax error), the pipeline
falls back to `chunk_raw_file` — the file is still chunked via sliding window. Nothing is skipped.

---

## Next Step

[04-embedding-engine.md](04-embedding-engine.md) — Convert chunks into vectors.
