# Step 1: Repository Scanner

**Week 1 | Goal:** Traverse a Git repo, detect languages, list files to be indexed.

---

## What This Component Does

- Walk directory trees
- Detect file languages by extension
- Respect `.gitignore` patterns
- Read Git metadata (commit hash, branch)
- Emit a list of `FileRecord` objects

---

## Concepts to Learn

| Concept | Resource |
|---|---|
| `os.walk` / `pathlib.Path` | Python stdlib |
| GitPython | Reading git repos programmatically |
| `.gitignore` parsing | `pathspec` library |
| Incremental scanning | Track file mtimes, skip unchanged |

---

## Code: `scanner/repo_scanner.py`

```python
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator
import git          # pip install gitpython
import pathspec     # pip install pathspec

LANGUAGE_MAP = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".java": "java",
    ".go":   "go",
    ".rs":   "rust",
    ".cpp":  "cpp",
    ".c":    "c",
    ".rb":   "ruby",
}

@dataclass
class FileRecord:
    path: str           # absolute path
    relative_path: str  # relative to repo root
    language: str
    size_bytes: int
    last_commit: str    # git commit hash


def load_gitignore(repo_root: str) -> pathspec.PathSpec:
    gitignore_path = Path(repo_root) / ".gitignore"
    patterns = ["**/.git/**", "**/node_modules/**", "**/__pycache__/**"]
    if gitignore_path.exists():
        patterns += gitignore_path.read_text().splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def get_commit_hash(repo: git.Repo) -> str:
    try:
        return repo.head.commit.hexsha[:8]
    except Exception:
        return "unknown"


def scan_repository(repo_path: str) -> Iterator[FileRecord]:
    root = Path(repo_path).resolve()
    spec = load_gitignore(str(root))

    try:
        repo = git.Repo(str(root), search_parent_directories=True)
        commit_hash = get_commit_hash(repo)
    except git.InvalidGitRepositoryError:
        commit_hash = "no-git"

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs in-place
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            abs_path = Path(dirpath) / filename
            rel_path = str(abs_path.relative_to(root))

            # Skip gitignored files
            if spec.match_file(rel_path):
                continue

            ext = abs_path.suffix.lower()
            language = LANGUAGE_MAP.get(ext)
            if not language:
                continue  # skip non-code files

            yield FileRecord(
                path=str(abs_path),
                relative_path=rel_path,
                language=language,
                size_bytes=abs_path.stat().st_size,
                last_commit=commit_hash,
            )
```

---

## Code: `scanner/incremental.py`

Track which files changed since last index using a SQLite timestamp store.

```python
import sqlite3
import os
from pathlib import Path

class IncrementalTracker:
    def __init__(self, db_path: str = "index.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_index (
                path TEXT PRIMARY KEY,
                mtime REAL,
                indexed_at REAL
            )
        """)
        self.conn.commit()

    def needs_reindex(self, path: str) -> bool:
        mtime = os.path.getmtime(path)
        row = self.conn.execute(
            "SELECT mtime FROM file_index WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return True
        return mtime > row[0]

    def mark_indexed(self, path: str):
        import time
        mtime = os.path.getmtime(path)
        self.conn.execute("""
            INSERT OR REPLACE INTO file_index (path, mtime, indexed_at)
            VALUES (?, ?, ?)
        """, (path, mtime, time.time()))
        self.conn.commit()
```

---

## Code: Test It

```python
# test_scanner.py
from scanner.repo_scanner import scan_repository

for record in scan_repository("./sample_repo"):
    print(f"[{record.language}] {record.relative_path} ({record.size_bytes} bytes)")
```

Expected output:

```
[python] services/auth.py (1240 bytes)
[python] services/payment.py (980 bytes)
[javascript] frontend/app.js (2100 bytes)
```

---

## Exercises

1. Add support for detecting monorepos (multiple `package.json` or `pyproject.toml`)
2. Add a `--since` flag that only scans files changed after a given date
3. Print a summary: total files per language

---

## What You Learned

- Traversing and filtering filesystems in Python
- Reading Git metadata with GitPython
- `.gitignore` pattern matching with `pathspec`
- Incremental indexing with SQLite

---

## Next Step

[02-parser-layer.md](02-parser-layer.md) — Parse code into functions, classes, and symbols.
