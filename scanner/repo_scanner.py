import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import git
import pathspec

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

# Directories always skipped regardless of .gitignore
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".tox",
}


@dataclass
class FileRecord:
    path: str           # absolute path on disk
    relative_path: str  # relative to repo root
    language: str
    size_bytes: int
    last_commit: str    # short git commit hash


def load_gitignore(repo_root: str) -> pathspec.PathSpec:
    """Load .gitignore patterns and combine with sensible defaults."""
    gitignore_path = Path(repo_root) / ".gitignore"
    patterns = [
        "**/.git/**",
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.DS_Store",
    ]
    if gitignore_path.exists():
        raw = gitignore_path.read_text(encoding="utf-8", errors="ignore")
        # Filter blank lines and comments
        extra = [l for l in raw.splitlines() if l.strip() and not l.startswith("#")]
        patterns.extend(extra)
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def get_commit_hash(repo: git.Repo) -> str:
    try:
        return repo.head.commit.hexsha[:8]
    except Exception:
        return "unknown"


def get_branch(repo: git.Repo) -> str:
    try:
        return repo.active_branch.name
    except Exception:
        return "detached"


def scan_repository(repo_path: str) -> Iterator[FileRecord]:
    """
    Walk a repository and yield FileRecord for every source file.

    Skips:
    - Files matched by .gitignore
    - Binary / non-code files
    - Hidden directories
    """
    root = Path(repo_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")

    spec = load_gitignore(str(root))

    try:
        repo = git.Repo(str(root), search_parent_directories=True)
        commit_hash = get_commit_hash(repo)
    except git.InvalidGitRepositoryError:
        commit_hash = "no-git"

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune unwanted dirs in-place (modifies os.walk traversal)
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            abs_path = Path(dirpath) / filename
            rel_path = str(abs_path.relative_to(root))

            if spec.match_file(rel_path):
                continue

            ext = abs_path.suffix.lower()
            language = LANGUAGE_MAP.get(ext)
            if not language:
                continue

            try:
                size = abs_path.stat().st_size
            except OSError:
                continue

            yield FileRecord(
                path=str(abs_path),
                relative_path=rel_path,
                language=language,
                size_bytes=size,
                last_commit=commit_hash,
            )


def scan_summary(repo_path: str) -> dict:
    """Return a summary dict: language -> file count and total bytes."""
    summary: dict[str, dict] = {}
    for record in scan_repository(repo_path):
        lang = record.language
        if lang not in summary:
            summary[lang] = {"files": 0, "total_bytes": 0}
        summary[lang]["files"] += 1
        summary[lang]["total_bytes"] += record.size_bytes
    return summary
