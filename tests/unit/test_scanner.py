import pytest
from pathlib import Path
from scanner.repo_scanner import scan_repository, scan_summary, FileRecord
from scanner.incremental import IncrementalTracker


SAMPLE_REPO = str(Path(__file__).parent.parent.parent / "sample_repo")


def test_scan_returns_file_records():
    records = list(scan_repository(SAMPLE_REPO))
    assert len(records) > 0
    for r in records:
        assert isinstance(r, FileRecord)
        assert r.language in ("python", "javascript", "typescript", "java", "go")
        assert r.size_bytes > 0


def test_scan_detects_languages():
    records = list(scan_repository(SAMPLE_REPO))
    languages = {r.language for r in records}
    assert "python" in languages
    assert "javascript" in languages


def test_scan_relative_paths():
    records = list(scan_repository(SAMPLE_REPO))
    for r in records:
        assert not r.relative_path.startswith("/"), "relative_path should not be absolute"


def test_scan_summary():
    summary = scan_summary(SAMPLE_REPO)
    assert "python" in summary
    assert summary["python"]["files"] >= 2
    assert summary["python"]["total_bytes"] > 0


def test_incremental_tracker(tmp_path):
    db = str(tmp_path / "test.db")
    tracker = IncrementalTracker(db_path=db)

    # Create a temp file
    test_file = tmp_path / "sample.py"
    test_file.write_text("def foo(): pass")

    path = str(test_file)
    assert tracker.needs_reindex(path) is True

    tracker.mark_indexed(path)
    assert tracker.needs_reindex(path) is False

    # Simulate file change
    import time
    time.sleep(0.05)
    test_file.write_text("def foo(): return 1")

    assert tracker.needs_reindex(path) is True

    tracker.close()


def test_scan_ignores_pycache(tmp_path):
    # Create a __pycache__ file that should be skipped
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-312.pyc").write_bytes(b"fake bytecode")
    (tmp_path / "real.py").write_text("def hello(): pass")

    records = list(scan_repository(str(tmp_path)))
    paths = [r.relative_path for r in records]
    assert "real.py" in paths
    assert not any("__pycache__" in p for p in paths)
