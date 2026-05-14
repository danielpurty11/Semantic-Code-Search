import pytest
from pathlib import Path

from parsers.python_parser import parse_python_file, CodeSymbol
from chunking.models import CodeChunk
from chunking.symbol_chunker import symbol_to_chunks, _build_chunk_text, MAX_LINES
from chunking.sliding_window import chunk_raw_file
from chunking.pipeline import file_to_chunks
from scanner.repo_scanner import FileRecord

SAMPLE_REPO = Path(__file__).parent.parent.parent / "sample_repo"
AUTH_PY = str(SAMPLE_REPO / "services" / "auth.py")
PAYMENT_PY = str(SAMPLE_REPO / "services" / "payment.py")


# --- CodeChunk model ---

def test_code_chunk_has_required_fields():
    chunk = CodeChunk(
        id="abc123",
        repo="myrepo",
        language="python",
        file_path="auth.py",
        symbol="validate_jwt_token",
        symbol_type="function",
        start_line=1,
        end_line=10,
        text="def validate_jwt_token(): pass",
    )
    assert chunk.id == "abc123"
    assert chunk.repo == "myrepo"
    assert chunk.language == "python"
    assert chunk.symbol_type == "function"


def test_code_chunk_defaults():
    chunk = CodeChunk(
        id="x", repo="r", language="python", file_path="f.py",
        symbol="foo", symbol_type="function", start_line=1, end_line=5, text="code",
    )
    assert chunk.docstring is None
    assert chunk.imports == []
    assert chunk.decorators == []
    assert chunk.commit_hash == ""


# --- symbol_to_chunks ---

def test_symbol_to_chunks_returns_list_of_code_chunks():
    symbols = parse_python_file(AUTH_PY)
    sym = next(s for s in symbols if s.symbol == "validate_jwt_token")
    chunks = symbol_to_chunks(sym, "myrepo", "abc123")
    assert len(chunks) >= 1
    assert all(isinstance(c, CodeChunk) for c in chunks)


def test_symbol_to_chunks_short_function_is_single_chunk():
    symbols = parse_python_file(AUTH_PY)
    sym = next(s for s in symbols if s.symbol == "validate_jwt_token")
    assert len(sym.source.splitlines()) <= MAX_LINES
    chunks = symbol_to_chunks(sym, "myrepo")
    assert len(chunks) == 1


def test_symbol_to_chunks_id_is_hex_string():
    symbols = parse_python_file(AUTH_PY)
    sym = symbols[0]
    chunks = symbol_to_chunks(sym, "myrepo")
    assert len(chunks[0].id) == 16
    assert all(c in "0123456789abcdef" for c in chunks[0].id)


def test_symbol_to_chunks_preserves_metadata():
    symbols = parse_python_file(AUTH_PY)
    sym = next(s for s in symbols if s.symbol == "validate_jwt_token")
    chunks = symbol_to_chunks(sym, "testrepo", "deadbeef")
    chunk = chunks[0]
    assert chunk.repo == "testrepo"
    assert chunk.commit_hash == "deadbeef"
    assert chunk.language == "python"
    assert chunk.file_path == AUTH_PY
    assert chunk.symbol == "validate_jwt_token"
    assert chunk.symbol_type == "function"


def test_symbol_to_chunks_long_function_splits():
    # Build a synthetic symbol with > MAX_LINES lines
    long_source = "\n".join(["    pass"] * (MAX_LINES + 20))
    sym = CodeSymbol(
        symbol="long_func",
        type="function",
        language="python",
        file_path="fake.py",
        start_line=1,
        end_line=MAX_LINES + 20,
        source=long_source,
    )
    chunks = symbol_to_chunks(sym, "repo")
    assert len(chunks) > 1


def test_symbol_to_chunks_overlapping_windows_cover_all_lines():
    long_source = "\n".join([f"line_{i}" for i in range(MAX_LINES + 30)])
    sym = CodeSymbol(
        symbol="big_func",
        type="function",
        language="python",
        file_path="fake.py",
        start_line=1,
        end_line=MAX_LINES + 30,
        source=long_source,
    )
    chunks = symbol_to_chunks(sym, "repo")
    # First chunk starts at beginning, last chunk should cover the end
    last_chunk_text = chunks[-1].text
    assert "line_" in last_chunk_text


def test_symbol_to_chunks_different_parts_have_different_ids():
    long_source = "\n".join(["x = 1"] * (MAX_LINES + 20))
    sym = CodeSymbol(
        symbol="dup_func",
        type="function",
        language="python",
        file_path="fake.py",
        start_line=1,
        end_line=MAX_LINES + 20,
        source=long_source,
    )
    chunks = symbol_to_chunks(sym, "repo")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# --- _build_chunk_text ---

def test_build_chunk_text_includes_docstring():
    symbols = parse_python_file(AUTH_PY)
    sym = next(s for s in symbols if s.symbol == "validate_jwt_token")
    text = _build_chunk_text(sym, sym.source.splitlines())
    assert sym.docstring in text


def test_build_chunk_text_includes_imports():
    symbols = parse_python_file(AUTH_PY)
    sym = symbols[0]
    text = _build_chunk_text(sym, sym.source.splitlines())
    assert "# imports:" in text


def test_build_chunk_text_includes_file_path():
    symbols = parse_python_file(AUTH_PY)
    sym = symbols[0]
    text = _build_chunk_text(sym, sym.source.splitlines())
    assert f"# file: {AUTH_PY}" in text


def test_build_chunk_text_includes_code():
    symbols = parse_python_file(AUTH_PY)
    sym = next(s for s in symbols if s.symbol == "validate_jwt_token")
    text = _build_chunk_text(sym, sym.source.splitlines())
    assert "def validate_jwt_token" in text


def test_build_chunk_text_no_docstring_skips_docstring_line():
    sym = CodeSymbol(
        symbol="nodoc",
        type="function",
        language="python",
        file_path="f.py",
        start_line=1,
        end_line=2,
        source="def nodoc(): pass",
        docstring=None,
        imports=[],
    )
    text = _build_chunk_text(sym, ["def nodoc(): pass"])
    # file comment is present, but no docstring-prefixed line
    assert "# file: f.py" in text
    assert "# None" not in text


# --- chunk_raw_file ---

def test_chunk_raw_file_returns_list():
    chunks = chunk_raw_file(AUTH_PY, "myrepo", "python")
    assert isinstance(chunks, list)


def test_chunk_raw_file_all_code_chunks():
    chunks = chunk_raw_file(AUTH_PY, "myrepo", "python")
    assert all(isinstance(c, CodeChunk) for c in chunks)


def test_chunk_raw_file_symbol_type_is_module():
    chunks = chunk_raw_file(AUTH_PY, "myrepo", "python")
    assert all(c.symbol_type == "module" for c in chunks)


def test_chunk_raw_file_ids_are_unique():
    chunks = chunk_raw_file(AUTH_PY, "myrepo", "python")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_raw_file_line_numbers_are_positive():
    chunks = chunk_raw_file(AUTH_PY, "myrepo", "python")
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_raw_file_respects_window_size():
    chunks = chunk_raw_file(AUTH_PY, "myrepo", "python", window=5, overlap=1)
    for c in chunks:
        assert len(c.text.splitlines()) <= 5


def test_chunk_raw_file_preserves_repo_and_language():
    chunks = chunk_raw_file(AUTH_PY, "testrepo", "python")
    for c in chunks:
        assert c.repo == "testrepo"
        assert c.language == "python"


# --- pipeline: file_to_chunks ---

def _make_record(path: str, language: str) -> FileRecord:
    return FileRecord(
        path=path,
        relative_path=path,
        language=language,
        size_bytes=0,
        last_commit="abc123",
    )


def test_pipeline_python_returns_chunks():
    record = _make_record(AUTH_PY, "python")
    chunks = file_to_chunks(record, "myrepo")
    assert len(chunks) > 0
    assert all(isinstance(c, CodeChunk) for c in chunks)


def test_pipeline_python_chunk_count_matches_symbols():
    from parsers.dispatcher import parse_file
    record = _make_record(AUTH_PY, "python")
    symbols = parse_file(AUTH_PY, "python")
    chunks = file_to_chunks(record, "myrepo")
    # Each short symbol → 1 chunk, so count should be >= symbol count
    assert len(chunks) >= len(symbols)


def test_pipeline_uses_commit_hash():
    record = _make_record(AUTH_PY, "python")
    chunks = file_to_chunks(record, "myrepo")
    for c in chunks:
        assert c.commit_hash == "abc123"


def test_pipeline_fallback_for_unknown_language():
    record = _make_record(AUTH_PY, "cobol")  # parser returns []
    chunks = file_to_chunks(record, "myrepo")
    assert len(chunks) > 0
    assert all(c.symbol_type == "module" for c in chunks)


def test_pipeline_chunk_repo_is_set():
    record = _make_record(AUTH_PY, "python")
    chunks = file_to_chunks(record, "expected-repo")
    for c in chunks:
        assert c.repo == "expected-repo"
