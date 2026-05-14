from parsers.python_parser import CodeSymbol
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
        chunks.append(_make_chunk(symbol, repo, commit_hash, chunk_lines, i))

    return chunks


def _make_chunk(
    symbol: CodeSymbol,
    repo: str,
    commit_hash: str,
    lines: list[str],
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
