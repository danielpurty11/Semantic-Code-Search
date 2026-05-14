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
