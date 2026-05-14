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
