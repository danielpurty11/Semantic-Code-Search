from dataclasses import dataclass
from chunking.models import CodeChunk


@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float   # cosine similarity: 1.0 = identical, 0.0 = unrelated
    rank: int      # 1-based position in result list
