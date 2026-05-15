from __future__ import annotations

import numpy as np

from chunking.models import CodeChunk
from embeddings.encoder import CodeEmbedder
from embeddings.cache import embed_with_cache


def embed_all_chunks(
    chunks: list[CodeChunk],
    embedder: CodeEmbedder,
    use_cache: bool = True,
) -> tuple[list[CodeChunk], np.ndarray]:
    """
    Embed all chunks.
    Returns (chunks, matrix) where matrix[i] is the vector for chunks[i].
    """
    if use_cache:
        pairs = embed_with_cache(chunks, embedder)
    else:
        vectors = embedder.embed_chunks(chunks)
        pairs = list(zip(chunks, vectors))

    ordered_chunks = [pair[0] for pair in pairs]
    matrix = np.stack([pair[1] for pair in pairs])

    return ordered_chunks, matrix
