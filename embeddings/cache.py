from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE_DIR = Path(".embedding_cache")


def _cache_key(chunk_id: str, model_name: str) -> str:
    return hashlib.md5(f"{chunk_id}:{model_name}".encode()).hexdigest()


def load_cached(chunk_id: str, model_name: str) -> np.ndarray | None:
    key = _cache_key(chunk_id, model_name)
    path = CACHE_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    return None


def save_cached(chunk_id: str, model_name: str, vector: np.ndarray) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(chunk_id, model_name)
    np.save(CACHE_DIR / f"{key}.npy", vector)


def embed_with_cache(
    chunks: list,
    embedder,
) -> list[tuple]:
    """
    Embed chunks, using cache where available.
    Returns list of (chunk, vector) pairs in the same order as input.
    """
    results: list[tuple] = []
    to_embed: list = []

    for chunk in chunks:
        cached = load_cached(chunk.id, embedder.model_name)
        if cached is not None:
            results.append((chunk, cached))
        else:
            to_embed.append(chunk)

    if to_embed:
        vectors = embedder.embed_chunks(to_embed)
        for chunk, vector in zip(to_embed, vectors):
            save_cached(chunk.id, embedder.model_name, vector)
            results.append((chunk, vector))

    return results
