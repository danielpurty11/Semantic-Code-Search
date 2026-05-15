from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np

from chunking.models import CodeChunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


class CodeEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimensions: int = self.model.get_sentence_embedding_dimension()

    def embed_chunks(self, chunks: list[CodeChunk]) -> np.ndarray:
        """
        Embed a list of chunks. Returns an array of shape (N, dimensions).
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed raw texts in batches. Returns shape (N, dimensions).
        """
        return self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize so dot product == cosine similarity
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query. Returns shape (dimensions,).
        """
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
