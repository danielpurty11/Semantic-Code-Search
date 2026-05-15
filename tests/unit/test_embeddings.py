"""
Tests for the embeddings module.

sentence-transformers and numpy are not available in this environment,
so both are mocked at the sys.modules level before any import of the
embeddings package occurs.
"""
import sys
import types
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Build minimal numpy stub so embeddings modules can be imported
# ---------------------------------------------------------------------------

class _FakeArray:
    """Minimal ndarray stand-in."""
    def __init__(self, data, shape=None):
        self._data = list(data)
        self.shape = shape or (len(self._data),)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __matmul__(self, other):
        # dot product: sum(a*b)
        return sum(a * b for a, b in zip(self._data, other._data))

    def __eq__(self, other):
        if isinstance(other, _FakeArray):
            return self._data == other._data
        return NotImplemented

    def __repr__(self):
        return f"FakeArray({self._data})"


def _fake_stack(arrays):
    data = [a._data for a in arrays]
    return _FakeArray(data, shape=(len(data), len(data[0]) if data else 0))


def _fake_save(path, arr):
    Path(path).write_text(repr(arr._data))


def _fake_load(path):
    import ast
    data = ast.literal_eval(Path(path).read_text())
    return _FakeArray(data)


np_stub = types.ModuleType("numpy")
np_stub.ndarray = _FakeArray
np_stub.stack = _fake_stack
np_stub.save = _fake_save
np_stub.load = _fake_load
sys.modules.setdefault("numpy", np_stub)

# ---------------------------------------------------------------------------
# Build minimal sentence_transformers stub
# ---------------------------------------------------------------------------

_DIMS = 384


class _FakeModel:
    def get_sentence_embedding_dimension(self):
        return _DIMS

    def encode(self, texts, batch_size=64, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            return _FakeArray([0.1] * _DIMS, shape=(_DIMS,))
        return [_FakeArray([0.1] * _DIMS, shape=(_DIMS,)) for _ in texts]


def _FakeSentenceTransformer(model_name):
    return _FakeModel()


st_stub = types.ModuleType("sentence_transformers")
st_stub.SentenceTransformer = _FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", st_stub)

# ---------------------------------------------------------------------------
# Now safe to import embeddings modules
# ---------------------------------------------------------------------------

from embeddings.encoder import CodeEmbedder, DEFAULT_MODEL, BATCH_SIZE
from embeddings.cache import (
    load_cached, save_cached, embed_with_cache,
    _cache_key, CACHE_DIR,
)
from embeddings.pipeline import embed_all_chunks
from chunking.models import CodeChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(symbol="foo", chunk_id="abc123") -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        repo="testrepo",
        language="python",
        file_path="f.py",
        symbol=symbol,
        symbol_type="function",
        start_line=1,
        end_line=5,
        text=f"def {symbol}(): pass",
    )


# ---------------------------------------------------------------------------
# CodeEmbedder
# ---------------------------------------------------------------------------

class TestCodeEmbedder:
    def setup_method(self):
        self.embedder = CodeEmbedder()

    def test_default_model_name(self):
        assert self.embedder.model_name == DEFAULT_MODEL

    def test_dimensions_set_from_model(self):
        assert self.embedder.dimensions == _DIMS

    def test_embed_chunks_returns_array(self):
        chunks = [_make_chunk("foo"), _make_chunk("bar")]
        result = self.embedder.embed_chunks(chunks)
        assert result is not None

    def test_embed_chunks_uses_text_field(self):
        chunk = _make_chunk("myfunc")
        with patch.object(self.embedder, "embed_texts", return_value=MagicMock()) as mock:
            self.embedder.embed_chunks([chunk])
            mock.assert_called_once_with(["def myfunc(): pass"])

    def test_embed_texts_returns_array(self):
        result = self.embedder.embed_texts(["hello", "world"])
        assert result is not None
        assert len(result) == 2

    def test_embed_query_returns_single_vector(self):
        result = self.embedder.embed_query("validate jwt token")
        assert result is not None
        assert result.shape == (_DIMS,)

    def test_custom_model_name_stored(self):
        embedder = CodeEmbedder(model_name="all-mpnet-base-v2")
        assert embedder.model_name == "all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_cache_key_is_hex(self):
        key = _cache_key("abc", "model")
        assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_length(self):
        key = _cache_key("abc", "model")
        assert len(key) == 32  # full md5 hex

    def test_cache_key_changes_with_chunk_id(self):
        assert _cache_key("id1", "model") != _cache_key("id2", "model")

    def test_cache_key_changes_with_model(self):
        assert _cache_key("id", "model-a") != _cache_key("id", "model-b")

    def test_cache_key_deterministic(self):
        assert _cache_key("x", "y") == _cache_key("x", "y")


class TestLoadSaveCached:
    def test_load_returns_none_when_missing(self, tmp_path):
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            result = load_cached("nonexistent", "model")
        assert result is None

    def test_save_then_load_roundtrip(self, tmp_path):
        vec = _FakeArray([0.1, 0.2, 0.3])
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            save_cached("chunk1", "model", vec)
            loaded = load_cached("chunk1", "model")
        assert loaded is not None
        assert list(loaded) == [0.1, 0.2, 0.3]

    def test_save_creates_cache_dir(self, tmp_path):
        subdir = tmp_path / "new_cache"
        vec = _FakeArray([1.0])
        with patch("embeddings.cache.CACHE_DIR", subdir):
            save_cached("c", "m", vec)
        assert subdir.exists()

    def test_different_chunk_ids_stored_separately(self, tmp_path):
        v1 = _FakeArray([1.0, 0.0])
        v2 = _FakeArray([0.0, 1.0])
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            save_cached("chunk-a", "model", v1)
            save_cached("chunk-b", "model", v2)
            r1 = load_cached("chunk-a", "model")
            r2 = load_cached("chunk-b", "model")
        assert list(r1) == [1.0, 0.0]
        assert list(r2) == [0.0, 1.0]

    def test_different_models_stored_separately(self, tmp_path):
        v1 = _FakeArray([1.0])
        v2 = _FakeArray([2.0])
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            save_cached("chunk", "model-a", v1)
            save_cached("chunk", "model-b", v2)
            r1 = load_cached("chunk", "model-a")
            r2 = load_cached("chunk", "model-b")
        assert list(r1) == [1.0]
        assert list(r2) == [2.0]


class TestEmbedWithCache:
    def _make_embedder(self):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.embed_chunks.return_value = [_FakeArray([0.5] * _DIMS)]
        return embedder

    def test_returns_list_of_chunk_vector_pairs(self, tmp_path):
        chunk = _make_chunk()
        embedder = self._make_embedder()
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            pairs = embed_with_cache([chunk], embedder)
        assert len(pairs) == 1
        assert pairs[0][0] is chunk

    def test_uncached_chunks_are_embedded(self, tmp_path):
        chunk = _make_chunk()
        embedder = self._make_embedder()
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            embed_with_cache([chunk], embedder)
        embedder.embed_chunks.assert_called_once_with([chunk])

    def test_cached_chunks_skip_model_call(self, tmp_path):
        chunk = _make_chunk(chunk_id="cached-id")
        vec = _FakeArray([0.9] * _DIMS)
        embedder = self._make_embedder()
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            save_cached(chunk.id, "test-model", vec)
            pairs = embed_with_cache([chunk], embedder)
        embedder.embed_chunks.assert_not_called()
        assert len(pairs) == 1

    def test_mixed_cached_and_uncached(self, tmp_path):
        cached_chunk = _make_chunk("cached", "id-cached")
        fresh_chunk = _make_chunk("fresh", "id-fresh")
        vec = _FakeArray([0.1] * _DIMS)
        embedder = self._make_embedder()
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            save_cached("id-cached", "test-model", vec)
            pairs = embed_with_cache([cached_chunk, fresh_chunk], embedder)
        assert len(pairs) == 2
        embedder.embed_chunks.assert_called_once_with([fresh_chunk])

    def test_newly_embedded_chunks_are_saved(self, tmp_path):
        chunk = _make_chunk(chunk_id="new-id")
        embedder = self._make_embedder()
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            embed_with_cache([chunk], embedder)
            loaded = load_cached("new-id", "test-model")
        assert loaded is not None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestEmbedAllChunks:
    def _make_embedder(self, n=2):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.embed_chunks.return_value = [_FakeArray([0.1] * _DIMS) for _ in range(n)]
        return embedder

    def test_returns_tuple_of_chunks_and_matrix(self, tmp_path):
        chunks = [_make_chunk("a", "id-a"), _make_chunk("b", "id-b")]
        embedder = self._make_embedder(2)
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            result_chunks, matrix = embed_all_chunks(chunks, embedder)
        assert len(result_chunks) == 2
        assert matrix is not None

    def test_matrix_shape(self, tmp_path):
        chunks = [_make_chunk("a", "id-a"), _make_chunk("b", "id-b")]
        embedder = self._make_embedder(2)
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            _, matrix = embed_all_chunks(chunks, embedder)
        assert matrix.shape == (2, _DIMS)

    def test_no_cache_calls_embed_directly(self, tmp_path):
        chunks = [_make_chunk("a", "id-a")]
        embedder = self._make_embedder(1)
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            embed_all_chunks(chunks, embedder, use_cache=False)
        embedder.embed_chunks.assert_called_once_with(chunks)

    def test_use_cache_true_goes_through_cache(self, tmp_path):
        chunks = [_make_chunk("a", "id-a")]
        embedder = self._make_embedder(1)
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            with patch("embeddings.pipeline.embed_with_cache", return_value=[(chunks[0], _FakeArray([0.1]*_DIMS))]) as mock_cache:
                embed_all_chunks(chunks, embedder, use_cache=True)
        mock_cache.assert_called_once_with(chunks, embedder)

    def test_chunks_in_output_match_input_order(self, tmp_path):
        c1 = _make_chunk("first", "id-1")
        c2 = _make_chunk("second", "id-2")
        embedder = self._make_embedder(2)
        with patch("embeddings.cache.CACHE_DIR", tmp_path):
            result_chunks, _ = embed_all_chunks([c1, c2], embedder)
        assert result_chunks[0].symbol == "first"
        assert result_chunks[1].symbol == "second"
