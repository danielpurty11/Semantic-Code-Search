"""
Tests for the vectorstore module.

faiss, chromadb, and numpy are not installable in this environment,
so they are stubbed via sys.modules before any vectorstore import.
"""
import sys
import types
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# numpy stub (reuse pattern from test_embeddings)
# ---------------------------------------------------------------------------

class _FakeArray:
    def __init__(self, data, shape=None):
        if isinstance(data, list) and data and isinstance(data[0], list):
            self._data = data
            self.shape = (len(data), len(data[0]))
        else:
            self._data = list(data)
            self.shape = shape or (len(self._data),)

    def reshape(self, *args):
        flat = self._data if not isinstance(self._data[0], list) else [x for row in self._data for x in row]
        return _FakeArray(flat, shape=args)

    def astype(self, dtype):
        return self

    def tolist(self):
        return self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


np_stub = types.ModuleType("numpy")
np_stub.ndarray = _FakeArray

def _fake_array(*args, **kwargs):
    if args:
        return _FakeArray(args[0])
    return _FakeArray([])

np_stub.array = _fake_array
sys.modules.setdefault("numpy", np_stub)


# ---------------------------------------------------------------------------
# faiss stub
# ---------------------------------------------------------------------------

class _FakeIndex:
    def __init__(self, dims):
        self.dims   = dims
        self.ntotal = 0
        self._vecs  = []

    def add(self, vectors):
        for v in vectors:
            self._vecs.append(list(v) if hasattr(v, '__iter__') else v)
        self.ntotal = len(self._vecs)

    def search(self, query, k):
        # Return fake scores and indices for however many we have
        n = min(k, self.ntotal)
        scores  = [[0.9 - i * 0.05 for i in range(n)]]
        indices = [[i for i in range(n)]]
        return scores, indices


faiss_stub = types.ModuleType("faiss")
faiss_stub.IndexFlatIP = _FakeIndex

_saved_index = {}

def _write_index(index, path):
    _saved_index[path] = index

def _read_index(path):
    return _saved_index[path]

faiss_stub.write_index = _write_index
faiss_stub.read_index  = _read_index
sys.modules.setdefault("faiss", faiss_stub)


# ---------------------------------------------------------------------------
# chromadb stub
# ---------------------------------------------------------------------------

class _FakeCollection:
    def __init__(self):
        self._items = {}  # id -> {embedding, document, metadata}

    def add(self, ids, embeddings, documents, metadatas):
        for i, id_ in enumerate(ids):
            self._items[id_] = {
                "embedding": embeddings[i],
                "document":  documents[i],
                "metadata":  metadatas[i],
            }

    def query(self, query_embeddings, n_results, include, where=None):
        items = list(self._items.items())
        if where:
            items = [(id_, v) for id_, v in items
                     if all(v["metadata"].get(k) == val for k, val in where.items())]
        items = items[:n_results]
        ids    = [[id_  for id_, _ in items]]
        dists  = [[0.1 + i * 0.05 for i in range(len(items))]]
        metas  = [[v["metadata"] for _, v in items]]
        docs   = [[v["document"]  for _, v in items]]
        return {"ids": ids, "distances": dists, "metadatas": metas, "documents": docs}

    def count(self):
        return len(self._items)


class _FakeChromaClient:
    def __init__(self, path):
        self._collections = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = _FakeCollection()
        return self._collections[name]


chroma_stub = types.ModuleType("chromadb")
chroma_stub.PersistentClient = _FakeChromaClient
sys.modules.setdefault("chromadb", chroma_stub)


# ---------------------------------------------------------------------------
# Now safe to import
# ---------------------------------------------------------------------------

from chunking.models import CodeChunk
from vectorstore.models import SearchResult
from vectorstore.faiss_store import FaissStore
from vectorstore.chroma_store import ChromaStore, _chunk_metadata, _metadata_to_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMS = 4

def _make_chunk(symbol="foo", chunk_id="id1", language="python") -> CodeChunk:
    return CodeChunk(
        id=chunk_id, repo="repo", language=language,
        file_path="f.py", symbol=symbol, symbol_type="function",
        start_line=1, end_line=5, text=f"def {symbol}(): pass",
        docstring=f"{symbol} docstring",
    )

def _make_vectors(n: int) -> _FakeArray:
    return _FakeArray([[0.1] * DIMS for _ in range(n)], shape=(n, DIMS))

def _single_vec() -> _FakeArray:
    return _FakeArray([0.1] * DIMS, shape=(DIMS,))


# ---------------------------------------------------------------------------
# SearchResult model
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_fields(self):
        chunk = _make_chunk()
        r = SearchResult(chunk=chunk, score=0.9, rank=1)
        assert r.chunk is chunk
        assert r.score == 0.9
        assert r.rank == 1


# ---------------------------------------------------------------------------
# FaissStore
# ---------------------------------------------------------------------------

class TestFaissStore:
    def test_initial_size_is_zero(self):
        store = FaissStore(dimensions=DIMS)
        assert store.size == 0

    def test_add_increases_size(self):
        store = FaissStore(dimensions=DIMS)
        chunks = [_make_chunk("a", "id-a"), _make_chunk("b", "id-b")]
        store.add(chunks, _make_vectors(2))
        assert store.size == 2

    def test_add_multiple_batches_accumulates(self):
        store = FaissStore(dimensions=DIMS)
        store.add([_make_chunk("a", "id-a")], _make_vectors(1))
        store.add([_make_chunk("b", "id-b")], _make_vectors(1))
        assert store.size == 2

    def test_search_returns_list_of_search_results(self):
        store = FaissStore(dimensions=DIMS)
        store.add([_make_chunk("foo", "id1")], _make_vectors(1))
        results = store.search(_single_vec(), top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_empty_store_returns_empty(self):
        store = FaissStore(dimensions=DIMS)
        results = store.search(_single_vec(), top_k=5)
        assert results == []

    def test_search_top_k_respected(self):
        store = FaissStore(dimensions=DIMS)
        chunks = [_make_chunk(f"fn{i}", f"id{i}") for i in range(5)]
        store.add(chunks, _make_vectors(5))
        results = store.search(_single_vec(), top_k=3)
        assert len(results) <= 3

    def test_search_ranks_are_1_based(self):
        store = FaissStore(dimensions=DIMS)
        chunks = [_make_chunk(f"fn{i}", f"id{i}") for i in range(3)]
        store.add(chunks, _make_vectors(3))
        results = store.search(_single_vec(), top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_search_scores_are_floats(self):
        store = FaissStore(dimensions=DIMS)
        store.add([_make_chunk("a", "id-a")], _make_vectors(1))
        results = store.search(_single_vec(), top_k=1)
        assert isinstance(results[0].score, float)

    def test_search_result_chunk_is_code_chunk(self):
        chunk = _make_chunk("myfunc", "myid")
        store = FaissStore(dimensions=DIMS)
        store.add([chunk], _make_vectors(1))
        results = store.search(_single_vec(), top_k=1)
        assert isinstance(results[0].chunk, CodeChunk)
        assert results[0].chunk.symbol == "myfunc"

    def test_save_and_load(self, tmp_path):
        index_path  = str(tmp_path / "test.index")
        chunks_path = str(tmp_path / "test.pkl")
        chunk = _make_chunk("saved_fn", "saved-id")

        store = FaissStore(DIMS, index_path=index_path, chunks_path=chunks_path)
        store.add([chunk], _make_vectors(1))
        store.save()

        # Verify pickle file was written
        assert Path(chunks_path).exists()

        loaded = FaissStore(DIMS, index_path=index_path, chunks_path=chunks_path)
        loaded.load()
        assert loaded.size == 1

    def test_save_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b"
        store = FaissStore(DIMS,
                           index_path=str(nested / "idx"),
                           chunks_path=str(nested / "chunks.pkl"))
        chunk = _make_chunk()
        store.add([chunk], _make_vectors(1))
        store.save()
        assert nested.exists()


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------

class TestChromaStore:
    def test_initial_size_is_zero(self):
        store = ChromaStore(persist_dir="/tmp/chroma_test")
        assert store.size == 0

    def test_add_increases_size(self):
        store = ChromaStore()
        chunks = [_make_chunk("a", "id-a"), _make_chunk("b", "id-b")]
        store.add(chunks, _make_vectors(2))
        assert store.size == 2

    def test_search_returns_search_results(self):
        store = ChromaStore()
        store.add([_make_chunk("fn", "id1")], _make_vectors(1))
        results = store.search(_single_vec(), top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_empty_store_returns_empty(self):
        store = ChromaStore(persist_dir="/tmp/empty_chroma", collection="empty_col")
        results = store.search(_single_vec(), top_k=5)
        assert results == []

    def test_search_top_k_respected(self):
        store = ChromaStore(collection="topk_test")
        chunks = [_make_chunk(f"fn{i}", f"topk-{i}") for i in range(5)]
        store.add(chunks, _make_vectors(5))
        results = store.search(_single_vec(), top_k=3)
        assert len(results) <= 3

    def test_search_ranks_are_1_based(self):
        store = ChromaStore(collection="rank_test")
        chunks = [_make_chunk(f"fn{i}", f"rank-{i}") for i in range(3)]
        store.add(chunks, _make_vectors(3))
        results = store.search(_single_vec(), top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_search_score_is_1_minus_distance(self):
        store = ChromaStore(collection="score_test")
        store.add([_make_chunk("fn", "score-id")], _make_vectors(1))
        results = store.search(_single_vec(), top_k=1)
        # distance=0.1 → score=0.9
        assert abs(results[0].score - 0.9) < 1e-6

    def test_search_with_where_filter(self):
        store = ChromaStore(collection="filter_test")
        py_chunk = _make_chunk("py_fn", "py-id", language="python")
        js_chunk = _make_chunk("js_fn", "js-id", language="javascript")
        store.add([py_chunk, js_chunk], _make_vectors(2))
        results = store.search(_single_vec(), top_k=5, where={"language": "python"})
        assert all(r.chunk.language == "python" for r in results)

    def test_chunk_reconstructed_from_metadata(self):
        chunk = _make_chunk("reconstruct_me", "rec-id")
        store = ChromaStore(collection="reconstruct_test")
        store.add([chunk], _make_vectors(1))
        results = store.search(_single_vec(), top_k=1)
        r = results[0]
        assert r.chunk.symbol == "reconstruct_me"
        assert r.chunk.id == "rec-id"
        assert r.chunk.language == "python"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestChunkMetadata:
    def test_all_required_keys_present(self):
        chunk = _make_chunk()
        meta = _chunk_metadata(chunk)
        for key in ("repo", "language", "file_path", "symbol", "symbol_type",
                    "start_line", "end_line", "commit_hash", "docstring"):
            assert key in meta

    def test_none_docstring_stored_as_empty_string(self):
        chunk = _make_chunk()
        chunk.docstring = None
        meta = _chunk_metadata(chunk)
        assert meta["docstring"] == ""


class TestMetadataToChunk:
    def test_roundtrip(self):
        original = _make_chunk("roundtrip", "rt-id")
        meta = _chunk_metadata(original)
        restored = _metadata_to_chunk("rt-id", meta, original.text)
        assert restored.id       == original.id
        assert restored.symbol   == original.symbol
        assert restored.language == original.language
        assert restored.repo     == original.repo
        assert restored.text     == original.text

    def test_empty_docstring_becomes_none(self):
        chunk = _make_chunk()
        chunk.docstring = None
        meta = _chunk_metadata(chunk)
        restored = _metadata_to_chunk(chunk.id, meta, chunk.text)
        assert restored.docstring is None
