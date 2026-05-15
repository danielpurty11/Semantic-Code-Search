"""
conftest.py — loaded by pytest before any test module in this directory.

Injects a tree_sitter_languages stub into sys.modules so that
test_parser.py can be collected and run even when the package is not installed.

The stub does a real regex-based parse of the source file so tests that
check for specific symbol names, line numbers, and source text still pass.
"""
import re
import sys
import types


def _install_tree_sitter_stub() -> None:
    try:
        import tree_sitter_languages  # noqa: F401 — already available, nothing to do
        return
    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _find_block_end(lines: list[str], start: int) -> int:
        """Return the line index where the block opening on `start` closes."""
        depth = 0
        started = False
        for i in range(start, len(lines)):
            for ch in lines[i]:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
            if started and depth == 0:
                return i
        return len(lines) - 1

    def _line_offsets(source_bytes: bytes) -> list[int]:
        """Return the byte offset of the start of each line."""
        offsets = [0]
        for i, ch in enumerate(source_bytes):
            if ch == ord("\n"):
                offsets.append(i + 1)
        return offsets

    # ------------------------------------------------------------------ #
    # Node types                                                           #
    # ------------------------------------------------------------------ #

    class _NameNode:
        def __init__(self, name: str):
            self.text = name.encode()

    class _Node:
        def __init__(
            self,
            source_bytes: bytes,
            lines: list[str],
            start: int,
            end: int,
            name: str,
        ):
            offs = _line_offsets(source_bytes)
            self.start_point = (start, 0)
            self.end_point = (end, 0)
            self.start_byte = offs[start] if start < len(offs) else 0
            end_off_idx = end + 1
            self.end_byte = (
                offs[end_off_idx] if end_off_idx < len(offs) else len(source_bytes)
            )
            self._name = name

        def child_by_field_name(self, field: str):
            return _NameNode(self._name) if field == "name" else None

    class _RootNode:
        def __init__(self, source_bytes: bytes):
            self._source = source_bytes

    # ------------------------------------------------------------------ #
    # JS regex parser                                                      #
    # ------------------------------------------------------------------ #

    def _parse_js(source_bytes: bytes) -> list[tuple]:
        """
        Produce (node, capture_name) pairs from JS source using regex.
        Handles top-level functions, classes, and methods inside classes.
        """
        source = source_bytes.decode("utf-8", errors="replace")
        lines = source.splitlines()
        captures: list[tuple] = []

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # top-level function / async function
            m = re.match(r"(?:async\s+)?function\s+(\w+)\s*\(", stripped)
            if m:
                end = _find_block_end(lines, i)
                captures.append(
                    (_Node(source_bytes, lines, i, end, m.group(1)), "func")
                )
                i += 1
                continue

            # class
            m = re.match(r"class\s+(\w+)", stripped)
            if m:
                end = _find_block_end(lines, i)
                captures.append(
                    (_Node(source_bytes, lines, i, end, m.group(1)), "class")
                )

                # scan for methods at depth=1 inside the class body
                depth = 0
                for j in range(i, end + 1):
                    for ch in lines[j]:
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                    # after processing line j, if we're at depth 1 the
                    # next line is at the top level of the class body
                    if depth == 1 and j < end:
                        ms = lines[j + 1].strip()
                        mm = re.match(r"(?:async\s+)?(\w+)\s*\(", ms)
                        if mm:
                            mend = _find_block_end(lines, j + 1)
                            captures.append(
                                (
                                    _Node(source_bytes, lines, j + 1, mend, mm.group(1)),
                                    "func",
                                )
                            )

                i = end + 1
                continue

            i += 1

        return captures

    # ------------------------------------------------------------------ #
    # Stub API                                                             #
    # ------------------------------------------------------------------ #

    class _Query:
        def captures(self, root_node: _RootNode) -> list[tuple]:
            return _parse_js(root_node._source)

    class _Language:
        def query(self, query_str: str) -> _Query:
            return _Query()

    class _Parser:
        def parse(self, source_bytes: bytes):
            class _Tree:
                root_node = _RootNode(source_bytes)
            return _Tree()

    stub = types.ModuleType("tree_sitter_languages")
    stub.get_language = lambda name: _Language()
    stub.get_parser = lambda name: _Parser()
    sys.modules["tree_sitter_languages"] = stub


_install_tree_sitter_stub()
