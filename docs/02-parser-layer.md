# Step 2: Parser Layer

**Week 1 | Goal:** Parse source files into structured metadata — functions, classes, imports, docstrings.

---

## What This Component Does

Takes a raw source file and outputs structured `CodeSymbol` objects:

```json
{
  "symbol": "validate_jwt_token",
  "type": "function",
  "language": "python",
  "file": "auth/middleware.py",
  "start_line": 42,
  "end_line": 68,
  "docstring": "Validates a Bearer JWT token from the request header.",
  "imports": ["jwt", "datetime"]
}
```

---

## Concepts to Learn

| Concept | Description |
|---|---|
| AST (Abstract Syntax Tree) | Tree representation of code structure |
| Tree-sitter | Universal parser for 40+ languages |
| Python `ast` module | Built-in Python AST |
| Node types | `function_definition`, `class_definition`, etc |

---

## Part A: Python Parser (using `ast`)

```python
# parsers/python_parser.py
import ast
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CodeSymbol:
    symbol: str
    type: str                    # "function" | "class" | "method"
    language: str
    file_path: str
    start_line: int
    end_line: int
    source: str                  # raw code text
    docstring: Optional[str] = None
    imports: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)


def extract_imports(tree: ast.Module) -> list[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def parse_python_file(file_path: str) -> list[CodeSymbol]:
    source_code = open(file_path).read()
    tree = ast.parse(source_code)
    lines = source_code.splitlines()
    imports = extract_imports(tree)
    symbols = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym_type = "function"
        elif isinstance(node, ast.ClassDef):
            sym_type = "class"
        else:
            continue

        start = node.lineno
        end = node.end_lineno or start
        raw_source = "\n".join(lines[start - 1 : end])
        docstring = ast.get_docstring(node)
        decorators = [
            ast.unparse(d) for d in getattr(node, "decorator_list", [])
        ]

        symbols.append(CodeSymbol(
            symbol=node.name,
            type=sym_type,
            language="python",
            file_path=file_path,
            start_line=start,
            end_line=end,
            source=raw_source,
            docstring=docstring,
            imports=imports,
            decorators=decorators,
        ))

    return symbols
```

---

## Part B: Universal Parser (Tree-sitter)

Tree-sitter parses ANY language the same way. You write a query in S-expression syntax.

### Install

```bash
pip install tree-sitter tree-sitter-languages
```

### JavaScript/TypeScript Parser

```python
# parsers/treesitter_parser.py
from tree_sitter_languages import get_language, get_parser
from dataclasses import dataclass
from typing import Optional

@dataclass
class CodeSymbol:
    symbol: str
    type: str
    language: str
    file_path: str
    start_line: int
    end_line: int
    source: str
    docstring: Optional[str] = None


FUNCTION_QUERIES = {
    "javascript": """
        (function_declaration name: (identifier) @name) @func
        (method_definition name: (property_identifier) @name) @func
        (arrow_function) @func
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @func
        (method_definition name: (property_identifier) @name) @func
    """,
    "python": """
        (function_definition name: (identifier) @name) @func
        (class_definition name: (identifier) @name) @class
    """,
}


def parse_with_treesitter(file_path: str, language: str) -> list[CodeSymbol]:
    if language not in FUNCTION_QUERIES:
        return []

    lang = get_language(language)
    parser = get_parser(language)
    source = open(file_path, "rb").read()
    tree = parser.parse(source)

    query = lang.query(FUNCTION_QUERIES[language])
    captures = query.captures(tree.root_node)

    symbols = []
    seen = set()

    for node, capture_name in captures:
        if capture_name == "name":
            continue
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        key = (start_line, end_line)
        if key in seen:
            continue
        seen.add(key)

        # Get the name from the sibling capture
        name_node = node.child_by_field_name("name")
        symbol_name = name_node.text.decode() if name_node else f"anonymous_{start_line}"

        raw_source = source[node.start_byte:node.end_byte].decode(errors="replace")

        symbols.append(CodeSymbol(
            symbol=symbol_name,
            type="function" if "func" in capture_name else "class",
            language=language,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source=raw_source,
        ))

    return symbols
```

---

## Part C: Dispatcher — Choose Parser by Language

```python
# parsers/dispatcher.py
from parsers.python_parser import parse_python_file, CodeSymbol
from parsers.treesitter_parser import parse_with_treesitter

def parse_file(file_path: str, language: str) -> list[CodeSymbol]:
    if language == "python":
        return parse_python_file(file_path)
    elif language in ("javascript", "typescript", "java", "go", "rust"):
        return parse_with_treesitter(file_path, language)
    return []
```

---

## Test It

```python
# test_parser.py
from parsers.dispatcher import parse_file

symbols = parse_file("sample_repo/auth/middleware.py", "python")
for s in symbols:
    print(f"[{s.type}] {s.symbol} (lines {s.start_line}-{s.end_line})")
    if s.docstring:
        print(f"  docstring: {s.docstring[:60]}")
```

---

## How Tree-sitter Queries Work

Tree-sitter uses S-expression pattern matching:

```
(function_declaration          <- node type
  name: (identifier) @name)    <- capture the name node
@func                          <- capture the whole function
```

You can explore node types interactively at: https://tree-sitter.github.io/tree-sitter/playground

---

## Exercises

1. Add a parser for Go using Tree-sitter (`function_declaration`)
2. Extract method bodies from inside classes separately
3. Build a stats function: how many functions per file?

---

## What You Learned

- Python `ast` module for static code analysis
- Tree-sitter for universal multi-language parsing
- Extracting structured metadata from source code
- S-expression query syntax for tree-sitter

---

## Next Step

[03-chunking-engine.md](03-chunking-engine.md) — Split code into searchable units.
