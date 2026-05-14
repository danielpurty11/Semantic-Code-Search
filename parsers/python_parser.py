import ast
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeSymbol:
    symbol: str
    type: str          # "function" | "class" | "method"
    language: str
    file_path: str
    start_line: int
    end_line: int
    source: str
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
