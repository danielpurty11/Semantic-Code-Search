import warnings
from typing import Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from tree_sitter_languages import get_language, get_parser

from parsers.python_parser import CodeSymbol

# S-expression queries per language: captures @func/@class for the whole node,
# @name for the identifier child that holds the symbol name.
QUERIES: dict[str, str] = {
    "javascript": """
        (function_declaration name: (identifier) @name) @func
        (method_definition name: (property_identifier) @name) @func
        (class_declaration name: (identifier) @name) @class
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @func
        (method_definition name: (property_identifier) @name) @func
        (class_declaration name: (type_identifier) @name) @class
    """,
    "java": """
        (method_declaration name: (identifier) @name) @func
        (class_declaration name: (identifier) @name) @class
    """,
    "go": """
        (function_declaration name: (identifier) @name) @func
        (method_declaration name: (field_identifier) @name) @func
        (type_spec name: (type_identifier) @name) @class
    """,
    "rust": """
        (function_item name: (identifier) @name) @func
        (struct_item name: (type_identifier) @name) @class
    """,
}


def parse_with_treesitter(file_path: str, language: str) -> list[CodeSymbol]:
    if language not in QUERIES:
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lang = get_language(language)
        parser = get_parser(language)

    source = open(file_path, "rb").read()
    tree = parser.parse(source)

    query = lang.query(QUERIES[language])
    captures = query.captures(tree.root_node)

    symbols: list[CodeSymbol] = []
    seen: set[tuple[int, int]] = set()

    for node, capture_name in captures:
        if capture_name not in ("func", "class"):
            continue

        key = (node.start_point[0], node.end_point[0])
        if key in seen:
            continue
        seen.add(key)

        name_node = node.child_by_field_name("name")
        symbol_name = (
            name_node.text.decode() if name_node else f"anonymous_{node.start_point[0] + 1}"
        )
        raw_source = source[node.start_byte : node.end_byte].decode(errors="replace")

        symbols.append(CodeSymbol(
            symbol=symbol_name,
            type="function" if capture_name == "func" else "class",
            language=language,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source=raw_source,
        ))

    return symbols
