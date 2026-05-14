from parsers.python_parser import CodeSymbol, parse_python_file

TREESITTER_LANGUAGES = {"javascript", "typescript", "java", "go", "rust"}


def parse_file(file_path: str, language: str) -> list[CodeSymbol]:
    if language == "python":
        return parse_python_file(file_path)
    elif language in TREESITTER_LANGUAGES:
        from parsers.treesitter_parser import parse_with_treesitter
        return parse_with_treesitter(file_path, language)
    return []
