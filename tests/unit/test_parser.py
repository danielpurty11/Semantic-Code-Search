import pytest
from pathlib import Path

from parsers.python_parser import parse_python_file, CodeSymbol
from parsers.treesitter_parser import parse_with_treesitter
from parsers.dispatcher import parse_file

SAMPLE_REPO = Path(__file__).parent.parent.parent / "sample_repo"
AUTH_PY = str(SAMPLE_REPO / "services" / "auth.py")
PAYMENT_PY = str(SAMPLE_REPO / "services" / "payment.py")
APP_JS = str(SAMPLE_REPO / "frontend" / "app.js")


# --- Python parser ---

def test_python_parser_returns_code_symbols():
    symbols = parse_python_file(AUTH_PY)
    assert len(symbols) > 0
    for s in symbols:
        assert isinstance(s, CodeSymbol)


def test_python_parser_detects_functions():
    symbols = parse_python_file(AUTH_PY)
    names = [s.symbol for s in symbols]
    assert "validate_jwt_token" in names
    assert "create_jwt_token" in names


def test_python_parser_extracts_docstrings():
    symbols = parse_python_file(AUTH_PY)
    sym = next(s for s in symbols if s.symbol == "validate_jwt_token")
    assert sym.docstring is not None
    assert "JWT" in sym.docstring


def test_python_parser_extracts_imports():
    symbols = parse_python_file(AUTH_PY)
    sym = symbols[0]
    assert "jwt" in sym.imports
    assert "datetime" in sym.imports


def test_python_parser_sets_language():
    symbols = parse_python_file(AUTH_PY)
    for s in symbols:
        assert s.language == "python"


def test_python_parser_line_numbers():
    symbols = parse_python_file(AUTH_PY)
    sym = next(s for s in symbols if s.symbol == "validate_jwt_token")
    assert sym.start_line > 0
    assert sym.end_line >= sym.start_line


def test_python_parser_source_contains_def():
    symbols = parse_python_file(AUTH_PY)
    sym = next(s for s in symbols if s.symbol == "validate_jwt_token")
    assert "def validate_jwt_token" in sym.source


def test_python_parser_detects_class():
    symbols = parse_python_file(PAYMENT_PY)
    class_syms = [s for s in symbols if s.type == "class"]
    names = [s.symbol for s in class_syms]
    assert "PaymentService" in names
    assert "PaymentGatewayError" in names


def test_python_parser_class_docstring():
    symbols = parse_python_file(PAYMENT_PY)
    sym = next(s for s in symbols if s.symbol == "PaymentService")
    assert sym.docstring is not None
    assert "payment" in sym.docstring.lower()


def test_python_parser_methods_inside_class():
    symbols = parse_python_file(PAYMENT_PY)
    names = [s.symbol for s in symbols]
    assert "process_payment" in names
    assert "retry_failed_payment" in names


def test_python_parser_file_path():
    symbols = parse_python_file(AUTH_PY)
    for s in symbols:
        assert s.file_path == AUTH_PY


# --- Tree-sitter parser ---

def test_treesitter_javascript_functions():
    symbols = parse_with_treesitter(APP_JS, "javascript")
    names = [s.symbol for s in symbols]
    assert "fetchUserProfile" in names
    assert "validateEmail" in names


def test_treesitter_javascript_class():
    symbols = parse_with_treesitter(APP_JS, "javascript")
    class_syms = [s for s in symbols if s.type == "class"]
    assert any(s.symbol == "AuthClient" for s in class_syms)


def test_treesitter_javascript_methods():
    symbols = parse_with_treesitter(APP_JS, "javascript")
    names = [s.symbol for s in symbols]
    assert "login" in names
    assert "logout" in names


def test_treesitter_sets_language():
    symbols = parse_with_treesitter(APP_JS, "javascript")
    for s in symbols:
        assert s.language == "javascript"


def test_treesitter_line_numbers():
    symbols = parse_with_treesitter(APP_JS, "javascript")
    for s in symbols:
        assert s.start_line > 0
        assert s.end_line >= s.start_line


def test_treesitter_source_not_empty():
    symbols = parse_with_treesitter(APP_JS, "javascript")
    for s in symbols:
        assert len(s.source) > 0


def test_treesitter_unsupported_language_returns_empty():
    symbols = parse_with_treesitter(APP_JS, "cobol")
    assert symbols == []


# --- Dispatcher ---

def test_dispatcher_routes_python():
    symbols = parse_file(AUTH_PY, "python")
    assert len(symbols) > 0
    assert all(s.language == "python" for s in symbols)


def test_dispatcher_routes_javascript():
    symbols = parse_file(APP_JS, "javascript")
    assert len(symbols) > 0
    assert all(s.language == "javascript" for s in symbols)


def test_dispatcher_unknown_language_returns_empty():
    symbols = parse_file(AUTH_PY, "brainfuck")
    assert symbols == []


def test_dispatcher_returns_code_symbol_instances():
    for lang, path in [("python", AUTH_PY), ("javascript", APP_JS)]:
        symbols = parse_file(path, lang)
        for s in symbols:
            assert isinstance(s, CodeSymbol)
