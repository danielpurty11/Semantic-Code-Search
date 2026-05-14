from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeChunk:
    id: str                          # unique ID: repo:path:symbol:chunk_n
    repo: str
    language: str
    file_path: str
    symbol: str                      # function/class name
    symbol_type: str                 # "function" | "class" | "module"
    start_line: int
    end_line: int
    text: str                        # the text to be embedded
    docstring: Optional[str] = None
    imports: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    commit_hash: str = ""
