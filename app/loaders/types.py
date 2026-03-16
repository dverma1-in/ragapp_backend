from typing import TypedDict, Dict, Any


class TextUnit(TypedDict):
    """A single page / section extracted from a source file."""
    text: str
    metadata: Dict[str, Any]
