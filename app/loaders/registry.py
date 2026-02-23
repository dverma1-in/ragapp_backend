from typing import Dict, Type

from app.loaders.base import BaseLoader
from app.loaders.pdf import PDFLoader
from app.loaders.txt import TXTLoader


LOADER_REGISTRY: Dict[str, Type[BaseLoader]] = {
    "pdf": PDFLoader,
    "txt": TXTLoader,
}