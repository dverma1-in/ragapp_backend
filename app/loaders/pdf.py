from fastapi import UploadFile
import fitz  # PyMuPDF

from app.loaders.base import BaseLoader
from app.loaders.types import TextUnit
from typing import List


class PDFLoader(BaseLoader):
    async def load(self, file: UploadFile) -> List[TextUnit]:
        content = await file.read()
        units: List[TextUnit] = []

        with fitz.open(stream=content, filetype="pdf") as doc:
            for page_index, page in enumerate(doc):
                text = page.get_text()

                if not text.strip():
                    continue

                units.append({
                    "text": text,
                    "metadata": {
                        "file_name": file.filename,
                        "file_type": "pdf",
                        "page": page_index + 1
                    }
                })

        return units