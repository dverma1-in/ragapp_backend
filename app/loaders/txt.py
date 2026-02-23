from fastapi import UploadFile
from typing import List

from app.loaders.base import BaseLoader
from app.loaders.types import TextUnit


class TXTLoader(BaseLoader):
    async def load(self, file: UploadFile) -> List[TextUnit]:
        content = await file.read()

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        if not text.strip():
            return []

        return [
            {
                "text": text,
                "metadata": {
                    "file_name": file.filename,
                    "file_type": "txt"
                }
            }
        ]