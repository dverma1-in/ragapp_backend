from typing import List
from fastapi import UploadFile
from app.loaders.types import TextUnit

class BaseLoader:
    async def load(self, file: UploadFile) -> List[TextUnit]:
        raise NotImplementedError("Loader must implement the load method.")