from abc import ABC, abstractmethod
from typing import List
from fastapi import UploadFile
from app.loaders.types import TextUnit


class BaseLoader(ABC):
    @abstractmethod
    async def load(self, file: UploadFile) -> List[TextUnit]:
        ...
