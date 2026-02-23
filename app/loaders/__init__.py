from pathlib import Path
from fastapi import UploadFile, HTTPException

from app.loaders.registry import LOADER_REGISTRY


async def extract_text(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name")

    # ".pdf" -> "pdf"
    ext = Path(file.filename).suffix.lower().lstrip(".")

    loader_class = LOADER_REGISTRY.get(ext)
    if not loader_class:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}"
        )

    loader = loader_class()
    return await loader.load(file)