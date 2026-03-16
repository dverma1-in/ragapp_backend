from fastapi import APIRouter, UploadFile, File
from app.services import ingest_file
from app.schemas import UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    result = await ingest_file(file)
    return UploadResponse(**result)
