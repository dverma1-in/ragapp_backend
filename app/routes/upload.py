from app.loaders import extract_text
from app.utils.chunk_text import chunk_text
from app.utils.embed_text import embed_text
from fastapi import APIRouter, UploadFile, File
from app.services import ingest_file

#router = APIRouter()
router = APIRouter(prefix="/upload", tags=['upload'])

@router.post("")
async def upload_file(file: UploadFile = File(...)):
    try:
      await ingest_file(file)
      return {"status": "success"}
    except Exception as e:
      return {"error": str(e)}