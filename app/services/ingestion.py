from app.vectorstore.client import add_documents
from fastapi import UploadFile
from app.loaders import extract_text
from app.utils import chunk_text, embed_text
import uuid

async def ingest_file(file: UploadFile):
    units = await extract_text(file)
    chunks_with_metadata = []
    for unit in units:
        unit_chunks = chunk_text(unit["text"], unit["metadata"])
        chunks_with_metadata.extend(unit_chunks)

    texts = [chunk["text"] for chunk in chunks_with_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]
    embeddings = embed_text(texts)
    ids = [str(uuid.uuid4()) for _ in texts]

    try:
      add_documents(ids, texts, embeddings, metadatas)
    except Exception as e:
      print("VECTOR DB ERROR:", e)
      raise

    return {
    "file_name": file.filename,
    "chunks_added": len(texts)
}