import chromadb
from app.config import CHROMA_DIR

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(name='documents')

def add_documents(ids, documents, embeddings, metadatas):
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

def query_documents(query_embeddings, n_results=5):
    return collection.query(
        query_embeddings=[query_embeddings], 
        n_results=n_results
        )    