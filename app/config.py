import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

CHROMA_DIR = "data/chroma"
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"