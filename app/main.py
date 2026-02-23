from fastapi import FastAPI
from app.routes import upload_router

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, World!"}

app.include_router(upload_router)