from fastapi import FastAPI
from app.routes import upload_router
from app.routes import chat_router
from app.middlewares.middleware import exception_handling_middleware

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, World!"}

app.middleware("http")(exception_handling_middleware)

app.include_router(upload_router)
app.include_router(chat_router)