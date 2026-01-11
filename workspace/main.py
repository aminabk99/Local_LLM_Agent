from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.routes.chat import router as chat_router

app = FastAPI(
    title="AI-Built FastAPI Service",
    version="1.0.1",
    description="A minimal API generated and iterated by an AI coding agent running locally with Ollama.",
)

@app.get("/")
def root():
    return {"message": "Service is running", "try": ["/health", "/docs", "/chat"]}

@app.get("/health")
def health():
    return {"status": "healthy"}

app.include_router(chat_router)
