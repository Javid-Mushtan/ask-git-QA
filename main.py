import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.github_loader import clone_repo
app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    repo_url: str

@app.get("/")
def reading():
    return {"message": "Hello UMAAAAAAirrrr"}

@app.post("/api/process")
def process(request: ProcessRequest):
    print(request.repo_url)
    clone_repo(request.repo_url)
    return {"message": "Repository cloned successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)