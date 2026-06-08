import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.github_loader import clone_repo, get_repo_id
from utils.file_loader import load_files
from dotenv import load_dotenv
from utils.vector_database import create_vector_database, load_vector_store

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo_status = {}

class ProcessRequest(BaseModel):
    repo_url: str

@app.get("/")
def reading():
    return {"message": "Hello UMAAAAAAirrrr"}

@app.post("/api/process")
async def process_repository(request: ProcessRequest,background_tasks: BackgroundTasks):
    repo_url = request.repo_url
    try:
        repo_id = get_repo_id(repo_url)
        repo_status[repo_id] = "processing"

        background_tasks.add_task(run_indexing, repo_url,repo_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def run_indexing(repo_url,repo_id):
    try:
        path = clone_repo(repo_url)
        docs = load_files(path)
        db = create_vector_database(docs)
        repo_status[repo_id] = "completed"
    except Exception as e:
        print(f"Error indexing {repo_id}: {e}")
        repo_status[repo_id] = f"failed: {str(e)}"

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)