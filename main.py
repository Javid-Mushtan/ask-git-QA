import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def reading():
    return {"message": "Hello UMAAAAAAirrrr"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)