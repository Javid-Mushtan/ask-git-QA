from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "upload"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K = 6
SEARCH_TYPE = "cosine similarity"

EMBEDDING_MODEL = "text-embedding-3-small"
LANGUAGE_MODEL = "openrouter/free"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2
EMBEDDING_BATCH_SIZE = 100

EXCLUDED_DIRS = {
    ",git","node_modules","__pycache__","dist","build","venv",
    ".venv","env",".env_vars","target","bin","obj",".idea",".vscode"
}

EXCLUDED_EXTENSIONS = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    # Media
    ".mp3", ".mp4", ".mov", ".avi", ".wav",
    # Archives
    ".zip", ".tar", ".gz", ".7z", ".rar",
    # Binaries/Compiled
    ".exe", ".dll", ".so", ".bin", ".pyc", ".pyd", ".o", ".a", ".lib",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Data/Large
    ".csv", ".sqlite", ".db", ".parquet", ".pickle",
}