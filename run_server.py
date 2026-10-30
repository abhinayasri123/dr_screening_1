import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.chdir(str(project_root))

print(f"Loading FastAPI app from {project_root}...", flush=True)
from backend.main import app
print("FastAPI app imported successfully! Starting uvicorn...", flush=True)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
