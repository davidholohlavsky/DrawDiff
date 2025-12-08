import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security
from dotenv import load_dotenv

from drawdiff import run_drawdiff

# --- konfigurace z .env ---
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = (BASE_DIR / os.getenv("WORK_DIR", "../work")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("API_KEY", "")  # prázdné = vypnuto (pro demo)
CORS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://127.0.0.1,http://localhost").split(",")
    if o.strip()
]

# --- FastAPI app ---
app = FastAPI(title="DrawDiff Prototype", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security: X-API-Key (zobrazí Authorize ve /docs) ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    # když není API_KEY v .env, ochranu vypneme (lokální demo)
    if not API_KEY:
        return
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Endpoints ---
@app.get("/")
def root():
    return {"ok": True, "message": "DrawDiff prototype running"}


@app.post("/drawdiff", dependencies=[Depends(verify_api_key)])
async def drawdiff(
    old: UploadFile = File(...),
    new: UploadFile = File(...),
):
    """
    Endpoint pro porovnání výkresů.

    Parametry:
      - old: původní PDF
      - new: nové PDF
      - variant: 'default' = zarovnávání os, 'fixed' = pouze položení na 3×3 plátno
    """

    job_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    old_path = job_dir / "old.pdf"
    new_path = job_dir / "new.pdf"
    old_path.write_bytes(await old.read())
    new_path.write_bytes(await new.read())

    # volání jádra s novým parametrem variant
    result = run_drawdiff(old_path, new_path, job_dir)
    result["job_id"] = job_id
    result["overlay_url"] = f"/file/overlay/{job_id}"

    return JSONResponse(content=result)


@app.get("/file/overlay/{job_id}", dependencies=[Depends(verify_api_key)])
def get_overlay(job_id: str):
    path = WORK_DIR / job_id / "overlay.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)
