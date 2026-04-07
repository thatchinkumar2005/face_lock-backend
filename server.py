"""
Face-lock backend — FastAPI
Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from face_engine import FaceEngine

engine: FaceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Loading FaceNet model…")
    engine = FaceEngine()
    print(f"Model ready. Trained users: {engine.list_users()}")
    yield


app = FastAPI(title="Face Lock API", lifespan=lifespan)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        data = upload.file.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}")


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #


@app.get("/")
def health():
    return {"status": "ok", "trained_users": engine.list_users()}


@app.post("/verify")
async def verify(photo: UploadFile = File(...)):
    """
    Send a single photo. Returns:
      { status: "open"|"denied", user: str|null, confidence: float, message: str }
    """
    image = _read_image(photo)
    result = engine.verify(image)
    return JSONResponse(content=result)


@app.post("/train")
async def train(
    user_id: str = Form(...),
    photos: list[UploadFile] = File(...),
):
    """
    Send user_id + 2-5 photos to train (or retrain) a user.
    Returns:
      { success: bool, photos_used: int, message: str }
    """
    if not user_id.strip():
        raise HTTPException(status_code=422, detail="user_id must not be empty.")
    if len(photos) < 1:
        raise HTTPException(status_code=422, detail="At least 1 photo required.")

    images = [_read_image(p) for p in photos]
    result = engine.train(user_id.strip(), images)
    status_code = 200 if result["success"] else 422
    return JSONResponse(content=result, status_code=status_code)


@app.get("/users")
def list_users():
    """Return all trained user IDs."""
    return {"users": engine.list_users()}


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    """Remove a trained user."""
    deleted = engine.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    return {"message": f"User '{user_id}' deleted."}
