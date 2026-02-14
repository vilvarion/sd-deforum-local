import json
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from generator import DeforumGenerator, GenerationConfig, Vid2VidConfig, JobStatus

generator = DeforumGenerator(
    output_root=os.path.join(os.path.dirname(__file__), "outputs"),
    models_dir=os.path.join(os.path.dirname(__file__), "models"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Stable Diffusion model...")
    generator.load_model()
    print("Model ready.")
    yield


app = FastAPI(title="Deforum Generator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
def list_models():
    return generator.list_models()


class GenerateRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    num_frames: int = 15
    width: int = 512
    height: int = 512
    denoising_strength: float = 0.55
    guidance_scale: float = 7.5
    steps: int = 25
    zoom_per_frame: float = 1.02
    rotate_per_frame: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    seed: Optional[int] = None
    fps: int = 12
    color_coherence: bool = True
    model_id: str = "runwayml/stable-diffusion-v1-5"


@app.post("/api/generate")
def generate(req: GenerateRequest):
    config = GenerationConfig(**req.model_dump())
    job_id = generator.submit_job(config)
    if job_id is None:
        raise HTTPException(status_code=409, detail="A job is already running")
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str):
    job = generator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job.status.value,
        "current_frame": job.current_frame,
        "total_frames": job.total_frames,
        "error_message": job.error_message,
    }


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not generator.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found or not running")
    return {"status": "cancelling"}


@app.get("/api/jobs/{job_id}/frames/{frame_number}")
def job_frame(job_id: str, frame_number: int):
    path = generator.get_frame_path(job_id, frame_number)
    if not path:
        raise HTTPException(status_code=404, detail="Frame not available yet")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    path = generator.get_video_path(job_id)
    if not path:
        raise HTTPException(status_code=404, detail="Video not available")
    return FileResponse(path, media_type="video/mp4", filename="deforum.mp4")


class Vid2VidRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    denoising_strength: float = 0.55
    guidance_scale: float = 7.5
    steps: int = 25
    seed: Optional[int] = None
    extraction_fps: int = 12
    model_id: str = "runwayml/stable-diffusion-v1-5"


@app.post("/api/vid2vid")
async def vid2vid(video: UploadFile = File(...), config_json: str = Form(...)):
    try:
        parsed = json.loads(config_json)
        req = Vid2VidRequest(**parsed)
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")

    config = Vid2VidConfig(**req.model_dump())

    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        job_id = generator.submit_vid2vid_job(config, tmp_path)
    finally:
        os.unlink(tmp_path)

    if job_id is None:
        raise HTTPException(status_code=409, detail="A job is already running")
    return {"job_id": job_id}


@app.post("/api/img2vid")
async def img2vid(image: UploadFile = File(...), config_json: str = Form(...)):
    try:
        parsed = json.loads(config_json)
        req = GenerateRequest(**parsed)
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")

    config = GenerationConfig(**req.model_dump())

    suffix = os.path.splitext(image.filename or "image.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        job_id = generator.submit_img2vid_job(config, tmp_path)
    finally:
        os.unlink(tmp_path)

    if job_id is None:
        raise HTTPException(status_code=409, detail="A job is already running")
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/config")
def job_config(job_id: str):
    job = generator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job.config)
