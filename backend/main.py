import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from generator import DeforumGenerator, GenerationConfig, PromptKeyframe, Vid2VidConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

generator = DeforumGenerator(
    output_root=os.path.join(os.path.dirname(__file__), "outputs"),
    models_dir=os.path.join(os.path.dirname(__file__), "models"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Stable Diffusion model...")
    generator.load_model()
    logger.info("Model ready.")
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


class PromptKeyframeModel(BaseModel):
    frame: int = Field(ge=1, le=1000)
    prompt: str
    blend_frames: int = Field(default=8, ge=0, le=120)


class GenerateRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    num_frames: int = Field(default=15, ge=2, le=120)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    denoising_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    steps: int = Field(default=25, ge=1, le=150)
    zoom_per_frame: float = Field(default=1.02, ge=0.5, le=2.0)
    rotate_per_frame: float = Field(default=0.0, ge=-45.0, le=45.0)
    translate_x: float = Field(default=0.0, ge=-100.0, le=100.0)
    translate_y: float = Field(default=0.0, ge=-100.0, le=100.0)
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295)
    fps: int = Field(default=12, ge=1, le=60)
    use_deforum: bool = True
    model_id: str = "runwayml/stable-diffusion-v1-5"
    prompt_schedule: list[PromptKeyframeModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_schedule(self):
        if not self.prompt_schedule:
            return self
        seen_frames = set()
        last_frame = 0
        for kf in sorted(self.prompt_schedule, key=lambda k: k.frame):
            if kf.frame in seen_frames:
                raise ValueError(f"duplicate keyframe at frame {kf.frame}")
            if kf.frame >= self.num_frames:
                raise ValueError(
                    f"keyframe frame {kf.frame} must be < num_frames ({self.num_frames})"
                )
            if not kf.prompt.strip():
                raise ValueError(f"keyframe at frame {kf.frame} has empty prompt")
            max_blend = kf.frame - last_frame
            if kf.blend_frames > max_blend:
                raise ValueError(
                    f"keyframe at frame {kf.frame}: blend_frames ({kf.blend_frames}) "
                    f"exceeds gap to previous keyframe ({max_blend})"
                )
            seen_frames.add(kf.frame)
            last_frame = kf.frame
        return self


def _to_generation_config(req: GenerateRequest) -> GenerationConfig:
    data = req.model_dump()
    schedule = [PromptKeyframe(**kf) for kf in data.pop("prompt_schedule", [])]
    return GenerationConfig(**data, prompt_schedule=schedule)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    config = _to_generation_config(req)
    job_id = generator.submit_job(config)
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
        "current_step": job.current_step,
        "total_steps": job.total_steps,
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
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    denoising_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    steps: int = Field(default=25, ge=1, le=150)
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295)
    extraction_fps: int = Field(default=12, ge=1, le=60)
    model_id: str = "runwayml/stable-diffusion-v1-5"


@app.post("/api/vid2vid")
async def vid2vid(video: UploadFile = File(...), config_json: str = Form(...)):
    try:
        parsed = json.loads(config_json)
        req = Vid2VidRequest(**parsed)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
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
    return {"job_id": job_id}


@app.post("/api/img2vid")
async def img2vid(image: UploadFile = File(...), config_json: str = Form(...)):
    try:
        parsed = json.loads(config_json)
        req = GenerateRequest(**parsed)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")

    config = _to_generation_config(req)

    suffix = os.path.splitext(image.filename or "image.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        job_id = generator.submit_img2vid_job(config, tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/config")
def job_config(job_id: str):
    job = generator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job.config)


@app.get("/api/queue")
def queue_snapshot():
    return generator.list_queue_snapshot()


@app.get("/api/gallery")
def gallery_list():
    return generator.list_gallery()


@app.delete("/api/gallery/{job_id}")
def gallery_delete(job_id: str):
    if not generator.delete_project(job_id):
        raise HTTPException(status_code=404, detail="Project not found or still running")
    return {"status": "deleted"}
