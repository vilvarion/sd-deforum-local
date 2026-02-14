# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Deforum Studio — a local web app for generating Deforum-style animated clips using Stable Diffusion 1.5. Targets Apple Silicon (MPS backend), float32 precision. Supports HuggingFace models and local `.safetensors`/`.ckpt` checkpoints.

## Running

**Backend** (port 8000):
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
First startup downloads `runwayml/stable-diffusion-v1-5` and loads it into memory. Requires `ffmpeg` on PATH for video stitching. Local model files go in `backend/models/`.

**Frontend** (port 5173):
```bash
cd frontend
npm install
npm run dev
```
Vite proxies `/api/*` to `localhost:8000`. Open `http://localhost:5173`.

**Type-check frontend:**
```bash
cd frontend && npx tsc --noEmit
```

## Architecture

Two independent processes communicating via REST:

**Backend** (`backend/`): FastAPI. Two files:
- `main.py` — API layer. 9 endpoints under `/api/`. Pydantic request validation. Loads model via FastAPI lifespan hook.
- `generator.py` — All generation logic. `DeforumGenerator` class owns the SD pipeline and job state. Two config dataclasses: `GenerationConfig` (deforum + img2vid) and `Vid2VidConfig` (vid2vid). `Job.config` is `Union[GenerationConfig, Vid2VidConfig]`. Jobs run in background threads, one at a time (mutex-gated). Each job gets a UUID-based output directory under `backend/outputs/`. Supports loading local `.safetensors`/`.ckpt` files from `backend/models/` as well as HuggingFace model IDs.

**Frontend** (`frontend/src/`): React + TypeScript + CSS Modules.
- `App.tsx` — Root component. Owns generation config state, job lifecycle, tab switching (`deforum` | `vid2vid` | `img2vid`), and 1-second polling loop. Fetches available models from `/api/models` on mount.
- `components/Controls.tsx` — Left panel for Deforum mode. All deforum config inputs. `SliderInput` helper for slider+number combos.
- `components/Vid2VidControls.tsx` — Left panel for Vid2Vid mode. File upload, prompt, img2img settings, extraction FPS. Reuses `Controls.module.css`.
- `components/Img2VidControls.tsx` — Left panel for Img2Vid mode. Image upload, deforum motion settings applied starting from an uploaded image.
- `components/SizeSelect.tsx` — Shared width/height dropdown component (256/384/512/768).
- `components/Preview.tsx` — Right panel. Progress bar during generation, video player when done, thumbnail strip. Shared across all modes.
- `types.ts` — Shared TypeScript interfaces (`GenerationConfig`, `Vid2VidConfig`, `JobStatus`, `ModelInfo`) and defaults. These mirror the backend's Pydantic/dataclass models exactly.
- `defaults.ts` — Default positive/negative prompt strings.

**API endpoints:**
- `GET /api/models` — List available models (HuggingFace + local files)
- `POST /api/generate` — Start a deforum generation job
- `POST /api/vid2vid` — Start a vid2vid job (multipart: video file + JSON config)
- `POST /api/img2vid` — Start an img2vid job (multipart: image file + JSON config)
- `GET /api/jobs/{job_id}/status` — Poll job progress
- `POST /api/jobs/{job_id}/cancel` — Cancel a running job
- `GET /api/jobs/{job_id}/frames/{frame_number}` — Fetch individual frame PNG
- `GET /api/jobs/{job_id}/video` — Download final MP4
- `GET /api/jobs/{job_id}/config` — Retrieve the config used for a job

**Deforum pipeline** (in `generator.py`, `_run_job`):
1. Frame 0: txt2img from prompt (or img2img from uploaded image in img2vid mode)
2. Frames 1+: warp previous frame (affine: zoom/rotate/translate via OpenCV) → img2img
3. Optional LAB-space color coherence matching against frame 0
4. After all frames: ffmpeg subprocess stitches PNGs into MP4 (H.264, CRF 18)
5. When `use_deforum` is false, each frame is independent txt2img (no warping)

**Vid2Vid pipeline** (in `generator.py`, `_run_vid2vid_job`):
1. Extract frames from uploaded video via ffmpeg (scale+pad to target size, preserving aspect ratio)
2. Each extracted frame → img2img with prompt and denoising strength
3. After all frames: ffmpeg subprocess stitches output PNGs into MP4 at extraction FPS
4. Upload accepted via `POST /api/vid2vid` as multipart form (video file + JSON config string)

All pipelines (txt2img, img2img) share the same model weights loaded once at startup. Model is swapped on-demand if a different model_id is requested.

## Key Constraints

- **MPS + float32 only**: float16 VAE decode on MPS produces NaN → black frames. The entire pipeline runs in float32.
- **Single job at a time**: POST endpoints return 409 if busy. No job queue. All modes share the same mutex.
- **In-memory job tracking**: Jobs dict lives in the generator instance. No persistence across restarts.
- **NumPy pinned to 1.26.4**: Required by kokoro dependency in the host environment.
- **Frames served individually**: Frontend fetches each frame by number as it becomes available, enabling live preview during generation.
