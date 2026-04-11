# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Deforum Studio — a local web app for generating Deforum-style animated clips using Stable Diffusion 1.5. Targets Apple Silicon (MPS backend), float32 precision. Supports HuggingFace models and local `.safetensors`/`.ckpt` checkpoints.

## Running

```bash
./start.sh
# or: npm start
```

Starts both backend (port 8000) and frontend (port 5173) in one command. Ctrl-C kills both.

On first run, `start.sh` creates `backend/.venv`, installs Python deps, and installs frontend `node_modules`. Subsequent runs skip setup if nothing changed. Open `http://localhost:5173`.

First startup also downloads `runwayml/stable-diffusion-v1-5` and loads it into memory. Requires `ffmpeg` on PATH for video stitching. Local model files go in `backend/models/`.

**Type-check frontend:**
```bash
cd frontend && npx tsc --noEmit
```

## Architecture

Two independent processes communicating via REST:

**Backend** (`backend/`): FastAPI. Two files:
- `main.py` — API layer. Endpoints under `/api/`. Pydantic request validation. Loads model via FastAPI lifespan hook.
- `generator.py` — All generation logic. `DeforumGenerator` class owns the SD pipeline and job state. Config dataclasses: `GenerationConfig` (deforum + img2vid), `Vid2VidConfig` (vid2vid), and `PromptKeyframe` (entries in `GenerationConfig.prompt_schedule`). `Job.config` is `Union[GenerationConfig, Vid2VidConfig]`. A single long-lived worker thread consumes a FIFO queue (`deque` + `Condition`) — submits are always accepted and run sequentially. Each job gets a UUID-based output directory under `backend/outputs/` and writes a `project.json` metadata sidecar on every status transition. Supports loading local `.safetensors`/`.ckpt` files from `backend/models/` as well as HuggingFace model IDs.

**Frontend** (`frontend/src/`): React + TypeScript + CSS Modules.
- `App.tsx` — Root component. Owns generation config state, job lifecycle, and tab switching (`deforum` | `img2vid` | `vid2vid` | `queue` | `gallery`). Fetches available models from `/api/models` on mount. Generation modes use the sidebar+Preview layout; queue/gallery take over the main area.
- `components/UnifiedControls.tsx` — Single left-panel controls component that adapts to the active generation mode. Shares prompt/size/sampler fields across modes and swaps in mode-specific inputs (deforum motion sliders, file pickers for img2vid/vid2vid). Mounts `PromptSchedule` under the main prompt when `mode === "deforum"`.
- `components/PromptSchedule.tsx` — Deforum-only keyframed-prompt editor. Rows of `[frame] [prompt textarea] [blend: cut/4f/8f/16f] [×]` plus a frame ruler with pins and translucent blend bands. Exports `validatePromptSchedule` — mirrors the backend validator and gates the Generate button.
- `components/Preview.tsx` — Right panel. Progress bar during generation, video player when done, thumbnail strip. Shared across all modes and works for historical jobs (driven purely by `jobId` + `JobStatus` props).
- `components/Queue.tsx` — Running / queued / recent lists. Rows are clickable (loads the job into Preview) with a Cancel action.
- `components/Gallery.tsx` — Grid of thumbnail cards loaded from `/api/gallery`. Click to replay in Preview, hover-reveal two-click delete.
- `components/SizeSelect.tsx` — Shared width/height dropdown component.
- `hooks/useJobPolling.ts` — Tracks the "current" job, polls `/api/jobs/{id}/status` at 1 Hz. Skips polling when the initial status is terminal (used for gallery replay).
- `hooks/useGenerationActions.ts` — Submit handlers for the three generation endpoints.
- `hooks/useQueuePolling.ts` — Polls `/api/queue` while the Queue tab is active; exposes `cancelItem`.
- `hooks/useGallery.ts` — Fetches `/api/gallery` on tab open; exposes `refresh` and `deleteItem`.
- `types.ts` — Shared TypeScript interfaces (`GenerationConfig`, `Vid2VidConfig`, `JobStatus`, `QueueItem`, `QueueSnapshot`, `GalleryItem`, `ModelInfo`, `PromptKeyframe`) and defaults. These mirror the backend's Pydantic/dataclass models exactly.
- `defaults.ts` — Default positive/negative prompt strings.

**API endpoints:**
- `GET /api/models` — List available models (HuggingFace + local files)
- `POST /api/generate` — Enqueue a deforum generation job
- `POST /api/vid2vid` — Enqueue a vid2vid job (multipart: video file + JSON config)
- `POST /api/img2vid` — Enqueue an img2vid job (multipart: image file + JSON config)
- `GET /api/jobs/{job_id}/status` — Poll job progress
- `POST /api/jobs/{job_id}/cancel` — Cancel a queued or running job
- `GET /api/jobs/{job_id}/frames/{frame_number}` — Fetch individual frame PNG (filesystem-backed; works for historical jobs)
- `GET /api/jobs/{job_id}/video` — Download final MP4 (filesystem-backed)
- `GET /api/jobs/{job_id}/config` — Retrieve the config used for a job
- `GET /api/queue` — Current queue snapshot: `{ queued, running, recent }`
- `GET /api/gallery` — List past projects from per-folder `project.json` metadata, newest first
- `DELETE /api/gallery/{job_id}` — Delete a project's output folder (refuses if the job is currently running)

**Deforum pipeline** (in `generator.py`, `_run_job`):
1. Before the frame loop, `_build_prompt_schedule` pre-computes per-frame `(prompt_embeds, negative_prompt_embeds)` by encoding each unique keyframe prompt once via `pipe.encode_prompt` and cosine-easing linear interpolation between adjacent keyframes across each keyframe's `blend_frames` window. With an empty `prompt_schedule`, every frame reuses the main prompt's embeds.
2. Frame 0: txt2img from the frame-0 embeds (or img2img from uploaded image in img2vid mode)
3. Frames 1+: warp previous frame (affine: zoom/rotate/translate via OpenCV) → img2img, using that frame's slot from the schedule
4. Optional LAB-space color coherence matching against frame 0
5. After all frames: ffmpeg subprocess stitches PNGs into MP4 (H.264, CRF 18)
6. When `use_deforum` is false, each frame is independent txt2img (no warping), still driven by the schedule

**Vid2Vid pipeline** (in `generator.py`, `_run_vid2vid_job`):
1. Extract frames from uploaded video via ffmpeg (scale+pad to target size, preserving aspect ratio)
2. Each extracted frame → img2img with prompt and denoising strength
3. After all frames: ffmpeg subprocess stitches output PNGs into MP4 at extraction FPS
4. Upload accepted via `POST /api/vid2vid` as multipart form (video file + JSON config string)

All pipelines (txt2img, img2img) share the same model weights loaded once at startup. Model is swapped on-demand if a different model_id is requested.

## Key Constraints

- **MPS + float32 only**: float16 VAE decode on MPS produces NaN → black frames. The entire pipeline runs in float32.
- **Serial worker with FIFO queue**: A single worker thread processes one job at a time. Submits are always accepted and queued; POST endpoints never return 409. All modes share the same queue.
- **In-memory queue + job tracking**: The queue and `jobs` dict live in the generator instance and are lost on restart. Past projects survive via `outputs/{job_id}/project.json` and are discoverable through `/api/gallery`.
- **Per-project metadata sidecar**: `project.json` is rewritten on every status transition (QUEUED → RUNNING → DONE/ERROR/CANCELLED) and contains the config snapshot, mode, timestamps, and error message. Gallery listing scans for these files; folders without one are silently skipped (backwards compatible with pre-queue outputs).
- **Filesystem-backed frame/video lookup**: `get_frame_path` / `get_video_path` validate `job_id` against `^[a-f0-9]{12}$` and resolve directly from `outputs/`, so historical projects play back even after a restart clears in-memory state.
- **NumPy pinned to 1.26.4**: Required by kokoro dependency in the host environment.
- **Frames served individually**: Frontend fetches each frame by number as it becomes available, enabling live preview during generation.
