# Deforum Studio

A local web app for generating Deforum-style animated clips using Stable Diffusion 1.5. 
Built for Apple Silicon Macs (MPS backend).

## Features

- **Deforum mode** — Generate animated sequences with camera motion (zoom, rotate, translate) and frame-to-frame coherence
- **Vid2Vid mode** — Restyle existing videos frame-by-frame using img2img
- **Img2Vid mode** — Start from an uploaded image and apply Deforum-style animation
- **Live preview** — Watch frames appear in real-time as they generate
- **Model support** — Use the default SD 1.5 weights from HuggingFace or drop local `.safetensors`/`.ckpt` checkpoints into `backend/models/`

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4) — runs on the MPS backend in float32
- **Python 3.10+**
- **Node.js 18+**
- **ffmpeg** — must be available on PATH (used for video stitching)

## Installation

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The first startup will download the `runwayml/stable-diffusion-v1-5` model (~4 GB).

To use custom models, place `.safetensors` or `.ckpt` files in `backend/models/`. They will appear in the model dropdown.

### Frontend

```bash
cd frontend
npm install
```

## Running

Start both the backend and frontend in separate terminals:

```bash
# Terminal 1 — Backend (port 8000)
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend (port 5173)
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## Usage

### Deforum

Generate animated clips from a text prompt. Frame 0 is created via txt2img, then each subsequent frame warps the previous one and runs img2img. Configure zoom, rotation, translation, denoising strength, and frame count. Enable color coherence to keep colors consistent across frames.

### Vid2Vid

Upload a video and restyle every frame with a text prompt. The video is extracted at a configurable FPS, each frame goes through img2img, and the results are stitched back into an MP4.

### Img2Vid

Upload a starting image and apply Deforum-style animation on top of it. Same motion controls as Deforum mode, but begins from your image instead of generating the first frame from scratch.

## Project Structure

```
backend/
  main.py           # FastAPI server, API endpoints
  generator.py       # Stable Diffusion pipeline, job management
  models/            # Drop custom .safetensors/.ckpt files here
  outputs/           # Generated frames and videos (per-job UUID dirs)
  requirements.txt

frontend/
  src/
    App.tsx          # Root component, job lifecycle, tab switching
    types.ts         # TypeScript interfaces mirroring backend models
    defaults.ts      # Default prompts
    components/
      Controls.tsx        # Deforum settings panel
      Vid2VidControls.tsx # Vid2Vid settings panel
      Img2VidControls.tsx # Img2Vid settings panel
      SizeSelect.tsx      # Shared width/height selector
      Preview.tsx         # Preview panel, progress bar, video player
```

## API

All endpoints are under `/api/`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List available models |
| POST | `/api/generate` | Start a deforum job (JSON body) |
| POST | `/api/vid2vid` | Start a vid2vid job (multipart form) |
| POST | `/api/img2vid` | Start an img2vid job (multipart form) |
| GET | `/api/jobs/{id}/status` | Poll job progress |
| POST | `/api/jobs/{id}/cancel` | Cancel a running job |
| GET | `/api/jobs/{id}/frames/{n}` | Get frame N as PNG |
| GET | `/api/jobs/{id}/video` | Download final MP4 |
| GET | `/api/jobs/{id}/config` | Get job config |

Only one job runs at a time. Submitting a new job while one is running returns `409 Conflict`.
