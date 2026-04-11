import json
import logging
import os
import re
import threading
import time
import uuid
import subprocess
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union

logger = logging.getLogger(__name__)

import glob as globmod
import shutil
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, EulerDiscreteScheduler


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"


@dataclass
class GenerationConfig:
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
    use_deforum: bool = True
    model_id: str = DEFAULT_MODEL_ID


@dataclass
class Vid2VidConfig:
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    denoising_strength: float = 0.55
    guidance_scale: float = 7.5
    steps: int = 25
    seed: Optional[int] = None
    extraction_fps: int = 12
    model_id: str = DEFAULT_MODEL_ID


@dataclass
class Job:
    job_id: str
    config: Union[GenerationConfig, Vid2VidConfig]
    mode: str = "deforum"  # deforum | img2vid | vid2vid
    status: JobStatus = JobStatus.QUEUED
    current_frame: int = 0
    total_frames: int = 0
    current_step: int = 0
    total_steps: int = 0
    error_message: str = ""
    output_dir: str = ""
    created_at: str = ""
    _cancel_requested: bool = field(default=False, repr=False)
    _init_image_path: Optional[str] = field(default=None, repr=False)
    _video_path: Optional[str] = field(default=None, repr=False)


JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class DeforumGenerator:
    def __init__(self, output_root: str = "outputs", models_dir: str = "models"):
        self.output_root = output_root
        self.models_dir = models_dir
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue_cond = threading.Condition(self._lock)
        self._queue: deque[str] = deque()
        self._current_job_id: Optional[str] = None
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._txt2img_pipe: Optional[StableDiffusionPipeline] = None
        self._img2img_pipe: Optional[StableDiffusionImg2ImgPipeline] = None
        self._current_model_id: Optional[str] = None
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Use float32 on MPS — float16 VAE decode produces NaN (black frames)
        # and mixed precision causes dtype mismatches. SD 1.5 fits in 16GB at float32.
        self._dtype = torch.float32

    def list_models(self) -> list[dict[str, str]]:
        models = [{"id": DEFAULT_MODEL_ID, "name": "SD 1.5 (default)"}]
        for fname in sorted(os.listdir(self.models_dir)):
            if fname.endswith((".safetensors", ".ckpt")):
                models.append({"id": fname, "name": fname})
        return models

    def _free_pipelines(self):
        if self._txt2img_pipe is not None:
            del self._txt2img_pipe, self._img2img_pipe
            self._txt2img_pipe = None
            self._img2img_pipe = None
            self._current_model_id = None
            if self._device == "mps":
                torch.mps.empty_cache()

    def _load_single_file(self, local_path: str) -> StableDiffusionPipeline:
        """Load a .safetensors/.ckpt file, handling linear-projection models."""
        try:
            return StableDiffusionPipeline.from_single_file(
                local_path,
                torch_dtype=self._dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except Exception as e:
            if "expected shape" not in str(e):
                raise
            # Model uses linear attention projections — provide original config
            logger.info("Retrying with linear-projection config for %s", local_path)
            return StableDiffusionPipeline.from_single_file(
                local_path,
                torch_dtype=self._dtype,
                safety_checker=None,
                requires_safety_checker=False,
                original_config=self._v1_linear_config(),
            )

    @staticmethod
    def _v1_linear_config() -> str:
        """Write a v1-inference yaml with use_linear_in_transformer and return its path."""
        yaml_content = (
            "model:\n"
            "  base_learning_rate: 1.0e-04\n"
            "  target: ldm.models.diffusion.ddpm.LatentDiffusion\n"
            "  params:\n"
            "    linear_start: 0.00085\n"
            "    linear_end: 0.0120\n"
            "    num_timesteps_cond: 1\n"
            "    log_every_t: 200\n"
            "    timesteps: 1000\n"
            "    first_stage_key: jpg\n"
            "    image_size: 64\n"
            "    channels: 4\n"
            "    monitor: val/loss_simple_ema\n"
            "    scale_factor: 0.18215\n"
            "    cond_stage_config:\n"
            "      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder\n"
            "    first_stage_config:\n"
            "      target: ldm.models.autoencoder.AutoencoderKL\n"
            "      params:\n"
            "        embed_dim: 4\n"
            "        ddconfig:\n"
            "          double_z: true\n"
            "          z_channels: 4\n"
            "          resolution: 256\n"
            "          in_channels: 3\n"
            "          out_ch: 3\n"
            "          ch: 128\n"
            "          ch_mult: [1, 2, 4, 4]\n"
            "          num_res_blocks: 2\n"
            "          attn_resolutions: []\n"
            "    unet_config:\n"
            "      target: ldm.modules.diffusionmodules.openaimodel.UNetModel\n"
            "      params:\n"
            "        image_size: 32\n"
            "        in_channels: 4\n"
            "        out_channels: 4\n"
            "        model_channels: 320\n"
            "        attention_resolutions: [4, 2, 1]\n"
            "        num_res_blocks: 2\n"
            "        channel_mult: [1, 2, 4, 4]\n"
            "        num_heads: 8\n"
            "        use_spatial_transformer: true\n"
            "        transformer_depth: 1\n"
            "        context_dim: 768\n"
            "        use_checkpoint: true\n"
            "        legacy: false\n"
            "        use_linear_in_transformer: true\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        tmp.write(yaml_content)
        tmp.close()
        return tmp.name

    def load_model(self, model_id: str = DEFAULT_MODEL_ID):
        if model_id == self._current_model_id:
            return

        self._free_pipelines()

        local_path = os.path.join(self.models_dir, model_id)
        is_local = os.path.isfile(local_path)

        if is_local:
            self._txt2img_pipe = self._load_single_file(local_path).to(self._device)
        else:
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            self._txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=scheduler,
                torch_dtype=self._dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self._device)

        self._img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self._txt2img_pipe.vae,
            text_encoder=self._txt2img_pipe.text_encoder,
            tokenizer=self._txt2img_pipe.tokenizer,
            unet=self._txt2img_pipe.unet,
            scheduler=self._txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(self._device)

        self._txt2img_pipe.enable_attention_slicing()
        self._img2img_pipe.enable_attention_slicing()
        self._current_model_id = model_id
        logger.info("Model '%s' loaded on %s with dtype %s", model_id, self._device, self._dtype)

    @staticmethod
    def _make_step_callback(job: "Job"):
        def callback(pipeline, step: int, timestep: int, callback_kwargs: dict) -> dict:
            job.current_step = step + 1
            return callback_kwargs
        return callback

    def _new_job_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _enqueue(self, job: Job) -> None:
        with self._queue_cond:
            self.jobs[job.job_id] = job
            self._queue.append(job.job_id)
            self._queue_cond.notify_all()

    def submit_job(self, config: GenerationConfig) -> str:
        job_id = self._new_job_id()
        output_dir = os.path.join(self.output_root, job_id)
        os.makedirs(output_dir, exist_ok=True)
        job = Job(
            job_id=job_id,
            config=config,
            mode="deforum",
            status=JobStatus.QUEUED,
            total_frames=config.num_frames,
            output_dir=output_dir,
            created_at=_utcnow_iso(),
        )
        self._write_project_json(job)
        self._enqueue(job)
        return job_id

    def submit_img2vid_job(self, config: GenerationConfig, image_path: str) -> str:
        job_id = self._new_job_id()
        output_dir = os.path.join(self.output_root, job_id)
        os.makedirs(output_dir, exist_ok=True)

        dest_image = os.path.join(output_dir, "input_image" + os.path.splitext(image_path)[1])
        shutil.copy2(image_path, dest_image)

        job = Job(
            job_id=job_id,
            config=config,
            mode="img2vid",
            status=JobStatus.QUEUED,
            total_frames=config.num_frames,
            output_dir=output_dir,
            created_at=_utcnow_iso(),
            _init_image_path=dest_image,
        )
        self._write_project_json(job)
        self._enqueue(job)
        return job_id

    def submit_vid2vid_job(self, config: Vid2VidConfig, video_path: str) -> str:
        job_id = self._new_job_id()
        output_dir = os.path.join(self.output_root, job_id)
        os.makedirs(output_dir, exist_ok=True)

        dest_video = os.path.join(output_dir, "input_video" + os.path.splitext(video_path)[1])
        shutil.copy2(video_path, dest_video)

        job = Job(
            job_id=job_id,
            config=config,
            mode="vid2vid",
            status=JobStatus.QUEUED,
            total_frames=0,
            output_dir=output_dir,
            created_at=_utcnow_iso(),
            _video_path=dest_video,
        )
        self._write_project_json(job)
        self._enqueue(job)
        return job_id

    def _worker_loop(self) -> None:
        while True:
            with self._queue_cond:
                while not self._queue:
                    self._queue_cond.wait()
                job_id = self._queue.popleft()
                job = self.jobs.get(job_id)
                if job is None:
                    continue
                if job._cancel_requested:
                    job.status = JobStatus.CANCELLED
                    self._write_project_json(job)
                    continue
                self._current_job_id = job_id

            try:
                if job.mode == "vid2vid":
                    self._run_vid2vid_job(job, job._video_path or "")
                else:
                    self._run_job(job, init_image_path=job._init_image_path)
            except Exception:
                logger.exception("Worker dispatch failed for job %s", job_id)
            finally:
                with self._queue_cond:
                    self._current_job_id = None

    def cancel_job(self, job_id: str) -> bool:
        with self._queue_cond:
            job = self.jobs.get(job_id)
            if not job or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
                return False
            job._cancel_requested = True
            if job.status == JobStatus.QUEUED:
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                job.status = JobStatus.CANCELLED
                self._write_project_json(job)
        return True

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def _project_dir(self, job_id: str) -> Optional[str]:
        if not JOB_ID_RE.match(job_id):
            return None
        path = os.path.join(self.output_root, job_id)
        return path if os.path.isdir(path) else None

    def get_frame_path(self, job_id: str, frame_number: int) -> Optional[str]:
        project = self._project_dir(job_id)
        if not project:
            return None
        path = os.path.join(project, f"frame_{frame_number:04d}.png")
        return path if os.path.exists(path) else None

    def get_video_path(self, job_id: str) -> Optional[str]:
        project = self._project_dir(job_id)
        if not project:
            return None
        path = os.path.join(project, "output.mp4")
        return path if os.path.exists(path) else None

    def _write_project_json(self, job: Job) -> None:
        try:
            data = {
                "job_id": job.job_id,
                "mode": job.mode,
                "status": job.status.value,
                "created_at": job.created_at or _utcnow_iso(),
                "config": asdict(job.config),
                "total_frames": job.total_frames,
                "has_video": os.path.exists(os.path.join(job.output_dir, "output.mp4")),
                "error_message": job.error_message,
            }
            with open(os.path.join(job.output_dir, "project.json"), "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.exception("Failed to write project.json for %s", job.job_id)

    def list_queue_snapshot(self) -> dict:
        def item(j: Job) -> dict:
            prompt = getattr(j.config, "prompt", "") or ""
            return {
                "job_id": j.job_id,
                "mode": j.mode,
                "status": j.status.value,
                "prompt": prompt,
                "created_at": j.created_at,
                "current_frame": j.current_frame,
                "total_frames": j.total_frames,
            }

        with self._queue_cond:
            queued = [item(self.jobs[jid]) for jid in self._queue if jid in self.jobs]
            running_id = self._current_job_id
            running = item(self.jobs[running_id]) if running_id and running_id in self.jobs else None
            recent = [
                item(j)
                for j in self.jobs.values()
                if j.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)
            ]
        recent.sort(key=lambda x: x["created_at"], reverse=True)
        return {"queued": queued, "running": running, "recent": recent[:10]}

    def list_gallery(self) -> list[dict]:
        items: list[dict] = []
        if not os.path.isdir(self.output_root):
            return items
        for entry in os.listdir(self.output_root):
            if not JOB_ID_RE.match(entry):
                continue
            meta_path = os.path.join(self.output_root, entry, "project.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                continue
            cfg = meta.get("config", {}) or {}
            width = int(cfg.get("width", 0) or 0)
            height = int(cfg.get("height", 0) or 0)
            num_frames = int(cfg.get("num_frames") or meta.get("total_frames") or 0)
            has_video = bool(meta.get("has_video")) and os.path.exists(
                os.path.join(self.output_root, entry, "output.mp4")
            )
            thumb_frame = self._find_thumbnail_frame(entry)
            items.append({
                "job_id": entry,
                "mode": meta.get("mode", "deforum"),
                "status": meta.get("status", "done"),
                "created_at": meta.get("created_at", ""),
                "prompt": (cfg.get("prompt") or "")[:200],
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "has_video": has_video,
                "thumbnail_frame": thumb_frame,
            })
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items

    def _find_thumbnail_frame(self, job_id: str) -> Optional[int]:
        project = os.path.join(self.output_root, job_id)
        frames = sorted(globmod.glob(os.path.join(project, "frame_*.png")))
        if not frames:
            return None
        # pick the middle frame for a more representative thumbnail
        mid = frames[len(frames) // 2]
        name = os.path.basename(mid)
        try:
            return int(name[len("frame_"):-len(".png")])
        except ValueError:
            return 0

    def delete_project(self, job_id: str) -> bool:
        project = self._project_dir(job_id)
        if not project:
            return False
        with self._queue_cond:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.RUNNING:
                return False
            if job and job.status == JobStatus.QUEUED:
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
            self.jobs.pop(job_id, None)
        shutil.rmtree(project, ignore_errors=True)
        return True

    @staticmethod
    def _cover_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Resize to cover target size, then center-crop to exact dimensions."""
        src_w, src_h = image.size
        scale = max(target_w / src_w, target_h / src_h)
        new_w = round(src_w * scale)
        new_h = round(src_h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return image.crop((left, top, left + target_w, top + target_h))

    def _apply_warp(self, image: np.ndarray, config: GenerationConfig) -> np.ndarray:
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2

        zoom = config.zoom_per_frame
        angle = config.rotate_per_frame
        tx = config.translate_x
        ty = config.translate_y

        # Build affine: translate to center, scale+rotate, translate back + pan
        M = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
        M[0, 2] += tx
        M[1, 2] += ty

        warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return warped

    def _match_color(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Simple color coherence via mean/std matching in LAB space."""
        src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float64)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float64)

        for ch in range(3):
            s_mean, s_std = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std() + 1e-6
            t_mean, t_std = tgt_lab[:, :, ch].mean(), tgt_lab[:, :, ch].std() + 1e-6
            src_lab[:, :, ch] = (src_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean

        src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)

    def _run_job(self, job: Job, init_image_path: Optional[str] = None):
        try:
            job.status = JobStatus.RUNNING
            self._write_project_json(job)
            config = job.config
            self.load_model(config.model_id)
            seed = config.seed if config.seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
            config.seed = seed
            generator = torch.Generator(device="cpu").manual_seed(seed)
            reference_frame = None

            # Pre-load source image for img2vid modes
            source_image = None
            if init_image_path is not None:
                source_image = Image.open(init_image_path).convert("RGB")
                source_image = self._cover_crop(source_image, config.width, config.height)

            for frame_idx in range(config.num_frames):
                if job._cancel_requested:
                    job.status = JobStatus.CANCELLED
                    self._write_project_json(job)
                    return
                job.current_frame = frame_idx

                img2img_steps = max(1, round(config.denoising_strength * config.steps))
                step_cb = self._make_step_callback(job)

                if init_image_path is not None and not config.use_deforum:
                    # Non-deforum img2vid: every frame uses the source image as base
                    job.current_step = 0
                    job.total_steps = img2img_steps
                    result = self._img2img_pipe(
                        prompt=config.prompt,
                        negative_prompt=config.negative_prompt or None,
                        image=source_image,
                        strength=config.denoising_strength,
                        num_inference_steps=config.steps,
                        guidance_scale=config.guidance_scale,
                        generator=generator,
                        callback_on_step_end=step_cb,
                    )
                    image = result.images[0]
                elif frame_idx == 0:
                    if source_image is not None:
                        job.current_step = 0
                        job.total_steps = img2img_steps
                        result = self._img2img_pipe(
                            prompt=config.prompt,
                            negative_prompt=config.negative_prompt or None,
                            image=source_image,
                            strength=config.denoising_strength,
                            num_inference_steps=config.steps,
                            guidance_scale=config.guidance_scale,
                            generator=generator,
                            callback_on_step_end=step_cb,
                        )
                        image = result.images[0]
                    else:
                        job.current_step = 0
                        job.total_steps = config.steps
                        result = self._txt2img_pipe(
                            prompt=config.prompt,
                            negative_prompt=config.negative_prompt or None,
                            width=config.width,
                            height=config.height,
                            num_inference_steps=config.steps,
                            guidance_scale=config.guidance_scale,
                            generator=generator,
                            callback_on_step_end=step_cb,
                        )
                        image = result.images[0]
                else:
                    prev_np = np.array(prev_image)
                    warped_np = self._apply_warp(prev_np, config)

                    # Mild unsharp mask to counteract accumulated warp blur (only when zooming)
                    if config.zoom_per_frame != 1.0:
                        _blur = cv2.GaussianBlur(warped_np, (0, 0), sigmaX=1.0)
                        warped_np = cv2.addWeighted(warped_np, 1.15, _blur, -0.15, 0)
                        warped_np = np.clip(warped_np, 0, 255).astype(np.uint8)

                    if config.color_coherence and reference_frame is not None:
                        warped_np = self._match_color(warped_np, reference_frame)

                    warped_pil = Image.fromarray(warped_np)

                    job.current_step = 0
                    job.total_steps = img2img_steps
                    result = self._img2img_pipe(
                        prompt=config.prompt,
                        negative_prompt=config.negative_prompt or None,
                        image=warped_pil,
                        strength=config.denoising_strength,
                        num_inference_steps=config.steps,
                        guidance_scale=config.guidance_scale,
                        generator=generator,
                        callback_on_step_end=step_cb,
                    )
                    image = result.images[0]

                frame_path = os.path.join(job.output_dir, f"frame_{frame_idx:04d}.png")
                image.save(frame_path)

                if frame_idx == 0:
                    reference_frame = np.array(image)
                prev_image = image

            job.current_frame = config.num_frames

            # Stitch into MP4
            frame_pattern = os.path.join(job.output_dir, "frame_%04d.png")
            video_path = os.path.join(job.output_dir, "output.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(config.fps),
                    "-i", frame_pattern,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    video_path,
                ],
                capture_output=True,
                check=True,
            )

            job.status = JobStatus.DONE
            self._write_project_json(job)

        except Exception as e:
            job.status = JobStatus.ERROR
            job.error_message = str(e)
            logger.exception("Job %s failed", job.job_id)
            self._write_project_json(job)

    def _extract_frames(self, video_path: str, output_dir: str, fps: int, width: int, height: int) -> int:
        extracted_dir = os.path.join(output_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                os.path.join(extracted_dir, "frame_%04d.png"),
            ],
            capture_output=True,
            check=True,
        )
        count = len(globmod.glob(os.path.join(extracted_dir, "frame_*.png")))
        if count == 0:
            raise RuntimeError("No frames extracted from video")
        return count

    def _run_vid2vid_job(self, job: Job, video_path: str):
        try:
            job.status = JobStatus.RUNNING
            self._write_project_json(job)
            config: Vid2VidConfig = job.config  # type: ignore[assignment]
            self.load_model(config.model_id)

            frame_count = self._extract_frames(
                video_path, job.output_dir, config.extraction_fps, config.width, config.height
            )
            job.total_frames = frame_count

            seed = config.seed if config.seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
            config.seed = seed
            generator = torch.Generator(device="cpu").manual_seed(seed)

            extracted_dir = os.path.join(job.output_dir, "extracted")
            for i in range(frame_count):
                if job._cancel_requested:
                    job.status = JobStatus.CANCELLED
                    self._write_project_json(job)
                    return
                job.current_frame = i

                src_path = os.path.join(extracted_dir, f"frame_{i + 1:04d}.png")
                src_image = Image.open(src_path).convert("RGB")

                job.current_step = 0
                job.total_steps = max(1, round(config.denoising_strength * config.steps))
                result = self._img2img_pipe(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt or None,
                    image=src_image,
                    strength=config.denoising_strength,
                    num_inference_steps=config.steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
                    callback_on_step_end=self._make_step_callback(job),
                )
                out_image = result.images[0]
                out_image.save(os.path.join(job.output_dir, f"frame_{i:04d}.png"))

            job.current_frame = frame_count

            frame_pattern = os.path.join(job.output_dir, "frame_%04d.png")
            video_out = os.path.join(job.output_dir, "output.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(config.extraction_fps),
                    "-i", frame_pattern,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    video_out,
                ],
                capture_output=True,
                check=True,
            )

            job.status = JobStatus.DONE
            self._write_project_json(job)

        except Exception as e:
            job.status = JobStatus.ERROR
            job.error_message = str(e)
            logger.exception("Job %s failed", job.job_id)
            self._write_project_json(job)
