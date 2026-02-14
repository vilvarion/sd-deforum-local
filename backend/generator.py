import os
import threading
import time
import uuid
import subprocess
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

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
    status: JobStatus = JobStatus.QUEUED
    current_frame: int = 0
    total_frames: int = 0
    error_message: str = ""
    output_dir: str = ""
    _cancel_requested: bool = field(default=False, repr=False)


class DeforumGenerator:
    def __init__(self, output_root: str = "outputs", models_dir: str = "models"):
        self.output_root = output_root
        self.models_dir = models_dir
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._busy = False
        self._lock = threading.Lock()
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
            print(f"Retrying with linear-projection config for {local_path}")
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
        print(f"Model '{model_id}' loaded on {self._device} with dtype {self._dtype}")

    @property
    def is_busy(self) -> bool:
        return self._busy

    def submit_job(self, config: GenerationConfig) -> Optional[str]:
        with self._lock:
            if self._busy:
                return None
            self._busy = True

        job_id = uuid.uuid4().hex[:12]
        output_dir = os.path.join(self.output_root, job_id)
        os.makedirs(output_dir, exist_ok=True)

        job = Job(
            job_id=job_id,
            config=config,
            status=JobStatus.QUEUED,
            total_frames=config.num_frames,
            output_dir=output_dir,
        )
        self.jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job_id

    def submit_img2vid_job(self, config: GenerationConfig, image_path: str) -> Optional[str]:
        with self._lock:
            if self._busy:
                return None
            self._busy = True

        job_id = uuid.uuid4().hex[:12]
        output_dir = os.path.join(self.output_root, job_id)
        os.makedirs(output_dir, exist_ok=True)

        dest_image = os.path.join(output_dir, "input_image" + os.path.splitext(image_path)[1])
        shutil.copy2(image_path, dest_image)

        job = Job(
            job_id=job_id,
            config=config,
            status=JobStatus.QUEUED,
            total_frames=config.num_frames,
            output_dir=output_dir,
        )
        self.jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job,), kwargs={"init_image_path": dest_image}, daemon=True)
        thread.start()
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return False
        job._cancel_requested = True
        return True

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def get_frame_path(self, job_id: str, frame_number: int) -> Optional[str]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        path = os.path.join(job.output_dir, f"frame_{frame_number:04d}.png")
        if os.path.exists(path):
            return path
        return None

    def get_video_path(self, job_id: str) -> Optional[str]:
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.DONE:
            return None
        path = os.path.join(job.output_dir, "output.mp4")
        if os.path.exists(path):
            return path
        return None

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

        warped = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
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
                    return
                job.current_frame = frame_idx

                if init_image_path is not None and not config.use_deforum:
                    # Non-deforum img2vid: every frame uses the source image as base
                    result = self._img2img_pipe(
                        prompt=config.prompt,
                        negative_prompt=config.negative_prompt or None,
                        image=source_image,
                        strength=config.denoising_strength,
                        num_inference_steps=config.steps,
                        guidance_scale=config.guidance_scale,
                        generator=generator,
                    )
                    image = result.images[0]
                elif frame_idx == 0:
                    if source_image is not None:
                        result = self._img2img_pipe(
                            prompt=config.prompt,
                            negative_prompt=config.negative_prompt or None,
                            image=source_image,
                            strength=config.denoising_strength,
                            num_inference_steps=config.steps,
                            guidance_scale=config.guidance_scale,
                            generator=generator,
                        )
                        image = result.images[0]
                    else:
                        result = self._txt2img_pipe(
                            prompt=config.prompt,
                            negative_prompt=config.negative_prompt or None,
                            width=config.width,
                            height=config.height,
                            num_inference_steps=config.steps,
                            guidance_scale=config.guidance_scale,
                            generator=generator,
                        )
                        image = result.images[0]
                else:
                    prev_np = np.array(prev_image)
                    warped_np = self._apply_warp(prev_np, config)

                    if config.color_coherence and reference_frame is not None:
                        warped_np = self._match_color(warped_np, reference_frame)

                    warped_pil = Image.fromarray(warped_np)

                    result = self._img2img_pipe(
                        prompt=config.prompt,
                        negative_prompt=config.negative_prompt or None,
                        image=warped_pil,
                        strength=config.denoising_strength,
                        num_inference_steps=config.steps,
                        guidance_scale=config.guidance_scale,
                        generator=generator,
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

        except Exception as e:
            job.status = JobStatus.ERROR
            job.error_message = str(e)
            import traceback
            traceback.print_exc()
        finally:
            with self._lock:
                self._busy = False

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
                    return
                job.current_frame = i

                src_path = os.path.join(extracted_dir, f"frame_{i + 1:04d}.png")
                src_image = Image.open(src_path).convert("RGB")

                result = self._img2img_pipe(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt or None,
                    image=src_image,
                    strength=config.denoising_strength,
                    num_inference_steps=config.steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
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

        except Exception as e:
            job.status = JobStatus.ERROR
            job.error_message = str(e)
            import traceback
            traceback.print_exc()
        finally:
            with self._lock:
                self._busy = False

    def submit_vid2vid_job(self, config: Vid2VidConfig, video_path: str) -> Optional[str]:
        with self._lock:
            if self._busy:
                return None
            self._busy = True

        job_id = uuid.uuid4().hex[:12]
        output_dir = os.path.join(self.output_root, job_id)
        os.makedirs(output_dir, exist_ok=True)

        # Copy uploaded video into job directory
        dest_video = os.path.join(output_dir, "input_video" + os.path.splitext(video_path)[1])
        shutil.copy2(video_path, dest_video)

        job = Job(
            job_id=job_id,
            config=config,
            status=JobStatus.QUEUED,
            total_frames=0,
            output_dir=output_dir,
        )
        self.jobs[job_id] = job

        thread = threading.Thread(target=self._run_vid2vid_job, args=(job, dest_video), daemon=True)
        thread.start()
        return job_id
