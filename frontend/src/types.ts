import {NEGATIVE_PROMPT, POSITIVE_PROMPT} from "./defaults.ts";

export interface ModelInfo {
  id: string;
  name: string;
}

export interface PromptKeyframe {
  frame: number;
  prompt: string;
  blend_frames: number;
}

export interface GenerationConfig {
  prompt: string;
  negative_prompt: string;
  num_frames: number;
  width: number;
  height: number;
  denoising_strength: number;
  guidance_scale: number;
  steps: number;
  zoom_per_frame: number;
  rotate_per_frame: number;
  translate_x: number;
  translate_y: number;
  seed: number | null;
  fps: number;
  color_coherence: boolean;
  use_deforum: boolean;
  model_id: string;
  prompt_schedule: PromptKeyframe[];
}

export interface JobStatus {
  status: "queued" | "running" | "done" | "error" | "cancelled";
  current_frame: number;
  total_frames: number;
  current_step: number;
  total_steps: number;
  error_message: string;
}

export type GenerationMode = "deforum" | "img2vid" | "vid2vid";

export interface QueueItem {
  job_id: string;
  mode: GenerationMode;
  status: JobStatus["status"];
  prompt: string;
  created_at: string;
  current_frame: number;
  total_frames: number;
}

export interface QueueSnapshot {
  queued: QueueItem[];
  running: QueueItem | null;
  recent: QueueItem[];
}

export interface GalleryItem {
  job_id: string;
  mode: GenerationMode;
  status: JobStatus["status"];
  created_at: string;
  prompt: string;
  width: number;
  height: number;
  num_frames: number;
  has_video: boolean;
  thumbnail_frame: number | null;
}

export interface Vid2VidConfig {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  denoising_strength: number;
  guidance_scale: number;
  steps: number;
  seed: number | null;
  extraction_fps: number;
  model_id: string;
}

export const DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5";

export const defaultVid2VidConfig: Vid2VidConfig = {
  prompt: POSITIVE_PROMPT,
  negative_prompt: NEGATIVE_PROMPT,
  width: 512,
  height: 512,
  denoising_strength: 0.55,
  guidance_scale: 7.5,
  steps: 25,
  seed: null,
  extraction_fps: 12,
  model_id: DEFAULT_MODEL_ID,
};

export const defaultConfig: GenerationConfig = {
  prompt: POSITIVE_PROMPT,
  negative_prompt: NEGATIVE_PROMPT,
  num_frames: 15,
  width: 512,
  height: 512,
  denoising_strength: 0.55,
  guidance_scale: 7.5,
  steps: 25,
  zoom_per_frame: 1.02,
  rotate_per_frame: 0.0,
  translate_x: 0,
  translate_y: 0,
  seed: null,
  fps: 12,
  color_coherence: true,
  use_deforum: true,
  model_id: DEFAULT_MODEL_ID,
  prompt_schedule: [],
};
