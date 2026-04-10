import { useState, useCallback } from "react";
import { GenerationConfig, Vid2VidConfig, JobStatus } from "../types";
import { ToastMessage } from "../components/ui/Toast";

interface UseGenerationActionsProps {
  config: GenerationConfig;
  vid2vidConfig: Vid2VidConfig;
  img2vidConfig: GenerationConfig;
  startJob: (id: string, initialStatus: JobStatus) => void;
}

interface UseGenerationActionsResult {
  toast: ToastMessage | null;
  clearToast: () => void;
  handleGenerate: (overrideConfig?: GenerationConfig) => Promise<void>;
  handleVid2VidGenerate: (file: File, overrideConfig?: Vid2VidConfig) => Promise<void>;
  handleImg2VidGenerate: (file: File, overrideConfig?: GenerationConfig) => Promise<void>;
}

export function useGenerationActions({
  config,
  vid2vidConfig,
  img2vidConfig,
  startJob,
}: UseGenerationActionsProps): UseGenerationActionsResult {
  const [toast, setToast] = useState<ToastMessage | null>(null);
  const clearToast = useCallback(() => setToast(null), []);

  const handleGenerate = useCallback(
    async (overrideConfig?: GenerationConfig) => {
      const cfg = overrideConfig ?? config;
      try {
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(cfg),
        });
        if (res.status === 409) {
          setToast({ message: "A job is already running.", type: "error" });
          return;
        }
        if (!res.ok) {
          setToast({ message: "Failed to start generation.", type: "error" });
          return;
        }
        const { job_id } = await res.json();
        startJob(job_id, {
          status: "queued",
          current_frame: 0,
          total_frames: cfg.num_frames,
          current_step: 0,
          total_steps: 0,
          error_message: "",
        });
      } catch {
        setToast({ message: "Could not reach server.", type: "error" });
      }
    },
    [config, startJob]
  );

  const handleVid2VidGenerate = useCallback(
    async (file: File, overrideConfig?: Vid2VidConfig) => {
      const cfg = overrideConfig ?? vid2vidConfig;
      try {
        const formData = new FormData();
        formData.append("video", file);
        formData.append("config_json", JSON.stringify(cfg));
        const res = await fetch("/api/vid2vid", { method: "POST", body: formData });
        if (res.status === 409) {
          setToast({ message: "A job is already running.", type: "error" });
          return;
        }
        if (!res.ok) {
          setToast({ message: "Failed to start vid2vid processing.", type: "error" });
          return;
        }
        const { job_id } = await res.json();
        startJob(job_id, {
          status: "queued",
          current_frame: 0,
          total_frames: 0,
          current_step: 0,
          total_steps: 0,
          error_message: "",
        });
      } catch {
        setToast({ message: "Could not reach server.", type: "error" });
      }
    },
    [vid2vidConfig, startJob]
  );

  const handleImg2VidGenerate = useCallback(
    async (file: File, overrideConfig?: GenerationConfig) => {
      const cfg = overrideConfig ?? img2vidConfig;
      try {
        const formData = new FormData();
        formData.append("image", file);
        formData.append("config_json", JSON.stringify(cfg));
        const res = await fetch("/api/img2vid", { method: "POST", body: formData });
        if (res.status === 409) {
          setToast({ message: "A job is already running.", type: "error" });
          return;
        }
        if (!res.ok) {
          setToast({ message: "Failed to start img2vid processing.", type: "error" });
          return;
        }
        const { job_id } = await res.json();
        startJob(job_id, {
          status: "queued",
          current_frame: 0,
          total_frames: cfg.num_frames,
          current_step: 0,
          total_steps: 0,
          error_message: "",
        });
      } catch {
        setToast({ message: "Could not reach server.", type: "error" });
      }
    },
    [img2vidConfig, startJob]
  );

  return { toast, clearToast, handleGenerate, handleVid2VidGenerate, handleImg2VidGenerate };
}
