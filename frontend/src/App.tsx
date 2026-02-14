import { useState, useRef, useCallback, useEffect } from "react";
import { GenerationConfig, Vid2VidConfig, JobStatus, ModelInfo, defaultConfig, defaultVid2VidConfig } from "./types";
import Controls from "./components/Controls";
import Vid2VidControls from "./components/Vid2VidControls";
import Img2VidControls from "./components/Img2VidControls";
import Preview from "./components/Preview";
import styles from "./App.module.css";

export default function App() {
  const [activeTab, setActiveTab] = useState<"deforum" | "vid2vid" | "img2vid">("deforum");
  const [config, setConfig] = useState<GenerationConfig>({ ...defaultConfig });
  const [vid2vidConfig, setVid2VidConfig] = useState<Vid2VidConfig>({ ...defaultVid2VidConfig });
  const [img2vidConfig, setImg2VidConfig] = useState<GenerationConfig>({ ...defaultConfig });
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [generating, setGenerating] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((data: ModelInfo[]) => setModels(data))
      .catch(() => {});
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (id: string) => {
      stopPolling();
      pollRef.current = window.setInterval(async () => {
        try {
          const res = await fetch(`/api/jobs/${id}/status`);
          if (!res.ok) return;
          const data: JobStatus = await res.json();
          setStatus(data);
          if (data.status === "done" || data.status === "error" || data.status === "cancelled") {
            stopPolling();
            setGenerating(false);
          }
        } catch {
          /* ignore network blips */
        }
      }, 1000);
    },
    [stopPolling]
  );

  useEffect(() => () => stopPolling(), [stopPolling]);

  const handleCancel = useCallback(async () => {
    if (!jobId) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      if (!res.ok) return;
      stopPolling();
      setGenerating(false);
      setStatus((prev) => prev ? { ...prev, status: "error", error_message: "Cancelled by user" } : prev);
    } catch {
      /* ignore */
    }
  }, [jobId, stopPolling]);

  const handleGenerate = async () => {
    setGenerating(true);
    setStatus(null);
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (res.status === 409) {
        alert("A job is already running.");
        setGenerating(false);
        return;
      }
      if (!res.ok) {
        alert("Failed to start generation.");
        setGenerating(false);
        return;
      }
      const { job_id } = await res.json();
      setJobId(job_id);
      setStatus({ status: "queued", current_frame: 0, total_frames: config.num_frames, error_message: "" });
      startPolling(job_id);
    } catch {
      alert("Could not reach server.");
      setGenerating(false);
    }
  };

  const handleVid2VidGenerate = async (file: File) => {
    setGenerating(true);
    setStatus(null);
    try {
      const formData = new FormData();
      formData.append("video", file);
      formData.append("config_json", JSON.stringify(vid2vidConfig));
      const res = await fetch("/api/vid2vid", {
        method: "POST",
        body: formData,
      });
      if (res.status === 409) {
        alert("A job is already running.");
        setGenerating(false);
        return;
      }
      if (!res.ok) {
        alert("Failed to start vid2vid processing.");
        setGenerating(false);
        return;
      }
      const { job_id } = await res.json();
      setJobId(job_id);
      setStatus({ status: "queued", current_frame: 0, total_frames: 0, error_message: "" });
      startPolling(job_id);
    } catch {
      alert("Could not reach server.");
      setGenerating(false);
    }
  };

  const handleImg2VidGenerate = async (file: File) => {
    setGenerating(true);
    setStatus(null);
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("config_json", JSON.stringify(img2vidConfig));
      const res = await fetch("/api/img2vid", {
        method: "POST",
        body: formData,
      });
      if (res.status === 409) {
        alert("A job is already running.");
        setGenerating(false);
        return;
      }
      if (!res.ok) {
        alert("Failed to start img2vid processing.");
        setGenerating(false);
        return;
      }
      const { job_id } = await res.json();
      setJobId(job_id);
      setStatus({ status: "queued", current_frame: 0, total_frames: img2vidConfig.num_frames, error_message: "" });
      startPolling(job_id);
    } catch {
      alert("Could not reach server.");
      setGenerating(false);
    }
  };

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        Deforum Studio
        <div className={styles.tabs}>
          <button
            className={`${styles.tabBtn} ${activeTab === "deforum" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("deforum")}
          >
            Deforum
          </button>
          <button
            className={`${styles.tabBtn} ${activeTab === "vid2vid" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("vid2vid")}
          >
            Vid2Vid
          </button>
          <button
            className={`${styles.tabBtn} ${activeTab === "img2vid" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("img2vid")}
          >
            Img2Vid
          </button>
        </div>
      </header>
      <main className={styles.main}>
        {activeTab === "deforum" ? (
          <Controls
            config={config}
            onChange={setConfig}
            onGenerate={handleGenerate}
            disabled={generating}
            models={models}
          />
        ) : activeTab === "vid2vid" ? (
          <Vid2VidControls
            config={vid2vidConfig}
            onChange={setVid2VidConfig}
            onGenerate={handleVid2VidGenerate}
            disabled={generating}
            models={models}
          />
        ) : (
          <Img2VidControls
            config={img2vidConfig}
            onChange={setImg2VidConfig}
            onGenerate={handleImg2VidGenerate}
            disabled={generating}
            models={models}
          />
        )}
        <Preview jobId={jobId} status={status} onCancel={generating ? handleCancel : undefined} />
      </main>
    </div>
  );
}
