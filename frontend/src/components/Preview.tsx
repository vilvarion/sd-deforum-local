import { memo, useMemo, useState, useEffect, useCallback } from "react";
import { JobStatus } from "../types";
import styles from "./Preview.module.css";

interface Props {
  jobId: string | null;
  status: JobStatus | null;
  onCancel?: () => void;
}

export default memo(function Preview({ jobId, status, onCancel }: Props) {
  const availableFrames = useMemo(() => {
    if (!status || !jobId) return [];
    const count = status.status === "done" ? status.total_frames : status.current_frame;
    return Array.from({ length: count }, (_, i) => i);
  }, [status, jobId]);

  const latestFrame = availableFrames.length > 0 ? availableFrames[availableFrames.length - 1] : null;
  const [selectedFrame, setSelectedFrame] = useState<number | null>(null);
  const isDone = status?.status === "done";
  const isError = status?.status === "error";
  const isRunning = status?.status === "running" || status?.status === "queued";

  // Reset selection when job changes
  useEffect(() => {
    setSelectedFrame(null);
  }, [jobId]);

  // Auto-follow latest frame when no manual selection
  const displayFrame = selectedFrame !== null && selectedFrame < availableFrames.length
    ? selectedFrame
    : latestFrame;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (availableFrames.length === 0) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setSelectedFrame((prev) => {
          const current = prev !== null ? prev : latestFrame ?? 0;
          return Math.max(0, current - 1);
        });
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setSelectedFrame((prev) => {
          const current = prev !== null ? prev : latestFrame ?? 0;
          return Math.min(availableFrames.length - 1, current + 1);
        });
      }
    },
    [availableFrames.length, latestFrame]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  if (!jobId || !status) {
    return (
      <div className={styles.panel}>
        <div className={styles.empty}>Configure settings and click Generate to start.</div>
      </div>
    );
  }

  const progressPct = status.total_frames > 0
    ? Math.round((status.current_frame / status.total_frames) * 100)
    : 0;

  return (
    <div className={styles.panel}>
      {isError && <div className={styles.error}>Error: {status.error_message}</div>}

      {isRunning && (
        <div className={styles.progressSection}>
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: `${progressPct}%` }} />
          </div>
          <span className={styles.progressText}>
            Frame {status.current_frame} / {status.total_frames} ({progressPct}%)
          </span>
          {onCancel && (
            <button className={styles.cancelBtn} onClick={onCancel}>
              Cancel
            </button>
          )}
        </div>
      )}

      {isDone && (
        <div className={styles.videoSection}>
          <video
            className={styles.video}
            src={`/api/jobs/${jobId}/video?t=${Date.now()}`}
            controls
            autoPlay
            loop
          />
          <a
            className={styles.downloadBtn}
            href={`/api/jobs/${jobId}/video`}
            download="deforum.mp4"
          >
            Download MP4
          </a>
        </div>
      )}

      {displayFrame !== null && (isDone ? selectedFrame !== null : true) && (
        <div className={styles.mainFrame}>
          <img
            src={`/api/jobs/${jobId}/frames/${displayFrame}?t=${Date.now()}`}
            alt={`Frame ${displayFrame}`}
            className={styles.mainImage}
          />
        </div>
      )}

      {availableFrames.length > 0 && (
        <div className={styles.thumbStrip}>
          {availableFrames.map((idx) => (
            <img
              key={idx}
              src={`/api/jobs/${jobId}/frames/${idx}`}
              alt={`Frame ${idx}`}
              className={`${styles.thumb} ${idx === displayFrame ? styles.thumbActive : ""}`}
              onClick={() => setSelectedFrame(idx)}
            />
          ))}
        </div>
      )}
    </div>
  );
})
