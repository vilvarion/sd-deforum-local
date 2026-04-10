import { memo, useMemo, useRef, useState, useEffect, useCallback } from "react";
import { ProgressBar, Button } from "react-aria-components";
// Button is used for Cancel and thumbnails; ProgressBar for frame/step progress
import { JobStatus } from "../types";
import styles from "./Preview.module.css";

const VideoPlayer = memo(function VideoPlayer({ jobId }: { jobId: string }) {
  const srcRef = useRef(`/api/jobs/${jobId}/video?t=${Date.now()}`);
  return (
    <div className={styles.videoSection}>
      <video className={styles.video} src={srcRef.current} controls autoPlay loop />
      <a
        className={styles.downloadBtn}
        href={`/api/jobs/${jobId}/video`}
        download="deforum.mp4"
      >
        Download MP4
      </a>
    </div>
  );
});

interface Props {
  jobId: string | null;
  status: JobStatus | null;
  onCancel?: () => void;
  imageSize?: { width: number; height: number } | null;
}

export default memo(function Preview({ jobId, status, onCancel, imageSize }: Props) {
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

  useEffect(() => {
    setSelectedFrame(null);
  }, [jobId]);

  const displayFrame =
    selectedFrame !== null && selectedFrame < availableFrames.length ? selectedFrame : latestFrame;

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

  const progressPct =
    status.total_frames > 0 ? Math.round((status.current_frame / status.total_frames) * 100) : 0;
  const stepPct =
    status.total_steps > 0 ? Math.round((status.current_step / status.total_steps) * 100) : 0;
  const showPlaceholder = isRunning && displayFrame === null && imageSize != null;

  return (
    <div className={styles.panel}>
      {isError && (
        <div role="alert" className={styles.error}>
          Error: {status.error_message}
        </div>
      )}

      {isRunning && (
        <div className={styles.progressSection}>
          <ProgressBar
            value={progressPct}
            minValue={0}
            maxValue={100}
            aria-label={`Frame progress: ${status.current_frame} of ${status.total_frames}`}
            className={styles.progressBarRoot}
          >
            <div className={styles.progressTrack}>
              <div className={styles.progressFill} style={{ width: `${progressPct}%` }} />
            </div>
          </ProgressBar>
          <span className={styles.progressText}>
            Frame {status.current_frame} / {status.total_frames} ({progressPct}%)
          </span>
          {status.total_steps > 0 && (
            <>
              <ProgressBar
                value={stepPct}
                minValue={0}
                maxValue={100}
                aria-label={`Step progress: ${status.current_step} of ${status.total_steps}`}
                className={styles.progressBarRoot}
              >
                <div className={styles.stepTrack}>
                  <div className={styles.stepFill} style={{ width: `${stepPct}%` }} />
                </div>
              </ProgressBar>
              <span className={styles.progressText}>
                Step {status.current_step} / {status.total_steps}
              </span>
            </>
          )}
          {onCancel && (
            <Button className={styles.cancelBtn} onPress={onCancel}>
              Cancel
            </Button>
          )}
        </div>
      )}

      {showPlaceholder && (
        <div className={styles.mainFrame} style={{ width: imageSize!.width, height: imageSize!.height }}>
          <div className={styles.placeholder} />
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
        <div className={styles.thumbStrip} role="list" aria-label="Frame thumbnails">
          {availableFrames.map((idx) => (
            <Button
              key={idx}
              onPress={() => setSelectedFrame(idx)}
              className={`${styles.thumb} ${idx === displayFrame ? styles.thumbActive : ""}`}
              aria-label={`Frame ${idx}`}
              aria-pressed={idx === displayFrame}
            >
              <img
                src={`/api/jobs/${jobId}/frames/${idx}`}
                alt=""
                className={styles.thumbImg}
              />
            </Button>
          ))}
        </div>
      )}

      {isDone && <VideoPlayer jobId={jobId} />}
    </div>
  );
});
