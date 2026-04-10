import { useState, useRef, useCallback, useEffect } from "react";
import { JobStatus, ModelInfo } from "../types";

interface UseJobPollingResult {
  status: JobStatus | null;
  jobId: string | null;
  generating: boolean;
  models: ModelInfo[];
  startJob: (id: string, initialStatus: JobStatus) => void;
  handleCancel: () => Promise<void>;
}

export function useJobPolling(): UseJobPollingResult {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const pollRef = useRef<number | null>(null);
  const jobIdRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchModels = async () => {
      while (!cancelled) {
        try {
          const r = await fetch("/api/models");
          if (r.ok) {
            const data: ModelInfo[] = await r.json();
            if (!cancelled) setModels(data);
            return;
          }
        } catch {
          /* retry */
        }
        await new Promise((res) => setTimeout(res, 2000));
      }
    };
    fetchModels();
    return () => {
      cancelled = true;
    };
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const startJob = useCallback(
    (id: string, initialStatus: JobStatus) => {
      stopPolling();
      setJobId(id);
      jobIdRef.current = id;
      setGenerating(true);
      setStatus(initialStatus);

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

  const handleCancel = useCallback(async () => {
    const id = jobIdRef.current;
    if (!id) return;
    try {
      const res = await fetch(`/api/jobs/${id}/cancel`, { method: "POST" });
      if (!res.ok) return;
      stopPolling();
      setGenerating(false);
      setStatus((prev) =>
        prev ? { ...prev, status: "cancelled", error_message: "Cancelled by user" } : prev
      );
    } catch {
      /* ignore */
    }
  }, [stopPolling]);

  return { status, jobId, generating, models, startJob, handleCancel };
}
