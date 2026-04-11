import { useCallback, useEffect, useRef, useState } from "react";
import { QueueSnapshot } from "../types";

const EMPTY: QueueSnapshot = { queued: [], running: null, recent: [] };

export function useQueuePolling(active: boolean) {
  const [snapshot, setSnapshot] = useState<QueueSnapshot>(EMPTY);
  const timerRef = useRef<number | null>(null);

  const fetchOnce = useCallback(async () => {
    try {
      const res = await fetch("/api/queue");
      if (!res.ok) return;
      const data: QueueSnapshot = await res.json();
      setSnapshot(data);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    if (!active) {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }
    fetchOnce();
    timerRef.current = window.setInterval(fetchOnce, 1000);
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [active, fetchOnce]);

  const cancelItem = useCallback(
    async (jobId: string) => {
      try {
        await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      } catch {
        /* ignore */
      }
      fetchOnce();
    },
    [fetchOnce]
  );

  return { snapshot, refresh: fetchOnce, cancelItem };
}
