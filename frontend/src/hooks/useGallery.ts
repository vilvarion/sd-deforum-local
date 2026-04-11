import { useCallback, useEffect, useState } from "react";
import { GalleryItem } from "../types";

export function useGallery(active: boolean) {
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/gallery");
      if (res.ok) {
        const data: GalleryItem[] = await res.json();
        setItems(data);
      }
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (active) refresh();
  }, [active, refresh]);

  const deleteItem = useCallback(
    async (jobId: string) => {
      try {
        const res = await fetch(`/api/gallery/${jobId}`, { method: "DELETE" });
        if (res.ok) {
          setItems((prev) => prev.filter((i) => i.job_id !== jobId));
        }
      } catch {
        /* ignore */
      }
    },
    []
  );

  return { items, loading, refresh, deleteItem };
}
