import { useState } from "react";
import { Button } from "react-aria-components";
import { GalleryItem, JobStatus } from "../types";
import styles from "./Gallery.module.css";

interface Props {
  items: GalleryItem[];
  onSelect: (jobId: string, status: JobStatus) => void;
  onDelete: (jobId: string) => void;
}

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

export default function Gallery({ items, onSelect, onDelete }: Props) {
  const [confirmId, setConfirmId] = useState<string | null>(null);

  if (items.length === 0) {
    return (
      <div className={styles.panel}>
        <div className={styles.empty}>No projects yet. Generate something to fill this space.</div>
      </div>
    );
  }

  const handleSelect = (item: GalleryItem) => {
    const status: JobStatus = {
      status: item.status === "running" ? "done" : item.status,
      current_frame: item.num_frames,
      total_frames: item.num_frames,
      current_step: 0,
      total_steps: 0,
      error_message: "",
    };
    onSelect(item.job_id, status);
  };

  return (
    <div className={styles.panel}>
      <div className={styles.grid}>
        {items.map((item) => {
          const thumbUrl =
            item.thumbnail_frame !== null
              ? `/api/jobs/${item.job_id}/frames/${item.thumbnail_frame}`
              : null;
          return (
            <div
              key={item.job_id}
              className={styles.card}
              onClick={() => handleSelect(item)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  handleSelect(item);
                }
              }}
            >
              {thumbUrl ? (
                <img src={thumbUrl} alt={item.prompt} className={styles.thumb} />
              ) : (
                <div className={styles.thumbPlaceholder}>No preview</div>
              )}
              <div className={styles.meta}>
                <div className={styles.metaRow}>
                  <span className={styles.mode}>{item.mode}</span>
                  <span>{formatDate(item.created_at)}</span>
                </div>
                <div className={styles.prompt} title={item.prompt}>
                  {item.prompt || <em>(no prompt)</em>}
                </div>
                <div className={styles.metaRow}>
                  <span>
                    {item.width}×{item.height}
                  </span>
                  <span>{item.num_frames} frames</span>
                </div>
              </div>
              <div
                className={styles.deleteWrap}
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => e.stopPropagation()}
              >
                <Button
                  className={`${styles.deleteBtn}${confirmId === item.job_id ? ` ${styles.deleteConfirm}` : ""}`}
                  onPress={() => {
                    if (confirmId === item.job_id) {
                      onDelete(item.job_id);
                      setConfirmId(null);
                    } else {
                      setConfirmId(item.job_id);
                    }
                  }}
                  aria-label={`Delete project ${item.job_id}`}
                >
                  {confirmId === item.job_id ? "Confirm?" : "Delete"}
                </Button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
