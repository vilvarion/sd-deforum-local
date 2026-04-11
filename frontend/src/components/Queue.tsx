import { Button } from "react-aria-components";
import { QueueItem, QueueSnapshot, JobStatus } from "../types";
import styles from "./Queue.module.css";

interface Props {
  snapshot: QueueSnapshot;
  onCancel: (jobId: string) => void;
  onSelect: (jobId: string, status: JobStatus) => void;
}

const statusToJobStatus = (item: QueueItem): JobStatus => ({
  status: item.status,
  current_frame: item.current_frame,
  total_frames: item.total_frames,
  current_step: 0,
  total_steps: 0,
  error_message: "",
});

function Row({
  item,
  onCancel,
  onSelect,
  running,
}: {
  item: QueueItem;
  onCancel?: (id: string) => void;
  onSelect: (id: string, status: JobStatus) => void;
  running?: boolean;
}) {
  const handleClick = () => onSelect(item.job_id, statusToJobStatus(item));
  return (
    <div
      className={`${styles.item}${running ? ` ${styles.itemRunning}` : ""}`}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      }}
    >
      <span className={styles.badge}>{item.mode}</span>
      <span className={styles.prompt} title={item.prompt}>
        {item.prompt || <em>(no prompt)</em>}
      </span>
      <span className={styles.status}>
        {running && item.total_frames > 0
          ? `${item.current_frame}/${item.total_frames}`
          : item.status}
      </span>
      {onCancel && (
        <Button
          className={styles.cancelBtn}
          onPress={() => onCancel(item.job_id)}
        >
          Cancel
        </Button>
      )}
    </div>
  );
}

export default function Queue({ snapshot, onCancel, onSelect }: Props) {
  const { running, queued, recent } = snapshot;
  return (
    <div className={styles.panel}>
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Running</h3>
        {running ? (
          <div className={styles.list}>
            <Row item={running} onCancel={onCancel} onSelect={onSelect} running />
          </div>
        ) : (
          <div className={styles.empty}>Nothing running.</div>
        )}
      </div>

      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Queued ({queued.length})</h3>
        {queued.length > 0 ? (
          <div className={styles.list}>
            {queued.map((q) => (
              <Row key={q.job_id} item={q} onCancel={onCancel} onSelect={onSelect} />
            ))}
          </div>
        ) : (
          <div className={styles.empty}>Queue is empty.</div>
        )}
      </div>

      {recent.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Recent</h3>
          <div className={styles.list}>
            {recent.map((r) => (
              <Row key={r.job_id} item={r} onSelect={onSelect} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
