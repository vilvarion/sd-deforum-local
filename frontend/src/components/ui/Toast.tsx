import { Button } from "react-aria-components";
import styles from "./Toast.module.css";

export interface ToastMessage {
  message: string;
  type: "error" | "info";
}

interface ToastProps {
  toast: ToastMessage | null;
  onDismiss: () => void;
}

export default function Toast({ toast, onDismiss }: ToastProps) {
  if (!toast) return null;

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={`${styles.toast} ${toast.type === "error" ? styles.error : styles.info}`}
    >
      <span className={styles.message}>{toast.message}</span>
      <Button
        onPress={onDismiss}
        className={styles.dismiss}
        aria-label="Dismiss notification"
      >
        ✕
      </Button>
    </div>
  );
}
