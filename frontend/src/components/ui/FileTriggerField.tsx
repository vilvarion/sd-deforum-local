import { FileTrigger, Button } from "react-aria-components";
import styles from "./FileTriggerField.module.css";

interface FileTriggerFieldProps {
  label: string;
  accept: string;
  selectedFile: File | null;
  onSelect: (file: File) => void;
}

export default function FileTriggerField({ label, accept, selectedFile, onSelect }: FileTriggerFieldProps) {
  return (
    <div className={styles.field}>
      <span className={styles.label}>{label}</span>
      <FileTrigger
        acceptedFileTypes={accept.split(",").map((s) => s.trim())}
        onSelect={(files) => {
          const file = files?.[0];
          if (file) onSelect(file);
        }}
      >
        <Button className={styles.button}>Choose File</Button>
      </FileTrigger>
      {selectedFile && (
        <span className={styles.filename}>{selectedFile.name}</span>
      )}
    </div>
  );
}
