import styles from "../App.module.css";

export type Mode = "deforum" | "img2vid" | "vid2vid" | "queue" | "gallery";

export const MODE_LABELS: Record<Mode, string> = {
  deforum: "Deforum",
  img2vid: "Img2Vid",
  vid2vid: "Vid2Vid",
  queue: "Queue",
  gallery: "Gallery",
};

export const GENERATION_MODES: ReadonlyArray<Mode> = ["deforum", "img2vid", "vid2vid"];

interface TabsProps {
  mode: Mode;
  onChange: (mode: Mode) => void;
}

export default function Tabs({ mode, onChange }: TabsProps) {
  return (
    <div className={styles.tabList} role="tablist">
      {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
        <button
          key={m}
          role="tab"
          aria-selected={mode === m}
          className={`${styles.tab}${mode === m ? ` ${styles.tabSelected}` : ""}`}
          onClick={() => onChange(m)}
        >
          {MODE_LABELS[m]}
        </button>
      ))}
    </div>
  );
}
