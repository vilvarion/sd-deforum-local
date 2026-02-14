import styles from "./Controls.module.css";

interface Props {
  width: number;
  height: number;
  onWidthChange: (v: number) => void;
  onHeightChange: (v: number) => void;
}

const SIZES = [256, 384, 512, 768];

export default function SizeSelect({ width, height, onWidthChange, onHeightChange }: Props) {
  return (
    <div className={styles.grid}>
      <div className={styles.field}>
        <label>Width</label>
        <select value={width} onChange={(e) => onWidthChange(parseInt(e.target.value))}>
          {SIZES.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>
      <div className={styles.field}>
        <label>Height</label>
        <select value={height} onChange={(e) => onHeightChange(parseInt(e.target.value))}>
          {SIZES.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>
    </div>
  );
}
