import { Slider, SliderTrack, SliderThumb, Label } from "react-aria-components";
import styles from "./SliderField.module.css";

interface SliderFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}

export default function SliderField({ label, value, min, max, step, onChange }: SliderFieldProps) {
  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const parsed = parseFloat(e.target.value);
    if (!isNaN(parsed)) {
      onChange(Math.min(max, Math.max(min, parsed)));
    }
  };

  return (
    <div className={styles.field}>
      <Slider
        minValue={min}
        maxValue={max}
        step={step}
        value={value}
        onChange={onChange}
        className={styles.slider}
      >
        <Label className={styles.label}>{label}</Label>
        <div className={styles.sliderRow}>
          <SliderTrack className={styles.track}>
            <SliderThumb className={styles.thumb} />
          </SliderTrack>
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={handleNumberChange}
            className={styles.numInput}
            aria-label={label}
          />
        </div>
      </Slider>
    </div>
  );
}
