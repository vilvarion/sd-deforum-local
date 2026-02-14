import { useState, useRef } from "react";
import { Vid2VidConfig, ModelInfo } from "../types";
import SizeSelect from "./SizeSelect";
import styles from "./Controls.module.css";

interface Props {
  config: Vid2VidConfig;
  onChange: (c: Vid2VidConfig) => void;
  onGenerate: (file: File) => void;
  disabled: boolean;
  models: ModelInfo[];
}

function SliderInput({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className={styles.field}>
      <label>{label}</label>
      <div className={styles.sliderRow}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value) || min)}
          className={styles.numInput}
        />
      </div>
    </div>
  );
}

export default function Vid2VidControls({ config, onChange, onGenerate, disabled, models }: Props) {
  const [randomSeed, setRandomSeed] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const set = <K extends keyof Vid2VidConfig>(key: K, value: Vid2VidConfig[K]) => {
    onChange({ ...config, [key]: value });
  };

  const handleGenerate = () => {
    if (!selectedFile) return;
    if (randomSeed) {
      onChange({ ...config, seed: null });
    }
    onGenerate(selectedFile);
  };

  return (
    <div className={styles.panel}>
      <div className={styles.field}>
        <label>Model</label>
        <select value={config.model_id} onChange={(e) => set("model_id", e.target.value)}>
          {models.map((m) => (
            <option key={m.id} value={m.id}>{m.name}</option>
          ))}
        </select>
      </div>

      <div className={styles.field}>
        <label>Video File</label>
        <input
          ref={fileRef}
          type="file"
          accept="video/*"
          onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
          style={{ fontSize: 13, color: "#ccc" }}
        />
        {selectedFile && (
          <span style={{ fontSize: 12, color: "#888", marginTop: 2 }}>{selectedFile.name}</span>
        )}
      </div>

      <div className={styles.section}>
        <label>Prompt</label>
        <textarea
          className={styles.promptArea}
          rows={4}
          value={config.prompt}
          onChange={(e) => set("prompt", e.target.value)}
          placeholder="A fantasy landscape, highly detailed..."
        />
      </div>

      <div className={styles.section}>
        <label>Negative Prompt</label>
        <textarea
          className={styles.negArea}
          rows={2}
          value={config.negative_prompt}
          onChange={(e) => set("negative_prompt", e.target.value)}
          placeholder="blurry, low quality..."
        />
      </div>

      <SizeSelect
        width={config.width}
        height={config.height}
        onWidthChange={(v) => set("width", v)}
        onHeightChange={(v) => set("height", v)}
      />

      <SliderInput label="Denoising Strength" value={config.denoising_strength} min={0} max={1} step={0.05} onChange={(v) => set("denoising_strength", v)} />
      <SliderInput label="Guidance Scale" value={config.guidance_scale} min={1} max={20} step={0.5} onChange={(v) => set("guidance_scale", v)} />
      <SliderInput label="Steps" value={config.steps} min={10} max={50} step={1} onChange={(v) => set("steps", v)} />
      <SliderInput label="Extraction FPS" value={config.extraction_fps} min={6} max={30} step={1} onChange={(v) => set("extraction_fps", v)} />

      <div className={styles.field}>
        <label>Seed</label>
        <div className={styles.seedRow}>
          <input
            type="number"
            value={config.seed ?? ""}
            disabled={randomSeed}
            onChange={(e) => set("seed", parseInt(e.target.value) || 0)}
            className={styles.numInput}
            placeholder="Random"
          />
          <label className={styles.checkLabel}>
            <input
              type="checkbox"
              checked={randomSeed}
              onChange={(e) => {
                setRandomSeed(e.target.checked);
                if (e.target.checked) set("seed", null);
              }}
            />
            Random
          </label>
        </div>
      </div>

      <button
        className={styles.generateBtn}
        onClick={handleGenerate}
        disabled={disabled || !config.prompt.trim() || !selectedFile}
      >
        {disabled ? "Processing..." : "Process Video"}
      </button>
    </div>
  );
}
