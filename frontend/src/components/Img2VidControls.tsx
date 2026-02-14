import { useState, useRef } from "react";
import { GenerationConfig, ModelInfo } from "../types";
import SizeSelect from "./SizeSelect";
import styles from "./Controls.module.css";

interface Props {
  config: GenerationConfig;
  onChange: (c: GenerationConfig) => void;
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

export default function Img2VidControls({ config, onChange, onGenerate, disabled, models }: Props) {
  const [randomSeed, setRandomSeed] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const set = <K extends keyof GenerationConfig>(key: K, value: GenerationConfig[K]) => {
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
        <label>Image File</label>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
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
      <SliderInput label="Frames" value={config.num_frames} min={2} max={30} step={1} onChange={(v) => set("num_frames", v)} />

      <div className={styles.field}>
        <label className={styles.checkLabel}>
          <input
            type="checkbox"
            checked={config.use_deforum}
            onChange={(e) => set("use_deforum", e.target.checked)}
          />
          Use Deforum
        </label>
      </div>

      {config.use_deforum && (
        <>
          <SliderInput label="Zoom / Frame" value={config.zoom_per_frame} min={0.9} max={1.1} step={0.01} onChange={(v) => set("zoom_per_frame", v)} />
          <SliderInput label="Rotate / Frame" value={config.rotate_per_frame} min={-5} max={5} step={0.5} onChange={(v) => set("rotate_per_frame", v)} />
          <SliderInput label="Translate X" value={config.translate_x} min={-20} max={20} step={1} onChange={(v) => set("translate_x", v)} />
          <SliderInput label="Translate Y" value={config.translate_y} min={-20} max={20} step={1} onChange={(v) => set("translate_y", v)} />
        </>
      )}

      <SliderInput label="FPS" value={config.fps} min={4} max={30} step={1} onChange={(v) => set("fps", v)} />

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

      {config.use_deforum && (
        <div className={styles.field}>
          <label className={styles.checkLabel}>
            <input
              type="checkbox"
              checked={config.color_coherence}
              onChange={(e) => set("color_coherence", e.target.checked)}
            />
            Color Coherence
          </label>
        </div>
      )}

      <button
        className={styles.generateBtn}
        onClick={handleGenerate}
        disabled={disabled || !config.prompt.trim() || !selectedFile}
      >
        {disabled ? "Generating..." : "Animate Image"}
      </button>
    </div>
  );
}
