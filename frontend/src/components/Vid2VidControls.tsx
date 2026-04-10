import { useState } from "react";
import { Button } from "react-aria-components";
import { Vid2VidConfig, ModelInfo } from "../types";
import SliderField from "./ui/SliderField";
import SelectField from "./ui/SelectField";
import CheckboxField from "./ui/CheckboxField";
import FileTriggerField from "./ui/FileTriggerField";
import SizeSelect from "./SizeSelect";
import styles from "./DeforumControls.module.css";

interface Props {
  config: Vid2VidConfig;
  onChange: (c: Vid2VidConfig) => void;
  onGenerate: (file: File) => void;
  disabled: boolean;
  models: ModelInfo[];
}

export default function Vid2VidControls({ config, onChange, onGenerate, disabled, models }: Props) {
  const [randomSeed, setRandomSeed] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const set = <K extends keyof Vid2VidConfig>(key: K, value: Vid2VidConfig[K]) =>
    onChange({ ...config, [key]: value });

  const handleGenerate = () => {
    if (!selectedFile) return;
    if (randomSeed) onChange({ ...config, seed: null });
    onGenerate(selectedFile);
  };

  const modelItems = models.map((m) => ({ id: m.id, label: m.name }));

  return (
    <div className={styles.panel}>
      <SelectField
        label="Model"
        selectedKey={config.model_id}
        onSelectionChange={(v) => set("model_id", v)}
        items={modelItems}
      />

      <FileTriggerField
        label="Video File"
        accept="video/*"
        selectedFile={selectedFile}
        onSelect={setSelectedFile}
      />

      <div className={styles.section}>
        <label className={styles.fieldLabel}>Prompt</label>
        <textarea
          className={styles.promptArea}
          rows={4}
          value={config.prompt}
          onChange={(e) => set("prompt", e.target.value)}
          placeholder="A fantasy landscape, highly detailed..."
          aria-label="Prompt"
        />
      </div>

      <div className={styles.section}>
        <label className={styles.fieldLabel}>Negative Prompt</label>
        <textarea
          className={styles.negArea}
          rows={2}
          value={config.negative_prompt}
          onChange={(e) => set("negative_prompt", e.target.value)}
          placeholder="blurry, low quality..."
          aria-label="Negative prompt"
        />
      </div>

      <SizeSelect
        width={config.width}
        height={config.height}
        onWidthChange={(v) => set("width", v)}
        onHeightChange={(v) => set("height", v)}
      />

      <SliderField label="Denoising Strength" value={config.denoising_strength} min={0} max={1} step={0.05} onChange={(v) => set("denoising_strength", v)} />
      <SliderField label="Guidance Scale" value={config.guidance_scale} min={1} max={20} step={0.5} onChange={(v) => set("guidance_scale", v)} />
      <SliderField label="Steps" value={config.steps} min={10} max={50} step={1} onChange={(v) => set("steps", v)} />
      <SliderField label="Extraction FPS" value={config.extraction_fps} min={6} max={30} step={1} onChange={(v) => set("extraction_fps", v)} />

      <div className={styles.field}>
        <label className={styles.fieldLabel}>Seed</label>
        <div className={styles.seedRow}>
          <input
            type="number"
            value={config.seed ?? ""}
            disabled={randomSeed}
            onChange={(e) => set("seed", parseInt(e.target.value) || 0)}
            className={styles.numInput}
            placeholder="Random"
            aria-label="Seed value"
          />
          <CheckboxField
            label="Random"
            isSelected={randomSeed}
            onChange={(checked) => {
              setRandomSeed(checked);
              if (checked) set("seed", null);
            }}
          />
        </div>
      </div>

      <Button
        className={styles.generateBtn}
        onPress={handleGenerate}
        isDisabled={disabled || !config.prompt.trim() || !selectedFile}
      >
        {disabled ? "Processing..." : "Process Video"}
      </Button>
    </div>
  );
}
