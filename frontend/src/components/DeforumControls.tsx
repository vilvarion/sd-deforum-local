import { useState } from "react";
import { Button } from "react-aria-components";
import { GenerationConfig, ModelInfo } from "../types";
import SliderField from "./ui/SliderField";
import SelectField from "./ui/SelectField";
import CheckboxField from "./ui/CheckboxField";
import SizeSelect from "./SizeSelect";
import styles from "./DeforumControls.module.css";

interface Props {
  config: GenerationConfig;
  onChange: (c: GenerationConfig) => void;
  onGenerate: () => void;
  disabled: boolean;
  models: ModelInfo[];
}

export default function DeforumControls({ config, onChange, onGenerate, disabled, models }: Props) {
  const [randomSeed, setRandomSeed] = useState(true);

  const set = <K extends keyof GenerationConfig>(key: K, value: GenerationConfig[K]) =>
    onChange({ ...config, [key]: value });

  const handleGenerate = () => {
    if (randomSeed) onChange({ ...config, seed: null });
    onGenerate();
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

      <div className={styles.section}>
        <label className={styles.fieldLabel}>Prompt</label>
        <textarea
          className={styles.promptArea}
          rows={3}
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

      <div className={styles.sliderGrid}>
        <SliderField label="Denoising" value={config.denoising_strength} min={0} max={1} step={0.05} onChange={(v) => set("denoising_strength", v)} />
        <SliderField label="CFG Scale" value={config.guidance_scale} min={1} max={20} step={0.5} onChange={(v) => set("guidance_scale", v)} />
        <SliderField label="Steps" value={config.steps} min={10} max={50} step={1} onChange={(v) => set("steps", v)} />
        <SliderField label="Frames" value={config.num_frames} min={2} max={30} step={1} onChange={(v) => set("num_frames", v)} />
        <SliderField label="Zoom / Frame" value={config.zoom_per_frame} min={0.9} max={1.1} step={0.01} onChange={(v) => set("zoom_per_frame", v)} />
        <SliderField label="Rotate / Frame" value={config.rotate_per_frame} min={-5} max={5} step={0.5} onChange={(v) => set("rotate_per_frame", v)} />
        <SliderField label="Translate X" value={config.translate_x} min={-20} max={20} step={1} onChange={(v) => set("translate_x", v)} />
        <SliderField label="Translate Y" value={config.translate_y} min={-20} max={20} step={1} onChange={(v) => set("translate_y", v)} />
        <SliderField label="FPS" value={config.fps} min={4} max={30} step={1} onChange={(v) => set("fps", v)} />
      </div>

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

      <CheckboxField
        label="Color Coherence"
        isSelected={config.color_coherence}
        onChange={(v) => set("color_coherence", v)}
      />

      <Button
        className={styles.generateBtn}
        onPress={handleGenerate}
        isDisabled={disabled || !config.prompt.trim()}
      >
        {disabled ? "Generating..." : "Generate"}
      </Button>
    </div>
  );
}
