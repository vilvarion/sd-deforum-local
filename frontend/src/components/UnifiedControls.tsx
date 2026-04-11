import { useEffect, useRef, useState } from "react";
import { Button } from "react-aria-components";
import { GenerationConfig, PromptKeyframe, Vid2VidConfig, ModelInfo } from "../types";
import SliderField from "./ui/SliderField";
import SelectField from "./ui/SelectField";
import CheckboxField from "./ui/CheckboxField";
import FileTriggerField from "./ui/FileTriggerField";
import SizeSelect from "./SizeSelect";
import PromptSchedule, { validatePromptSchedule } from "./PromptSchedule";
import styles from "./DeforumControls.module.css";

type Mode = "deforum" | "img2vid" | "vid2vid";

type SharedKey =
  | "prompt"
  | "negative_prompt"
  | "width"
  | "height"
  | "denoising_strength"
  | "guidance_scale"
  | "steps"
  | "seed"
  | "model_id";

interface Props {
  mode: Mode;
  config: GenerationConfig;
  onConfigChange: (c: GenerationConfig) => void;
  vid2vidConfig: Vid2VidConfig;
  onVid2VidConfigChange: (c: Vid2VidConfig) => void;
  onGenerateDeforum: () => void;
  onGenerateImg2Vid: (file: File) => void;
  onGenerateVid2Vid: (file: File) => void;
  disabled: boolean;
  models: ModelInfo[];
}

export default function UnifiedControls({
  mode,
  config,
  onConfigChange,
  vid2vidConfig,
  onVid2VidConfigChange,
  onGenerateDeforum,
  onGenerateImg2Vid,
  onGenerateVid2Vid,
  disabled,
  models,
}: Props) {
  const [randomSeed, setRandomSeed] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const prevModeRef = useRef<Mode>(mode);

  useEffect(() => {
    const prev = prevModeRef.current;
    prevModeRef.current = mode;
    if (
      (prev === "img2vid" && mode === "vid2vid") ||
      (prev === "vid2vid" && mode === "img2vid") ||
      (prev !== "deforum" && mode === "deforum")
    ) {
      setSelectedFile(null);
    }
  }, [mode]);

  const setShared = <K extends SharedKey>(key: K, value: GenerationConfig[K]) => {
    onConfigChange({ ...config, [key]: value });
    onVid2VidConfigChange({ ...vid2vidConfig, [key]: value as unknown as Vid2VidConfig[K] });
  };

  const setConfig = <K extends keyof GenerationConfig>(key: K, value: GenerationConfig[K]) =>
    onConfigChange({ ...config, [key]: value });

  const setVid2Vid = <K extends keyof Vid2VidConfig>(key: K, value: Vid2VidConfig[K]) =>
    onVid2VidConfigChange({ ...vid2vidConfig, [key]: value });

  const handleGenerate = () => {
    if (randomSeed) {
      if (mode !== "vid2vid") onConfigChange({ ...config, seed: null });
      else onVid2VidConfigChange({ ...vid2vidConfig, seed: null });
    }
    if (mode === "deforum") {
      onGenerateDeforum();
    } else if (mode === "img2vid" && selectedFile) {
      onGenerateImg2Vid(selectedFile);
    } else if (mode === "vid2vid" && selectedFile) {
      onGenerateVid2Vid(selectedFile);
    }
  };

  const modelItems = models.map((m) => ({ id: m.id, label: m.name }));
  const activePrompt = mode !== "vid2vid" ? config.prompt : vid2vidConfig.prompt;
  const needsFile = mode !== "deforum";
  const scheduleValid =
    mode !== "deforum" || validatePromptSchedule(config.prompt_schedule, config.num_frames);
  const isDisabled =
    disabled ||
    !activePrompt.trim() ||
    (needsFile && !selectedFile) ||
    !scheduleValid;

  const btnLabel = disabled
    ? mode === "vid2vid" ? "Processing..." : "Generating..."
    : mode === "deforum" ? "Generate"
    : mode === "img2vid" ? "Animate Image"
    : "Process Video";

  return (
    <div className={styles.panel}>
      <SelectField
        label="Model"
        selectedKey={config.model_id}
        onSelectionChange={(v) => setShared("model_id", v as string)}
        items={modelItems}
      />

      {needsFile && (
        <FileTriggerField
          label={mode === "img2vid" ? "Image File" : "Video File"}
          accept={mode === "img2vid" ? "image/*" : "video/*"}
          selectedFile={selectedFile}
          onSelect={setSelectedFile}
        />
      )}

      <div className={styles.section}>
        <label className={styles.fieldLabel}>Prompt</label>
        <textarea
          className={styles.promptArea}
          rows={3}
          value={mode !== "vid2vid" ? config.prompt : vid2vidConfig.prompt}
          onChange={(e) => setShared("prompt", e.target.value)}
          placeholder="A fantasy landscape, highly detailed..."
          aria-label="Prompt"
        />
      </div>

      {mode === "deforum" && (
        <PromptSchedule
          schedule={config.prompt_schedule}
          numFrames={config.num_frames}
          onChange={(schedule: PromptKeyframe[]) =>
            setConfig("prompt_schedule", schedule)
          }
          disabled={disabled}
        />
      )}

      <div className={styles.section}>
        <label className={styles.fieldLabel}>Negative Prompt</label>
        <textarea
          className={styles.negArea}
          rows={2}
          value={mode !== "vid2vid" ? config.negative_prompt : vid2vidConfig.negative_prompt}
          onChange={(e) => setShared("negative_prompt", e.target.value)}
          placeholder="blurry, low quality..."
          aria-label="Negative prompt"
        />
      </div>

      <SizeSelect
        width={config.width}
        height={config.height}
        onWidthChange={(v) => setShared("width", v)}
        onHeightChange={(v) => setShared("height", v)}
      />

      <div className={styles.sliderGrid}>
        <SliderField
          label="Denoising"
          value={mode !== "vid2vid" ? config.denoising_strength : vid2vidConfig.denoising_strength}
          min={0} max={1} step={0.05}
          onChange={(v) => setShared("denoising_strength", v)}
        />
        <SliderField
          label="CFG Scale"
          value={mode !== "vid2vid" ? config.guidance_scale : vid2vidConfig.guidance_scale}
          min={1} max={20} step={0.5}
          onChange={(v) => setShared("guidance_scale", v)}
        />
        <SliderField
          label="Steps"
          value={mode !== "vid2vid" ? config.steps : vid2vidConfig.steps}
          min={10} max={50} step={1}
          onChange={(v) => setShared("steps", v)}
        />
        {mode !== "vid2vid" ? (
          <SliderField
            label="Frames"
            value={config.num_frames}
            min={2} max={30} step={1}
            onChange={(v) => setConfig("num_frames", v)}
          />
        ) : (
          <SliderField
            label="Extract FPS"
            value={vid2vidConfig.extraction_fps}
            min={6} max={30} step={1}
            onChange={(v) => setVid2Vid("extraction_fps", v)}
          />
        )}
        {mode !== "vid2vid" && (
          <SliderField
            label="FPS"
            value={config.fps}
            min={4} max={30} step={1}
            onChange={(v) => setConfig("fps", v)}
          />
        )}
      </div>

      {mode !== "vid2vid" && (
        <CheckboxField
          label="Use Deforum"
          isSelected={config.use_deforum}
          onChange={(v) => setConfig("use_deforum", v)}
        />
      )}

      {mode !== "vid2vid" && config.use_deforum && (
        <div className={styles.sliderGrid}>
          <SliderField label="Zoom / Frame" value={config.zoom_per_frame} min={0.9} max={1.1} step={0.01} onChange={(v) => setConfig("zoom_per_frame", v)} />
          <SliderField label="Rotate / Frame" value={config.rotate_per_frame} min={-5} max={5} step={0.5} onChange={(v) => setConfig("rotate_per_frame", v)} />
          <SliderField label="Translate X" value={config.translate_x} min={-20} max={20} step={1} onChange={(v) => setConfig("translate_x", v)} />
          <SliderField label="Translate Y" value={config.translate_y} min={-20} max={20} step={1} onChange={(v) => setConfig("translate_y", v)} />
        </div>
      )}

      <div className={styles.field}>
        <div className={styles.seedRow}>
          <label className={styles.fieldLabel}>Seed</label>
          <input
            type="number"
            value={mode !== "vid2vid" ? (config.seed ?? "") : (vid2vidConfig.seed ?? "")}
            disabled={randomSeed}
            onChange={(e) => setShared("seed", parseInt(e.target.value) || 0)}
            className={styles.numInput}
            placeholder="Random"
            aria-label="Seed value"
          />
          <CheckboxField
            label="Random"
            isSelected={randomSeed}
            onChange={(checked) => {
              setRandomSeed(checked);
              if (checked) setShared("seed", null);
            }}
          />
        </div>
      </div>

      <Button
        className={styles.generateBtn}
        onPress={handleGenerate}
        isDisabled={isDisabled}
      >
        {btnLabel}
      </Button>
    </div>
  );
}
