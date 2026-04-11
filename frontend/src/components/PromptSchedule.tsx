import { Button } from "react-aria-components";
import { PromptKeyframe } from "../types";
import styles from "./PromptSchedule.module.css";

interface Props {
  schedule: PromptKeyframe[];
  numFrames: number;
  onChange: (schedule: PromptKeyframe[]) => void;
  disabled?: boolean;
}

const BLEND_OPTIONS = [0, 4, 8, 16];

function validate(schedule: PromptKeyframe[], numFrames: number): string[] {
  const sorted = [...schedule].sort((a, b) => a.frame - b.frame);
  const errors: string[] = [];
  const seen = new Set<number>();
  let prevFrame = 0;
  for (const kf of sorted) {
    if (!kf.prompt.trim()) {
      errors.push(`Keyframe at frame ${kf.frame} needs a prompt`);
    }
    if (kf.frame < 1 || kf.frame >= numFrames) {
      errors.push(`Frame ${kf.frame} must be between 1 and ${numFrames - 1}`);
    }
    if (seen.has(kf.frame)) {
      errors.push(`Duplicate keyframe at frame ${kf.frame}`);
    }
    const maxBlend = kf.frame - prevFrame;
    if (kf.blend_frames > maxBlend) {
      errors.push(
        `Frame ${kf.frame}: blend of ${kf.blend_frames} exceeds available gap (${maxBlend})`,
      );
    }
    seen.add(kf.frame);
    prevFrame = kf.frame;
  }
  return errors;
}

export function validatePromptSchedule(
  schedule: PromptKeyframe[],
  numFrames: number,
): boolean {
  return validate(schedule, numFrames).length === 0;
}

export default function PromptSchedule({
  schedule,
  numFrames,
  onChange,
  disabled,
}: Props) {
  const errors = validate(schedule, numFrames);
  const sorted = [...schedule].sort((a, b) => a.frame - b.frame);

  const addKeyframe = () => {
    const lastFrame = sorted.length > 0 ? sorted[sorted.length - 1].frame : 0;
    const nextFrame = Math.min(numFrames - 1, Math.max(1, lastFrame + 1));
    onChange([
      ...schedule,
      { frame: nextFrame, prompt: "", blend_frames: Math.min(8, nextFrame) },
    ]);
  };

  const updateRow = (index: number, patch: Partial<PromptKeyframe>) => {
    const next = schedule.map((kf, i) => (i === index ? { ...kf, ...patch } : kf));
    onChange(next);
  };

  const removeRow = (index: number) => {
    onChange(schedule.filter((_, i) => i !== index));
  };

  const maxFrameIndex = Math.max(1, numFrames - 1);
  const pct = (frame: number) => (frame / maxFrameIndex) * 100;

  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        <label className={styles.label}>Prompt Schedule</label>
        <Button
          className={styles.addBtn}
          onPress={addKeyframe}
          isDisabled={disabled || numFrames < 2}
        >
          + Add keyframe
        </Button>
      </div>

      <div className={styles.rows}>
        {schedule.length === 0 && (
          <div className={styles.empty}>
            No keyframes — the main prompt is used for every frame.
          </div>
        )}
        {schedule.map((kf, i) => {
          const frameInvalid = kf.frame < 1 || kf.frame >= numFrames;
          return (
            <div key={i} className={styles.row}>
              <input
                type="number"
                value={kf.frame}
                min={1}
                max={numFrames - 1}
                onChange={(e) =>
                  updateRow(i, { frame: parseInt(e.target.value) || 0 })
                }
                disabled={disabled}
                aria-label="Keyframe frame index"
                className={frameInvalid ? styles.invalid : undefined}
              />
              <textarea
                rows={2}
                value={kf.prompt}
                onChange={(e) => updateRow(i, { prompt: e.target.value })}
                placeholder="e.g. a photorealistic skeleton in a forest"
                disabled={disabled}
                aria-label="Keyframe prompt"
                className={!kf.prompt.trim() ? styles.invalid : undefined}
              />
              <select
                value={kf.blend_frames}
                onChange={(e) =>
                  updateRow(i, { blend_frames: parseInt(e.target.value) })
                }
                disabled={disabled}
                aria-label="Blend frames"
              >
                {BLEND_OPTIONS.map((v) => (
                  <option key={v} value={v}>
                    {v === 0 ? "cut" : `${v}f`}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className={styles.removeBtn}
                onClick={() => removeRow(i)}
                disabled={disabled}
                aria-label="Remove keyframe"
                title="Remove keyframe"
              >
                ×
              </button>
            </div>
          );
        })}
      </div>

      {schedule.length > 0 && (
        <div className={styles.ruler} aria-hidden="true">
          <div className={styles.rulerTrack} />
          <div
            className={`${styles.pin} ${styles.pinMain}`}
            style={{ left: "0%" }}
          />
          {sorted.map((kf, i) => {
            const bandStart = Math.max(0, kf.frame - kf.blend_frames);
            const bandLeft = pct(bandStart);
            const bandWidth = pct(kf.frame) - bandLeft;
            return (
              <div key={i}>
                {kf.blend_frames > 0 && (
                  <div
                    className={styles.blendBand}
                    style={{ left: `${bandLeft}%`, width: `${bandWidth}%` }}
                  />
                )}
                <div className={styles.pin} style={{ left: `${pct(kf.frame)}%` }} />
              </div>
            );
          })}
        </div>
      )}

      {errors.length > 0 && (
        <div className={styles.hint}>{errors[0]}</div>
      )}
    </div>
  );
}
