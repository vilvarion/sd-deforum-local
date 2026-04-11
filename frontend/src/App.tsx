import { useState } from "react";
import { GenerationConfig, Vid2VidConfig, defaultConfig, defaultVid2VidConfig } from "./types";
import { useJobPolling } from "./hooks/useJobPolling";
import { useGenerationActions } from "./hooks/useGenerationActions";
import UnifiedControls from "./components/UnifiedControls";
import Preview from "./components/Preview";
import Toast from "./components/ui/Toast";
import styles from "./App.module.css";

type Mode = "deforum" | "img2vid" | "vid2vid";

const MODE_LABELS: Record<Mode, string> = {
  deforum: "Deforum",
  img2vid: "Img2Vid",
  vid2vid: "Vid2Vid",
};

export default function App() {
  const [mode, setMode] = useState<Mode>("deforum");
  const [config, setConfig] = useState<GenerationConfig>({ ...defaultConfig });
  const [vid2vidConfig, setVid2VidConfig] = useState<Vid2VidConfig>({ ...defaultVid2VidConfig });

  const { status, jobId, generating, models, startJob, handleCancel } = useJobPolling();

  const { toast, clearToast, handleGenerate, handleVid2VidGenerate, handleImg2VidGenerate } =
    useGenerationActions({ config, vid2vidConfig, img2vidConfig: config, startJob });

  return (
    <div className={styles.root}>
      <div className={styles.main}>
        <div className={styles.sidebar}>
          <div className={styles.tabList} role="tablist">
            {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
              <button
                key={m}
                role="tab"
                aria-selected={mode === m}
                className={`${styles.tab}${mode === m ? ` ${styles.tabSelected}` : ""}`}
                onClick={() => setMode(m)}
              >
                {MODE_LABELS[m]}
              </button>
            ))}
          </div>
          <UnifiedControls
            mode={mode}
            config={config}
            onConfigChange={setConfig}
            vid2vidConfig={vid2vidConfig}
            onVid2VidConfigChange={setVid2VidConfig}
            onGenerateDeforum={() => handleGenerate(config)}
            onGenerateImg2Vid={(file) => handleImg2VidGenerate(file, config)}
            onGenerateVid2Vid={(file) => handleVid2VidGenerate(file, vid2vidConfig)}
            disabled={generating}
            models={models}
          />
        </div>
        <Preview
          jobId={jobId}
          status={status}
          onCancel={generating ? handleCancel : undefined}
          imageSize={{ width: config.width, height: config.height }}
        />
      </div>
      <Toast toast={toast} onDismiss={clearToast} />
    </div>
  );
}
