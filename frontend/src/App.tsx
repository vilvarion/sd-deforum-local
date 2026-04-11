import { useState } from "react";
import { GenerationConfig, Vid2VidConfig, defaultConfig, defaultVid2VidConfig } from "./types";
import { useJobPolling } from "./hooks/useJobPolling";
import { useGenerationActions } from "./hooks/useGenerationActions";
import { useQueuePolling } from "./hooks/useQueuePolling";
import { useGallery } from "./hooks/useGallery";
import UnifiedControls from "./components/UnifiedControls";
import Preview from "./components/Preview";
import Queue from "./components/Queue";
import Gallery from "./components/Gallery";
import Toast from "./components/ui/Toast";
import styles from "./App.module.css";

type Mode = "deforum" | "img2vid" | "vid2vid" | "queue" | "gallery";

const MODE_LABELS: Record<Mode, string> = {
  deforum: "Deforum",
  img2vid: "Img2Vid",
  vid2vid: "Vid2Vid",
  queue: "Queue",
  gallery: "Gallery",
};

const GENERATION_MODES: ReadonlyArray<Mode> = ["deforum", "img2vid", "vid2vid"];

export default function App() {
  const [mode, setMode] = useState<Mode>("deforum");
  const [config, setConfig] = useState<GenerationConfig>({ ...defaultConfig });
  const [vid2vidConfig, setVid2VidConfig] = useState<Vid2VidConfig>({ ...defaultVid2VidConfig });

  const { status, jobId, generating, models, startJob, handleCancel } = useJobPolling();

  const { toast, clearToast, handleGenerate, handleVid2VidGenerate, handleImg2VidGenerate } =
    useGenerationActions({ config, vid2vidConfig, img2vidConfig: config, startJob });

  const queuePolling = useQueuePolling(mode === "queue");
  const gallery = useGallery(mode === "gallery");

  const isGenerationMode = GENERATION_MODES.includes(mode);

  const tabs = (
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
  );

  return (
    <div className={styles.root}>
      <div className={styles.main}>
        {isGenerationMode ? (
          <>
            <div className={styles.sidebar}>
              {tabs}
              <UnifiedControls
                mode={mode as "deforum" | "img2vid" | "vid2vid"}
                config={config}
                onConfigChange={setConfig}
                vid2vidConfig={vid2vidConfig}
                onVid2VidConfigChange={setVid2VidConfig}
                onGenerateDeforum={() => handleGenerate(config)}
                onGenerateImg2Vid={(file) => handleImg2VidGenerate(file, config)}
                onGenerateVid2Vid={(file) => handleVid2VidGenerate(file, vid2vidConfig)}
                disabled={false}
                models={models}
              />
            </div>
            <Preview
              jobId={jobId}
              status={status}
              onCancel={generating ? handleCancel : undefined}
              imageSize={{ width: config.width, height: config.height }}
            />
          </>
        ) : (
          <div className={styles.fullView}>
            {tabs}
            {mode === "queue" && (
              <Queue
                snapshot={queuePolling.snapshot}
                onCancel={queuePolling.cancelItem}
                onSelect={(id, s) => {
                  startJob(id, s);
                  setMode("deforum");
                }}
              />
            )}
            {mode === "gallery" && (
              <Gallery
                items={gallery.items}
                onSelect={(id, s) => {
                  startJob(id, s);
                  setMode("deforum");
                }}
                onDelete={gallery.deleteItem}
              />
            )}
          </div>
        )}
      </div>
      <Toast toast={toast} onDismiss={clearToast} />
    </div>
  );
}
