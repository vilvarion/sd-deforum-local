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
import Tabs, { Mode, GENERATION_MODES } from "./components/Tabs";
import Toast from "./components/ui/Toast";
import styles from "./App.module.css";

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

  return (
    <div className={styles.root}>
      <nav className={styles.tabNav}>
        <Tabs mode={mode} onChange={setMode} />
      </nav>
      <div className={styles.main}>
        {isGenerationMode ? (
          <>
            <div className={styles.sidebar}>
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
