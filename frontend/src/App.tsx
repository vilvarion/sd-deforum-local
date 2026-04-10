import { useState } from "react";
import { Tabs, TabList, Tab, TabPanel } from "react-aria-components";
import { GenerationConfig, Vid2VidConfig, defaultConfig, defaultVid2VidConfig } from "./types";
import { useJobPolling } from "./hooks/useJobPolling";
import { useGenerationActions } from "./hooks/useGenerationActions";
import DeforumControls from "./components/DeforumControls";
import Vid2VidControls from "./components/Vid2VidControls";
import Img2VidControls from "./components/Img2VidControls";
import Preview from "./components/Preview";
import Toast from "./components/ui/Toast";
import styles from "./App.module.css";

export default function App() {
  const [config, setConfig] = useState<GenerationConfig>({ ...defaultConfig });
  const [vid2vidConfig, setVid2VidConfig] = useState<Vid2VidConfig>({ ...defaultVid2VidConfig });
  const [img2vidConfig, setImg2VidConfig] = useState<GenerationConfig>({ ...defaultConfig });

  const { status, jobId, generating, models, startJob, handleCancel } = useJobPolling();

  const { toast, clearToast, handleGenerate, handleVid2VidGenerate, handleImg2VidGenerate } =
    useGenerationActions({ config, vid2vidConfig, img2vidConfig, startJob });

  const activeConfig = config;
  const imageSize = { width: activeConfig.width, height: activeConfig.height };

  return (
    <div className={styles.root}>
      <Tabs className={styles.tabs}>
        <header className={styles.header}>
          <span className={styles.title}>Deforum Studio</span>
          <TabList className={styles.tabList} aria-label="Generation mode">
            <Tab id="deforum" className={styles.tab}>Deforum</Tab>
            <Tab id="vid2vid" className={styles.tab}>Vid2Vid</Tab>
            <Tab id="img2vid" className={styles.tab}>Img2Vid</Tab>
          </TabList>
        </header>

        <div className={styles.main}>
          <TabPanel id="deforum" className={styles.tabPanel}>
            <DeforumControls
              config={config}
              onChange={setConfig}
              onGenerate={() => handleGenerate(config)}
              disabled={generating}
              models={models}
            />
          </TabPanel>
          <TabPanel id="vid2vid" className={styles.tabPanel}>
            <Vid2VidControls
              config={vid2vidConfig}
              onChange={setVid2VidConfig}
              onGenerate={(file) => handleVid2VidGenerate(file, vid2vidConfig)}
              disabled={generating}
              models={models}
            />
          </TabPanel>
          <TabPanel id="img2vid" className={styles.tabPanel}>
            <Img2VidControls
              config={img2vidConfig}
              onChange={setImg2VidConfig}
              onGenerate={(file) => handleImg2VidGenerate(file, img2vidConfig)}
              disabled={generating}
              models={models}
            />
          </TabPanel>
          <Preview
            jobId={jobId}
            status={status}
            onCancel={generating ? handleCancel : undefined}
            imageSize={imageSize}
          />
        </div>
      </Tabs>

      <Toast toast={toast} onDismiss={clearToast} />
    </div>
  );
}
