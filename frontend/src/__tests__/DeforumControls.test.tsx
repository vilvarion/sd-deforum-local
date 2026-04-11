import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import UnifiedControls from "../components/UnifiedControls";
import { defaultConfig, defaultVid2VidConfig } from "../types";

const models = [{ id: "runwayml/stable-diffusion-v1-5", name: "Stable Diffusion v1.5" }];

const baseProps = {
  mode: "deforum" as const,
  config: defaultConfig,
  onConfigChange: () => {},
  vid2vidConfig: defaultVid2VidConfig,
  onVid2VidConfigChange: () => {},
  onGenerateDeforum: () => {},
  onGenerateImg2Vid: () => {},
  onGenerateVid2Vid: () => {},
  disabled: false,
  models,
};

describe("UnifiedControls", () => {
  it("renders generate button", () => {
    render(<UnifiedControls {...baseProps} />);
    expect(screen.getByRole("button", { name: /generate/i })).toBeInTheDocument();
  });

  it("generate button is disabled when prompt is empty", () => {
    render(<UnifiedControls {...baseProps} config={{ ...defaultConfig, prompt: "" }} />);
    expect(screen.getByRole("button", { name: /generate/i })).toBeDisabled();
  });

  it("generate button is enabled when prompt has text", () => {
    render(<UnifiedControls {...baseProps} config={{ ...defaultConfig, prompt: "A beautiful landscape" }} />);
    expect(screen.getByRole("button", { name: /generate/i })).not.toBeDisabled();
  });

  it("generate button is disabled when generating", () => {
    render(<UnifiedControls {...baseProps} config={{ ...defaultConfig, prompt: "A beautiful landscape" }} disabled={true} />);
    expect(screen.getByRole("button", { name: /generating/i })).toBeDisabled();
  });

  it("calls onGenerateDeforum when generate button clicked", async () => {
    const onGenerateDeforum = vi.fn();
    render(<UnifiedControls {...baseProps} config={{ ...defaultConfig, prompt: "A beautiful landscape" }} onGenerateDeforum={onGenerateDeforum} />);
    await userEvent.click(screen.getByRole("button", { name: /generate/i }));
    expect(onGenerateDeforum).toHaveBeenCalledTimes(1);
  });

  it("renders core slider labels", () => {
    render(<UnifiedControls {...baseProps} />);
    expect(screen.getByText("Denoising")).toBeInTheDocument();
    expect(screen.getByText("CFG Scale")).toBeInTheDocument();
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("Frames")).toBeInTheDocument();
    expect(screen.getByText("FPS")).toBeInTheDocument();
  });
});
