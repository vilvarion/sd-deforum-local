import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DeforumControls from "../components/DeforumControls";
import { defaultConfig } from "../types";

const models = [{ id: "runwayml/stable-diffusion-v1-5", name: "Stable Diffusion v1.5" }];

describe("DeforumControls", () => {
  it("renders generate button", () => {
    render(
      <DeforumControls
        config={defaultConfig}
        onChange={() => {}}
        onGenerate={() => {}}
        disabled={false}
        models={models}
      />
    );
    expect(screen.getByRole("button", { name: /generate/i })).toBeInTheDocument();
  });

  it("generate button is disabled when prompt is empty", () => {
    render(
      <DeforumControls
        config={{ ...defaultConfig, prompt: "" }}
        onChange={() => {}}
        onGenerate={() => {}}
        disabled={false}
        models={models}
      />
    );
    expect(screen.getByRole("button", { name: /generate/i })).toBeDisabled();
  });

  it("generate button is enabled when prompt has text", () => {
    render(
      <DeforumControls
        config={{ ...defaultConfig, prompt: "A beautiful landscape" }}
        onChange={() => {}}
        onGenerate={() => {}}
        disabled={false}
        models={models}
      />
    );
    expect(screen.getByRole("button", { name: /generate/i })).not.toBeDisabled();
  });

  it("generate button is disabled when generating", () => {
    render(
      <DeforumControls
        config={{ ...defaultConfig, prompt: "A beautiful landscape" }}
        onChange={() => {}}
        onGenerate={() => {}}
        disabled={true}
        models={models}
      />
    );
    expect(screen.getByRole("button", { name: /generating/i })).toBeDisabled();
  });

  it("calls onGenerate when generate button clicked", async () => {
    const onGenerate = vi.fn();
    render(
      <DeforumControls
        config={{ ...defaultConfig, prompt: "A beautiful landscape" }}
        onChange={() => {}}
        onGenerate={onGenerate}
        disabled={false}
        models={models}
      />
    );
    await userEvent.click(screen.getByRole("button", { name: /generate/i }));
    expect(onGenerate).toHaveBeenCalledTimes(1);
  });

  it("renders all slider labels", () => {
    render(
      <DeforumControls
        config={defaultConfig}
        onChange={() => {}}
        onGenerate={() => {}}
        disabled={false}
        models={models}
      />
    );
    expect(screen.getByText("Denoising Strength")).toBeInTheDocument();
    expect(screen.getByText("Guidance Scale")).toBeInTheDocument();
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("Frames")).toBeInTheDocument();
    expect(screen.getByText("FPS")).toBeInTheDocument();
  });
});
