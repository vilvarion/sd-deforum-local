import { render, screen } from "@testing-library/react";
import FileTriggerField from "../components/ui/FileTriggerField";

describe("FileTriggerField", () => {
  it("renders with label", () => {
    render(<FileTriggerField label="Video File" accept="video/*" selectedFile={null} onSelect={() => {}} />);
    expect(screen.getByText("Video File")).toBeInTheDocument();
  });

  it("renders choose file button", () => {
    render(<FileTriggerField label="Video File" accept="video/*" selectedFile={null} onSelect={() => {}} />);
    expect(screen.getByRole("button", { name: "Choose File" })).toBeInTheDocument();
  });

  it("does not show filename when no file selected", () => {
    render(<FileTriggerField label="Video File" accept="video/*" selectedFile={null} onSelect={() => {}} />);
    expect(screen.queryByText(/\.mp4|\.mov|\.avi/i)).not.toBeInTheDocument();
  });

  it("shows filename when file is selected", () => {
    const file = new File(["content"], "test-video.mp4", { type: "video/mp4" });
    render(<FileTriggerField label="Video File" accept="video/*" selectedFile={file} onSelect={() => {}} />);
    expect(screen.getByText("test-video.mp4")).toBeInTheDocument();
  });
});
