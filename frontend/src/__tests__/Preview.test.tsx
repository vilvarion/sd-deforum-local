import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Preview from "../components/Preview";
import { JobStatus } from "../types";

const runningStatus: JobStatus = {
  status: "running",
  current_frame: 5,
  total_frames: 15,
  current_step: 10,
  total_steps: 25,
  error_message: "",
};

const doneStatus: JobStatus = {
  status: "done",
  current_frame: 15,
  total_frames: 15,
  current_step: 25,
  total_steps: 25,
  error_message: "",
};

const errorStatus: JobStatus = {
  status: "error",
  current_frame: 3,
  total_frames: 15,
  current_step: 0,
  total_steps: 0,
  error_message: "CUDA out of memory",
};

describe("Preview", () => {
  it("shows empty state when no jobId", () => {
    render(<Preview jobId={null} status={null} />);
    expect(screen.getByText(/configure settings/i)).toBeInTheDocument();
  });

  it("shows error message when status is error", () => {
    render(<Preview jobId="job-123" status={errorStatus} />);
    expect(screen.getByRole("alert")).toHaveTextContent("CUDA out of memory");
  });

  it("shows progress bar with correct ARIA when running", () => {
    render(<Preview jobId="job-123" status={runningStatus} />);
    const progressBars = screen.getAllByRole("progressbar");
    expect(progressBars.length).toBeGreaterThan(0);
    expect(progressBars[0]).toHaveAttribute("aria-valuenow", "33");
  });

  it("shows cancel button when onCancel provided", () => {
    render(<Preview jobId="job-123" status={runningStatus} onCancel={() => {}} />);
    expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
  });

  it("calls onCancel when cancel button clicked", async () => {
    const onCancel = vi.fn();
    render(<Preview jobId="job-123" status={runningStatus} onCancel={onCancel} />);
    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("does not show cancel button without onCancel prop", () => {
    render(<Preview jobId="job-123" status={runningStatus} />);
    expect(screen.queryByRole("button", { name: /cancel/i })).not.toBeInTheDocument();
  });

  it("shows done status without progress bar", () => {
    render(<Preview jobId="job-123" status={doneStatus} />);
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });
});
