import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CheckboxField from "../components/ui/CheckboxField";

describe("CheckboxField", () => {
  it("renders with label", () => {
    render(<CheckboxField label="Color Coherence" isSelected={false} onChange={() => {}} />);
    expect(screen.getByText("Color Coherence")).toBeInTheDocument();
  });

  it("has correct ARIA role and checked state", () => {
    render(<CheckboxField label="Color Coherence" isSelected={true} onChange={() => {}} />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeChecked();
  });

  it("calls onChange with true when unchecked and clicked", async () => {
    const onChange = vi.fn();
    render(<CheckboxField label="Color Coherence" isSelected={false} onChange={onChange} />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("calls onChange with false when checked and clicked", async () => {
    const onChange = vi.fn();
    render(<CheckboxField label="Color Coherence" isSelected={true} onChange={onChange} />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it("does not call onChange when disabled", async () => {
    const onChange = vi.fn();
    render(<CheckboxField label="Color Coherence" isSelected={false} onChange={onChange} isDisabled />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(onChange).not.toHaveBeenCalled();
  });
});
