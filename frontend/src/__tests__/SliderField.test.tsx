import React, { useState } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SliderField from "../components/ui/SliderField";

function StatefulSlider(props: { min: number; max: number; step: number; initial: number; onChange?: (v: number) => void }) {
  const [value, setValue] = useState(props.initial);
  return (
    <SliderField
      label="Steps"
      value={value}
      min={props.min}
      max={props.max}
      step={props.step}
      onChange={(v) => {
        setValue(v);
        props.onChange?.(v);
      }}
    />
  );
}

describe("SliderField", () => {
  it("renders with label", () => {
    render(<SliderField label="Denoising Strength" value={0.5} min={0} max={1} step={0.05} onChange={() => {}} />);
    expect(screen.getByText("Denoising Strength")).toBeInTheDocument();
  });

  it("number input shows current value", () => {
    render(<SliderField label="Steps" value={25} min={1} max={50} step={1} onChange={() => {}} />);
    expect(screen.getByRole("spinbutton")).toHaveValue(25);
  });

  it("calls onChange when number input changes", () => {
    const onChange = vi.fn();
    render(<SliderField label="Steps" value={25} min={1} max={50} step={1} onChange={onChange} />);
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "30" } });
    expect(onChange).toHaveBeenLastCalledWith(30);
  });

  it("clamps number input to max", () => {
    const onChange = vi.fn();
    render(<SliderField label="Steps" value={25} min={1} max={50} step={1} onChange={onChange} />);
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "999" } });
    expect(onChange).toHaveBeenLastCalledWith(50);
  });

  it("clamps number input to min", () => {
    const onChange = vi.fn();
    render(<SliderField label="Steps" value={25} min={10} max={50} step={1} onChange={onChange} />);
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "1" } });
    expect(onChange).toHaveBeenLastCalledWith(10);
  });

  it("slider element is present in the DOM", () => {
    render(<SliderField label="Guidance Scale" value={7.5} min={1} max={20} step={0.5} onChange={() => {}} />);
    expect(screen.getByRole("slider")).toBeInTheDocument();
  });
});
