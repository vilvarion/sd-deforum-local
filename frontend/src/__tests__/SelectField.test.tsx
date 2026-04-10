import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SelectField from "../components/ui/SelectField";

const items = [
  { id: "a", label: "Option A" },
  { id: "b", label: "Option B" },
  { id: "c", label: "Option C" },
];

describe("SelectField", () => {
  it("renders with label", () => {
    render(<SelectField label="Model" selectedKey="a" onSelectionChange={() => {}} items={items} />);
    expect(screen.getByText("Model")).toBeInTheDocument();
  });

  it("shows current selected value in the button", () => {
    render(<SelectField label="Model" selectedKey="b" onSelectionChange={() => {}} items={items} />);
    // React Aria renders a hidden native select alongside the custom one; query within the button
    const button = screen.getByRole("button");
    expect(button).toHaveTextContent("Option B");
  });

  it("opens listbox on button click", async () => {
    render(<SelectField label="Model" selectedKey="a" onSelectionChange={() => {}} items={items} />);
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("listbox")).toBeInTheDocument();
  });

  it("calls onSelectionChange when option selected", async () => {
    const onSelectionChange = vi.fn();
    render(<SelectField label="Model" selectedKey="a" onSelectionChange={onSelectionChange} items={items} />);
    await userEvent.click(screen.getByRole("button"));
    await userEvent.click(screen.getByRole("option", { name: "Option C" }));
    expect(onSelectionChange).toHaveBeenCalledWith("c");
  });

  it("converts string key to number for numeric items", async () => {
    const numItems = [{ id: 512, label: "512" }, { id: 768, label: "768" }];
    const onSelectionChange = vi.fn();
    render(<SelectField label="Width" selectedKey={512} onSelectionChange={onSelectionChange} items={numItems} />);
    await userEvent.click(screen.getByRole("button"));
    await userEvent.click(screen.getByRole("option", { name: "768" }));
    expect(onSelectionChange).toHaveBeenCalledWith(768);
  });
});
