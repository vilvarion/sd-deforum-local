import SelectField from "./ui/SelectField";
import styles from "./DeforumControls.module.css";

interface Props {
  width: number;
  height: number;
  onWidthChange: (v: number) => void;
  onHeightChange: (v: number) => void;
}

const SIZES = [256, 384, 512, 768];
const SIZE_ITEMS = SIZES.map((s) => ({ id: s, label: String(s) }));

export default function SizeSelect({ width, height, onWidthChange, onHeightChange }: Props) {
  return (
    <div className={styles.sizeGrid}>
      <SelectField
        label="Width"
        selectedKey={width}
        onSelectionChange={onWidthChange}
        items={SIZE_ITEMS}
      />
      <SelectField
        label="Height"
        selectedKey={height}
        onSelectionChange={onHeightChange}
        items={SIZE_ITEMS}
      />
    </div>
  );
}
