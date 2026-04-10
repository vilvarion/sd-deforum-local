import {
  Select,
  Label,
  Button,
  SelectValue,
  Popover,
  ListBox,
  ListBoxItem,
} from "react-aria-components";
import styles from "./SelectField.module.css";

interface SelectItem<T extends string | number> {
  id: T;
  label: string;
}

interface SelectFieldProps<T extends string | number> {
  label: string;
  selectedKey: T;
  onSelectionChange: (key: T) => void;
  items: SelectItem<T>[];
  "aria-label"?: string;
}

export default function SelectField<T extends string | number>({
  label,
  selectedKey,
  onSelectionChange,
  items,
  "aria-label": ariaLabel,
}: SelectFieldProps<T>) {
  return (
    <Select
      selectedKey={String(selectedKey)}
      onSelectionChange={(key) => {
        const raw = key as string;
        const parsed = typeof selectedKey === "number" ? (Number(raw) as T) : (raw as T);
        onSelectionChange(parsed);
      }}
      aria-label={ariaLabel ?? label}
      className={styles.select}
    >
      <Label className={styles.label}>{label}</Label>
      <Button className={styles.trigger}>
        <SelectValue className={styles.value} />
        <span aria-hidden className={styles.chevron}>▾</span>
      </Button>
      <Popover className={styles.popover}>
        <ListBox className={styles.listbox}>
          {items.map((item) => (
            <ListBoxItem
              key={String(item.id)}
              id={String(item.id)}
              className={styles.option}
            >
              {item.label}
            </ListBoxItem>
          ))}
        </ListBox>
      </Popover>
    </Select>
  );
}
