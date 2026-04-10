import { Checkbox } from "react-aria-components";
import styles from "./CheckboxField.module.css";

interface CheckboxFieldProps {
  label: string;
  isSelected: boolean;
  onChange: (checked: boolean) => void;
  isDisabled?: boolean;
}

export default function CheckboxField({ label, isSelected, onChange, isDisabled }: CheckboxFieldProps) {
  return (
    <Checkbox
      isSelected={isSelected}
      onChange={onChange}
      isDisabled={isDisabled}
      className={styles.checkbox}
    >
      <div className={styles.box} aria-hidden>
        {isSelected && <span className={styles.checkmark}>✓</span>}
      </div>
      <span className={styles.label}>{label}</span>
    </Checkbox>
  );
}
