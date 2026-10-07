"use client";
import { useState, type FormEvent } from "react";
import { Button } from "@/components/common/ui";
import { SelectField, TextField } from "@/components/common/fields";
import { validateBacktestForm, type BacktestFormValues, type FieldErrors } from "@/lib/validation";
import type { BacktestRequest, RebalanceFrequency } from "@/lib/types";

export const BACKTEST_DEFAULTS: BacktestFormValues = { capital: "100000", basketSize: "10", threshold: "0.90", frequency: "quarterly", startDate: "", endDate: "" };

const FREQUENCIES: Array<{ value: RebalanceFrequency; label: string }> = [
  { value: "monthly", label: "Monthly" },
  { value: "quarterly", label: "Quarterly" },
  { value: "semiannual", label: "Semi-annual" },
  { value: "annual", label: "Annual" },
];

interface Props {
  onSubmit: (payload: BacktestRequest) => void;
  onCancel?: () => void;
  loading?: boolean;
  elapsed?: number;
  maxSize?: number;
  /** Dataset date range (from the API), used as the limits of the date inputs. */
  minDate?: string;
  maxDate?: string;
  serverErrors?: FieldErrors;
}

export function BacktestForm({ onSubmit, onCancel, loading = false, elapsed = 0, maxSize, minDate, maxDate, serverErrors }: Props) {
  const [values, setValues] = useState<BacktestFormValues>(BACKTEST_DEFAULTS);
  const [clientErrors, setClientErrors] = useState<FieldErrors>({});
  const errors = { ...serverErrors, ...clientErrors };
  const set = (key: keyof BacktestFormValues) => (event: { target: { value: string } }) => {
    setValues((v) => ({ ...v, [key]: event.target.value }));
    setClientErrors((e) => ({ ...e, [key]: "" }));
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const result = validateBacktestForm(values, maxSize);
    setClientErrors(result.errors);
    if (result.payload) onSubmit(result.payload);
  };

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <TextField id="bt-start" label="Start date" type="date" min={minDate} max={maxDate} value={values.startDate} onChange={set("startDate")} error={errors.startDate} hint="Optional. Blank uses the start of the data." />
        <TextField id="bt-end" label="End date" type="date" min={minDate} max={maxDate} value={values.endDate} onChange={set("endDate")} error={errors.endDate} hint="Optional. Blank uses the end of the data." />
        <TextField id="bt-capital" label="Capital (₹)" type="number" inputMode="decimal" min="1" step="any" value={values.capital} onChange={set("capital")} error={errors.capital} />
        <TextField id="bt-size" label="Basket size" type="number" inputMode="numeric" step="1" value={values.basketSize} onChange={set("basketSize")} error={errors.basketSize} hint={maxSize ? `3 to ${maxSize} stocks` : undefined} />
        <SelectField id="bt-frequency" label="Rebalance frequency" value={values.frequency} onChange={set("frequency")} error={errors.frequency}>
          {FREQUENCIES.map((f) => (
            <option key={f.value} value={f.value}>
              {f.label}
            </option>
          ))}
        </SelectField>
        <TextField id="bt-threshold" label="Similarity threshold" type="number" inputMode="decimal" step="0.01" value={values.threshold} onChange={set("threshold")} error={errors.threshold} hint="Between -1 and 1." />
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" disabled={loading}>
          {loading ? "Running backtest..." : "Run Backtest"}
        </Button>
        {loading && onCancel && (
          <Button type="button" variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
        )}
        {loading && (
          <span role="status" aria-live="polite" className="text-sm text-ink-2">
            Running walk-forward backtest... <span className="num">{elapsed}s</span>. This usually takes 10 to 20 seconds.
          </span>
        )}
      </div>
    </form>
  );
}
