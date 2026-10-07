"use client";
import { useState, type FormEvent } from "react";
import { Button, Notice } from "@/components/common/ui";
import { TextField } from "@/components/common/fields";
import { validateBasketForm, type BasketFormValues, type FieldErrors } from "@/lib/validation";
import type { BasketRequest } from "@/lib/types";

export const BASKET_DEFAULTS: BasketFormValues = { capital: "50000", basketSize: "6", threshold: "0.90" };

interface Props {
  onSubmit: (payload: BasketRequest) => void;
  loading?: boolean;
  /** Number of stocks in the universe (from the API). Basket size cannot exceed it. */
  maxSize?: number;
  serverErrors?: FieldErrors;
}

export function BasketForm({ onSubmit, loading = false, maxSize, serverErrors }: Props) {
  const [values, setValues] = useState<BasketFormValues>(BASKET_DEFAULTS);
  const [clientErrors, setClientErrors] = useState<FieldErrors>({});
  const errors = { ...serverErrors, ...clientErrors };
  const set = (key: keyof BasketFormValues) => (event: { target: { value: string } }) => {
    setValues((v) => ({ ...v, [key]: event.target.value }));
    setClientErrors((e) => ({ ...e, [key === "basketSize" ? "basketSize" : key]: "" }));
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const result = validateBasketForm(values, maxSize);
    setClientErrors(result.errors);
    if (result.payload) onSubmit(result.payload);
  };

  const wholeUniverse = maxSize !== undefined && Number(values.basketSize) === maxSize;

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-3">
        <TextField id="basket-capital" label="Capital (₹)" type="number" inputMode="decimal" min="1" step="any" value={values.capital} onChange={set("capital")} error={errors.capital} />
        <TextField
          id="basket-size"
          label="Basket size"
          type="number"
          inputMode="numeric"
          step="1"
          value={values.basketSize}
          onChange={set("basketSize")}
          error={errors.basketSize}
          hint={maxSize ? `3 to ${maxSize} stocks` : undefined}
        />
        <TextField
          id="basket-threshold"
          label="Similarity threshold"
          type="number"
          inputMode="decimal"
          step="0.01"
          value={values.threshold}
          onChange={set("threshold")}
          error={errors.threshold}
          hint="Between -1 and 1. Higher means stricter diversification."
        />
      </div>
      {wholeUniverse && (
        <Notice tone="warn">A basket of {maxSize} stocks is the whole universe, so no selection takes place. Try a smaller basket to see the construction rules at work.</Notice>
      )}
      <Button type="submit" disabled={loading}>
        {loading ? "Generating basket..." : "Generate Basket"}
      </Button>
    </form>
  );
}
