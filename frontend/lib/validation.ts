import type { BacktestRequest, BasketRequest, ErrorDetail, RebalanceFrequency } from "./types";

export type FieldErrors = Record<string, string>;

export interface BasketFormValues {
  capital: string;
  basketSize: string;
  threshold: string;
}
export interface BacktestFormValues extends BasketFormValues {
  frequency: RebalanceFrequency;
  startDate: string;
  endDate: string;
}

const MIN_BASKET = 3;
const MAX_CAPITAL = 1e12;

function toNumber(text: string): number {
  return text.trim() === "" ? NaN : Number(text);
}

/** Checks the shared fields. `maxSize` is the number of stocks available (from the universe API) when known. */
function validateCommon(values: BasketFormValues, maxSize?: number): { errors: FieldErrors; capital: number; size: number; threshold: number } {
  const errors: FieldErrors = {};
  const capital = toNumber(values.capital);
  const size = toNumber(values.basketSize);
  const threshold = toNumber(values.threshold);

  if (!Number.isFinite(capital) || capital <= 0) errors.capital = "Capital must be a number greater than 0.";
  else if (capital > MAX_CAPITAL) errors.capital = "Capital is too large.";

  const upper = maxSize ?? 500;
  if (!Number.isInteger(size) || size < MIN_BASKET || size > upper) errors.basketSize = `Basket size must be between ${MIN_BASKET} and ${upper}.`;

  if (!Number.isFinite(threshold) || threshold < -1 || threshold > 1) errors.threshold = "Similarity threshold must be between -1 and 1.";
  return { errors, capital, size, threshold };
}

export function validateBasketForm(values: BasketFormValues, maxSize?: number): { errors: FieldErrors; payload?: BasketRequest } {
  const { errors, capital, size, threshold } = validateCommon(values, maxSize);
  if (Object.keys(errors).length) return { errors };
  return { errors, payload: { capital, basket_size: size, similarity_threshold: threshold } };
}

export function validateBacktestForm(values: BacktestFormValues, maxSize?: number): { errors: FieldErrors; payload?: BacktestRequest } {
  const { errors, capital, size, threshold } = validateCommon(values, maxSize);
  if (values.startDate && values.endDate && values.startDate > values.endDate) errors.endDate = "End date must be on or after the start date.";
  if (Object.keys(errors).length) return { errors };
  return {
    errors,
    payload: {
      capital,
      basket_size: size,
      frequency: values.frequency,
      similarity_threshold: threshold,
      start_date: values.startDate || null,
      end_date: values.endDate || null,
    },
  };
}

const FIELD_MAP: Record<string, string> = {
  capital: "capital",
  basket_size: "basketSize",
  similarity_threshold: "threshold",
  start_date: "startDate",
  end_date: "endDate",
  frequency: "frequency",
};

/** Maps backend 422 details (field names like "body.basket_size") onto the form's field names. */
export function mapServerErrors(details: ErrorDetail[]): FieldErrors {
  const out: FieldErrors = {};
  for (const detail of details) {
    const key = detail.field.split(".").pop() ?? "";
    const formKey = FIELD_MAP[key];
    if (formKey && !out[formKey]) out[formKey] = detail.message;
  }
  return out;
}
