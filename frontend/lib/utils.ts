import { ApiError } from "./api";

export const cn = (...parts: Array<string | false | null | undefined>) => parts.filter(Boolean).join(" ");

const rupees = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const rupeesExact = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const integer = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

export const formatRupees = (value: number, exact = false) => (exact ? rupeesExact : rupees).format(value);
export const formatInteger = (value: number) => integer.format(value);

/** 0.359 -> "35.9%". `signed` adds a + for gains. */
export function formatPercent(fraction: number | null | undefined, digits = 1, signed = false): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return "n/a";
  const text = (fraction * 100).toFixed(digits);
  return signed && fraction > 0 ? `+${text}%` : `${text}%`;
}

/** A number that is already a percentage (0-100). */
export const formatPercentPoints = (value: number, digits = 1) => `${value.toFixed(digits)}%`;

export function formatNumber(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "n/a" : value.toFixed(digits);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** '2019-01-01' -> '1 Jan 2019' (no time zone maths: the text is split, not parsed as a Date). */
export function formatDate(iso: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!match) return iso;
  return `${Number(match[3])} ${MONTHS[Number(match[2]) - 1]} ${match[1]}`;
}

export const formatYear = (iso: string) => iso.slice(0, 4);

/** '2019-07-01' -> 'Jul 2019', for chart axes that span less than a few years. */
export function formatMonthYear(iso: string): string {
  const match = /^(\d{4})-(\d{2})/.exec(iso);
  return match ? `${MONTHS[Number(match[2]) - 1]} ${match[1]}` : iso;
}

/** Plain-language message for a failed request. Never shows stack traces or backend internals. */
export function describeError(error: unknown, what: string): string {
  if (error instanceof ApiError) {
    if (error.kind === "network") return `Unable to load ${what}. Please check that the backend is running.`;
    if (error.kind === "config") return error.message;
    if (error.status !== undefined && error.status >= 500) return `The backend hit a problem while loading ${what}. Please try again.`;
    const extra = error.details.map((d) => d.message).filter(Boolean);
    return extra.length ? `${error.message} ${extra.join(" ")}` : error.message;
  }
  return `Unable to load ${what}. Please try again.`;
}

export const sharpeText = (value: number | null | undefined) => formatNumber(value, 2);
