import type { DatasetResponse } from "./types";

export interface DataSourceLabel {
  /** The full label, for the dataset card and the sidebar: "Synthetic Research Dataset". */
  label: string;
  /** The compact label, for the badge: "Synthetic Dataset". */
  badge: string;
  /** Synthetic data always gets the warning colour. Everything else is neutral. */
  synthetic: boolean;
}

const prettify = (provider: string) =>
  provider
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

/**
 * What to call the data, decided only by what the API says. Nothing here is hard-coded to the synthetic dataset, so the
 * day the backend reports `provider: "angel_one"` the UI reads "Angel One" without a code change.
 *
 * It never says "live". Whether data is live is not something the dataset response states, so the UI does not claim it.
 */
export function describeDataSource(dataset: Pick<DatasetResponse, "provider" | "source" | "is_synthetic">): DataSourceLabel {
  if (dataset.is_synthetic) return { label: "Synthetic Research Dataset", badge: "Synthetic Dataset", synthetic: true };
  const provider = (dataset.provider || dataset.source || "").trim().toLowerCase();
  if (provider === "angel_one") return { label: "Angel One", badge: "Angel One", synthetic: false };
  if (provider === "local") return { label: "Local Dataset", badge: "Local Dataset", synthetic: false };
  const name = prettify(provider) || "Unknown source";
  return { label: name, badge: name, synthetic: false };
}
