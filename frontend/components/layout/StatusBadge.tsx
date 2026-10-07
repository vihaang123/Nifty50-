"use client";
import { useDataset } from "./useDataset";
import { describeDataSource } from "@/lib/dataSource";
import { cn } from "@/lib/utils";

/** The data-source indicator. It is driven by the API response, never hard-coded. */
export function StatusBadge({ className }: { className?: string }) {
  const { data, error, loading } = useDataset();
  let text = "Checking dataset...";
  let dot = "bg-muted";
  let tone = "text-ink-2";
  if (data) {
    const source = describeDataSource(data);
    text = source.badge;
    dot = source.synthetic ? "bg-[#c98a00]" : "bg-accent";
    tone = source.synthetic ? "text-amber-ink" : "text-ink-2";
  } else if (!loading && error) {
    text = "Backend unavailable";
    dot = "bg-danger-ink";
    tone = "text-danger-ink";
  }
  return (
    <span className={cn("inline-flex items-center gap-2 text-sm font-medium", tone, className)}>
      <span aria-hidden className={cn("h-2 w-2 rounded-full", dot)} />
      {text}
    </span>
  );
}
