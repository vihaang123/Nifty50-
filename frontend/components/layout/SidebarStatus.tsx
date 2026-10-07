"use client";
import { useDataset } from "./useDataset";
import { StatusBadge } from "./StatusBadge";
import { formatDate } from "@/lib/utils";

export function SidebarStatus() {
  const { data, error, loading } = useDataset();
  return (
    <div className="border-t border-line p-4 text-sm">
      <div className="mb-1 text-muted">Data source</div>
      <StatusBadge />
      <p className="mt-2 text-ink-2">Synthetic Research Dataset</p>
      {loading && <p className="mt-1 text-muted">Loading dataset...</p>}
      {data && (
        <p className="num mt-1 text-muted">
          {data.stock_count} stocks, {formatDate(data.start_date)} to {formatDate(data.end_date)}
        </p>
      )}
      {!loading && error !== undefined && <p className="mt-1 text-muted">Start the API to see dataset details.</p>}
    </div>
  );
}
