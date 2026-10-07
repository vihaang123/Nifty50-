import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Button } from "./ui";

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("rounded bg-line/70 motion-safe:animate-pulse", className)} />;
}

export function LoadingBlock({ message, lines = 3 }: { message: string; lines?: number }) {
  return (
    <div role="status" aria-live="polite" className="space-y-2">
      <p className="text-sm text-muted">{message}</p>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} className={cn("h-4", i === lines - 1 ? "w-2/3" : "w-full")} />
      ))}
    </div>
  );
}

export function ChartSkeleton({ message, height = 280 }: { message: string; height?: number }) {
  return (
    <div role="status" aria-live="polite">
      <p className="mb-2 text-sm text-muted">{message}</p>
      <div style={{ height }} className="rounded bg-line/50 motion-safe:animate-pulse" aria-hidden />
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-md border border-danger-ink/30 bg-danger-soft px-4 py-3 text-sm text-danger-ink">
      <p>{message}</p>
      {onRetry && (
        <Button variant="secondary" className="mt-3" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="rounded-md border border-dashed border-line-strong bg-surface px-4 py-8 text-center text-sm text-ink-2">{children}</div>;
}
