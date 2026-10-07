"use client";
import Link from "next/link";
import { MetricGrid } from "@/components/backtest/MetricGrid";
import { ErrorState } from "@/components/common/states";
import { Button, Notice, Panel } from "@/components/common/ui";
import { runBacktest } from "@/lib/api";
import { useAction } from "@/lib/hooks";
import type { BacktestRequest } from "@/lib/types";
import { describeError } from "@/lib/utils";

// Shown on screen too, so nobody mistakes these for hidden settings.
export const QUICK_BACKTEST: BacktestRequest = { capital: 50000, basket_size: 6, frequency: "quarterly", similarity_threshold: 0.9 };

export function QuickBacktest() {
  const { state, run, cancel, elapsed } = useAction(runBacktest);
  const loading = state.status === "loading";
  return (
    <Panel title="Quick backtest" description="Runs only when you press the button. Settings: ₹50,000 capital, 6 stocks, quarterly rebalance, similarity threshold 0.90, full date range.">
      <div className="flex flex-wrap items-center gap-3">
        <Button disabled={loading} onClick={() => run(QUICK_BACKTEST)}>
          {loading ? "Running backtest..." : "Run Quick Backtest"}
        </Button>
        {loading && (
          <>
            <Button variant="secondary" onClick={cancel}>
              Cancel
            </Button>
            <span role="status" aria-live="polite" className="text-sm text-ink-2">
              Running walk-forward backtest... <span className="num">{elapsed}s</span>
            </span>
          </>
        )}
      </div>
      <div className="mt-4 space-y-4">
        {state.status === "error" && <ErrorState message={describeError(state.error, "the backtest")} />}
        {state.status === "success" && (
          <>
            {state.data.is_synthetic && <Notice tone="warn">Synthetic data. This is not real market performance.</Notice>}
            <MetricGrid summary={state.data.summary} benchmark={state.data.benchmark} benchmarkName={state.data.benchmark_name} />
            <p className="text-sm">
              <Link href="/backtest" className="text-accent underline underline-offset-2">
                Open the full Backtest page
              </Link>{" "}
              for charts, parameters and rebalance history.
            </p>
          </>
        )}
      </div>
    </Panel>
  );
}
