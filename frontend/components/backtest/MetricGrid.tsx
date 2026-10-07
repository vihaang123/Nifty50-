import { StatCard } from "@/components/common/ui";
import type { BenchmarkSummary, StrategySummary } from "@/lib/types";
import { formatNumber, formatPercent, formatRupees } from "@/lib/utils";

/** The six headline numbers, straight from the API. The hint under each shows the benchmark's value where one exists. */
export function MetricGrid({ summary, benchmark, benchmarkName }: { summary: StrategySummary; benchmark?: BenchmarkSummary; benchmarkName?: string }) {
  const b = benchmark;
  const tag = benchmarkName ? `${benchmarkName}: ` : "Benchmark: ";
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <StatCard label="Final Portfolio Value" value={formatRupees(summary.final_portfolio_value)} hint={`from ${formatRupees(summary.initial_capital)}${b ? `. ${tag}${formatRupees(b.final_value)}` : ""}`} />
      <StatCard label="Cumulative Return" value={formatPercent(summary.cumulative_return, 1, true)} tone={summary.cumulative_return < 0 ? "bad" : undefined} hint={b && `${tag}${formatPercent(b.cumulative_return, 1, true)}`} />
      <StatCard label="Annualized Return" value={formatPercent(summary.annualized_return, 1, true)} tone={summary.annualized_return < 0 ? "bad" : undefined} hint={b && `${tag}${formatPercent(b.annualized_return, 1, true)}`} />
      <StatCard label="Annualized Volatility" value={formatPercent(summary.annualized_volatility)} hint={b && `${tag}${formatPercent(b.annualized_volatility)}`} />
      <StatCard label="Sharpe Ratio" value={formatNumber(summary.sharpe_ratio, 2)} hint={b && `${tag}${formatNumber(b.sharpe_ratio, 2)}`} />
      <StatCard label="Maximum Drawdown" value={formatPercent(summary.max_drawdown)} tone="bad" hint={b && `${tag}${formatPercent(b.max_drawdown)}`} />
    </div>
  );
}
