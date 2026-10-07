"use client";
import { Area, AreaChart, CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ChartBox, GRID, TOOLTIP_STYLE } from "@/components/charts/ChartBox";
import { EmptyState } from "@/components/common/states";
import { Legend, Notice, Panel } from "@/components/common/ui";
import type { BacktestResponse } from "@/lib/types";
import { formatDate, formatMonthYear, formatPercent, formatRupees, formatYear } from "@/lib/utils";
import { MetricGrid } from "./MetricGrid";
import { RebalanceTable } from "./RebalanceTable";

const STRATEGY = "#2a6fb5";
const BENCH = "#8a94a0";

export function BacktestResults({ result }: { result: BacktestResponse }) {
  if (result.equity_curve.length === 0) return <EmptyState>The backtest returned no results for these settings. Try a wider date range or a smaller basket.</EmptyState>;
  // Long histories read best by year; short ones need the month to avoid repeated labels.
  const axisDate = result.equity_curve.length > 1100 ? formatYear : formatMonthYear;
  const bench = result.benchmark_name || "Benchmark";
  const info = result.data_info;
  const s = result.settings;
  return (
    <div className="space-y-5">
      {result.is_synthetic && <Notice tone="warn">These results come from the synthetic research dataset. They are not real market performance.</Notice>}
      {result.notices.length > 0 && (
        <Notice title="Backend notices">
          <ul className="list-disc space-y-1 pl-5">
            {result.notices.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </Notice>
      )}

      <MetricGrid summary={result.summary} benchmark={result.benchmark} benchmarkName={bench} />

      <Panel title="Equity curve" description={`Strategy against ${bench}. Both start at ${formatRupees(result.summary.initial_capital)} on ${formatDate(info.baseline_date)}.`}>
        <Legend items={[{ label: "Strategy", color: STRATEGY }, { label: bench, color: BENCH }]} />
        <div className="mt-3">
          <ChartBox height={320} label="Equity curve of the strategy and the benchmark">
            <LineChart data={result.equity_curve} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="date" {...AXIS} tickFormatter={axisDate} minTickGap={60} />
              <YAxis {...AXIS} width={70} tickFormatter={(v) => formatRupees(Number(v))} domain={["auto", "auto"]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(l) => formatDate(String(l))} formatter={(v, n) => [formatRupees(Number(v)), n === "portfolio_value" ? "Strategy" : bench]} />
              <Line type="monotone" dataKey="portfolio_value" stroke={STRATEGY} strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="benchmark_value" stroke={BENCH} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ChartBox>
        </div>
      </Panel>

      <Panel title="Drawdown" description="How far each line sat below its previous peak.">
        <Legend items={[{ label: "Strategy", color: STRATEGY }, { label: bench, color: BENCH }]} />
        <div className="mt-3">
          <ChartBox height={240} label="Drawdown of the strategy and the benchmark">
            <AreaChart data={result.drawdown} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="date" {...AXIS} tickFormatter={axisDate} minTickGap={60} />
              <YAxis {...AXIS} width={56} tickFormatter={(v) => formatPercent(Number(v), 0)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(l) => formatDate(String(l))} formatter={(v, n) => [formatPercent(Number(v)), n === "portfolio_drawdown" ? "Strategy" : bench]} />
              <Area type="monotone" dataKey="benchmark_drawdown" stroke={BENCH} strokeWidth={1.5} fill={BENCH} fillOpacity={0.15} dot={false} isAnimationActive={false} />
              <Area type="monotone" dataKey="portfolio_drawdown" stroke={STRATEGY} strokeWidth={2} fill={STRATEGY} fillOpacity={0.15} dot={false} isAnimationActive={false} />
            </AreaChart>
          </ChartBox>
        </div>
      </Panel>

      <Panel title="Rebalance history" description={`${result.rebalances.length} rebalance dates. The portfolio value is the value on the rebalance date, before new weights are applied.`}>
        <RebalanceTable rebalances={result.rebalances} history={result.rebalance_history} />
      </Panel>

      {(result.skipped_rebalances.length > 0 || result.missing_data_events.length > 0) && (
        <Panel title="Diagnostics">
          {result.skipped_rebalances.length > 0 && (
            <div className="mb-3">
              <h3 className="mb-1 text-sm font-medium text-ink-2">Skipped rebalances</h3>
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {result.skipped_rebalances.map((r) => (
                  <li key={r.rebalance_date}>
                    <span className="num">{formatDate(r.rebalance_date)}</span>: {r.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {result.missing_data_events.length > 0 && (
            <div>
              <h3 className="mb-1 text-sm font-medium text-ink-2">Missing-price events</h3>
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {result.missing_data_events.map((e) => (
                  <li key={`${e.rebalance_date}-${e.symbol}`}>
                    {e.symbol} stopped on <span className="num">{formatDate(e.stop_date)}</span> (held since {formatDate(e.rebalance_date)})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Panel>
      )}

      <Panel title="Assumptions used">
        <p className="text-sm text-ink-2">
          Walk-forward: at each rebalance the models are fitted only on data before that date. Equal weights, {formatPercent(s.transaction_cost)} transaction cost, {formatPercent(s.slippage)} slippage, risk-free rate{" "}
          {formatPercent(s.risk_free_rate)}. Period {formatDate(info.backtest_start)} to {formatDate(info.backtest_end)}, {info.n_rebalance_dates} rebalances, basket of {s.basket_size}, {s.frequency}.
        </p>
      </Panel>
    </div>
  );
}
