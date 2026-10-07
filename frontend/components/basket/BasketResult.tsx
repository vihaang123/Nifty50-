"use client";
import { Cell, Pie, PieChart, Tooltip } from "recharts";
import { ChartBox, TOOLTIP_STYLE } from "@/components/charts/ChartBox";
import { EmptyState } from "@/components/common/states";
import { Notice, Panel, StatCard, Swatch } from "@/components/common/ui";
import { BEHAVIOR_ORDER, CAP_ORDER, behaviorColor, capColor } from "@/lib/colors";
import type { BasketResponse } from "@/lib/types";
import { formatNumber, formatPercent, formatRupees } from "@/lib/utils";
import { BasketTable } from "./BasketTable";
import { DistributionBar } from "./DistributionBar";

interface Props {
  basket: BasketResponse;
  /** Stocks in the universe, to warn when the basket is simply every stock. */
  universeSize?: number;
  compact?: boolean;
}

export function BasketResult({ basket, universeSize, compact = false }: Props) {
  if (basket.stocks.length === 0) return <EmptyState>The backend returned an empty basket for these settings. Try a lower similarity threshold or a smaller basket.</EmptyState>;
  const stats = basket.statistics;
  const wholeUniverse = universeSize !== undefined && basket.stocks.length >= universeSize;
  const donut = basket.stocks.map((s) => ({ name: s.symbol, value: s.weight, behavior: s.behavior_class }));

  return (
    <div className="space-y-5">
      {wholeUniverse && <Notice tone="warn">This basket contains every stock in the universe, so the selection rules had no effect on which stocks were chosen.</Notice>}
      {basket.notes.length > 0 && (
        <Notice tone="warn" title="Backend notes">
          <ul className="list-disc space-y-1 pl-5">
            {basket.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </Notice>
      )}

      <BasketTable stocks={basket.stocks} showReason={!compact} />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Total allocation" value={formatRupees(basket.total_allocation)} hint={`of ${formatRupees(basket.capital)} capital`} />
        <StatCard label="Average pairwise similarity" value={formatNumber(stats.average_similarity, 2)} />
        <StatCard label="Maximum pairwise similarity" value={formatNumber(stats.maximum_similarity, 2)} hint={stats.maximum_pair.join(" and ")} />
        <StatCard label="Pairs above threshold" value={`${stats.pairs_above_threshold} of ${stats.n_pairs}`} hint={`threshold ${formatNumber(basket.similarity_threshold, 2)}`} />
      </div>

      {!compact && (
        <>
          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Allocation" description="Slices are coloured by behaviour class; the list shows each stock's weight.">
              <div className="grid items-center gap-4 sm:grid-cols-2">
                <ChartBox height={220} label="Allocation donut chart by stock weight">
                  <PieChart>
                    <Pie data={donut} dataKey="value" nameKey="name" innerRadius="55%" outerRadius="90%" stroke="#ffffff" strokeWidth={2} isAnimationActive={false}>
                      {donut.map((d) => (
                        <Cell key={d.name} fill={behaviorColor(d.behavior)} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => formatPercent(Number(v))} />
                  </PieChart>
                </ChartBox>
                <ul className="space-y-1 text-sm">
                  {donut.map((d) => (
                    <li key={d.name} className="flex items-center justify-between gap-3">
                      <span className="flex items-center gap-1.5">
                        <Swatch color={behaviorColor(d.behavior)} />
                        {d.name}
                        <span className="text-muted">({d.behavior})</span>
                      </span>
                      <span className="num">{formatPercent(d.value)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </Panel>
            <Panel title="Diversification">
              <div className="space-y-5">
                <DistributionBar title="Cap distribution" distribution={stats.cap_distribution} colorOf={(n) => capColor(n)} order={CAP_ORDER} />
                <DistributionBar title="Behaviour distribution" distribution={stats.behavior_distribution} colorOf={(n) => behaviorColor(n)} order={BEHAVIOR_ORDER} />
                <div className="text-sm text-ink-2">
                  <div className="mb-1 font-medium">Target and planned stocks per cap category</div>
                  {Object.keys(basket.target_allocation).map((cap) => (
                    <div key={cap} className="num flex justify-between">
                      <span>{cap}</span>
                      <span>
                        target {basket.target_allocation[cap]}, planned {basket.planned_allocation[cap] ?? 0}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>
          </div>
          <p className="text-sm text-muted">{basket.notice}</p>
        </>
      )}
    </div>
  );
}
