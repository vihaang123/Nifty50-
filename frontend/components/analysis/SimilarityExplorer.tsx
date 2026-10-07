"use client";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ChartBox, GRID, TOOLTIP_STYLE } from "@/components/charts/ChartBox";
import { SelectField } from "@/components/common/fields";
import { ErrorState, LoadingBlock } from "@/components/common/states";
import { TableWrap, tableClasses as t } from "@/components/common/ui";
import { getSimilarity, getUniverse } from "@/lib/api";
import { useRequest } from "@/lib/hooks";
import { describeError, formatNumber } from "@/lib/utils";

const TOP_N = 5;

function SimilarityResult({ symbol, showChart }: { symbol: string; showChart: boolean }) {
  const { data, error, loading, reload } = useRequest(() => getSimilarity(symbol, TOP_N), `similarity:${symbol}`);
  if (loading) return <LoadingBlock message="Loading similarity..." />;
  if (error !== undefined || !data) return <ErrorState message={describeError(error, "similarity results")} onRetry={reload} />;
  const chartData = data.similar_stocks.map((s) => ({ symbol: s.symbol, similarity: s.similarity }));
  return (
    <div className="space-y-4">
      <div className="rounded-md bg-accent-soft px-4 py-3 text-sm">
        <span className="text-ink-2">Selected stock: </span>
        <span className="font-medium">{data.selected.symbol}</span>
        <span className="text-ink-2">
          {" "}
          ({data.selected.cap_category ?? "cap n/a"}, {data.selected.behavior_class ?? "behaviour n/a"})
        </span>
      </div>
      <div>
        <h3 className="mb-2 text-sm font-medium text-ink-2">Most similar stocks</h3>
        <TableWrap>
          <table className={t.table}>
            <thead>
              <tr>
                <th className={t.th}>Rank</th>
                <th className={t.th}>Symbol</th>
                <th className={`${t.th} text-right`}>Similarity</th>
                <th className={t.th}>Cap</th>
                <th className={t.th}>Behaviour</th>
              </tr>
            </thead>
            <tbody>
              {data.similar_stocks.map((s) => (
                <tr key={s.symbol}>
                  <td className={`${t.td} num`}>{s.rank}</td>
                  <td className={`${t.td} font-medium`}>{s.symbol}</td>
                  <td className={`${t.td} num text-right`}>{formatNumber(s.similarity, 3)}</td>
                  <td className={t.td}>{s.cap_category ?? "n/a"}</td>
                  <td className={t.td}>{s.behavior_class ?? "n/a"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </div>
      {showChart && (
        <ChartBox height={40 + chartData.length * 44} label={`Similarity of the stocks closest to ${data.selected.symbol}`}>
          <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
            <CartesianGrid {...GRID} horizontal={false} vertical />
            <XAxis type="number" {...AXIS} domain={[-1, 1]} />
            <YAxis type="category" dataKey="symbol" {...AXIS} width={90} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [formatNumber(Number(v), 3), "Similarity"]} />
            <Bar dataKey="similarity" fill="#2a6fb5" radius={[0, 3, 3, 0]} barSize={18} isAnimationActive={false} />
          </BarChart>
        </ChartBox>
      )}
      <p className="text-sm text-muted">Similarity represents historical behavioural similarity in PCA space. It does not represent expected future returns.</p>
    </div>
  );
}

export function SimilarityExplorer({ showChart = true }: { showChart?: boolean }) {
  const universe = useRequest(getUniverse, "universe");
  const [chosen, setChosen] = useState<string | null>(null);

  if (universe.loading) return <LoadingBlock message="Loading stock universe..." />;
  if (universe.error !== undefined || !universe.data) return <ErrorState message={describeError(universe.error, "the stock list")} onRetry={universe.reload} />;

  const symbols = universe.data.stocks.map((s) => s.symbol);
  const symbol = chosen && symbols.includes(chosen) ? chosen : symbols[0];
  return (
    <div className="space-y-4">
      <div className="max-w-xs">
        <SelectField id="similarity-symbol" label="Stock" value={symbol} onChange={(e) => setChosen(e.target.value)}>
          {universe.data.stocks.map((s) => (
            <option key={s.symbol} value={s.symbol}>
              {s.symbol} ({s.cap_category})
            </option>
          ))}
        </SelectField>
      </div>
      {symbol ? <SimilarityResult symbol={symbol} showChart={showChart} /> : null}
    </div>
  );
}
