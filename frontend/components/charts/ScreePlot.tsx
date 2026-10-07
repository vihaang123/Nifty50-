"use client";
import { Bar, CartesianGrid, ComposedChart, Line, Tooltip, XAxis, YAxis } from "recharts";
import { Legend } from "@/components/common/ui";
import { formatPercent } from "@/lib/utils";
import { AXIS, ChartBox, GRID, TOOLTIP_STYLE } from "./ChartBox";

const BAR = "#2a6fb5";
const LINE = "#16202e";

/** Explained variance per component (bars) with the running total (line). Both are fractions on one axis. */
export function ScreePlot({ names, explained, cumulative, height = 280 }: { names: string[]; explained: number[]; cumulative: number[]; height?: number }) {
  const data = names.map((name, i) => ({ name, explained: explained[i], cumulative: cumulative[i] }));
  return (
    <div>
      <Legend items={[{ label: "Explained variance", color: BAR }, { label: "Cumulative variance", color: LINE }]} />
      <div className="mt-3">
        <ChartBox height={height} label="Explained variance by principal component, with cumulative variance">
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="name" {...AXIS} />
            <YAxis {...AXIS} width={48} domain={[0, 1]} tickFormatter={(v) => formatPercent(Number(v), 0)} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v, n) => [formatPercent(Number(v)), n === "explained" ? "Explained variance" : "Cumulative variance"]} />
            <Bar dataKey="explained" fill={BAR} radius={[3, 3, 0, 0]} maxBarSize={48} isAnimationActive={false} />
            <Line type="monotone" dataKey="cumulative" stroke={LINE} strokeWidth={2} dot={{ r: 4, fill: LINE, stroke: "#fff", strokeWidth: 2 }} isAnimationActive={false} />
          </ComposedChart>
        </ChartBox>
      </div>
    </div>
  );
}
