"use client";
import { CartesianGrid, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { Legend } from "@/components/common/ui";
import { AXIS, ChartBox, TOOLTIP_STYLE } from "./ChartBox";

export interface ScatterPoint {
  x: number;
  y: number;
  symbol: string;
  date: string;
}
export interface ScatterGroup {
  name: string;
  color: string;
  points: ScatterPoint[];
  opacity?: number;
}

function PointTip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ScatterPoint }> }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div style={TOOLTIP_STYLE} className="bg-white px-3 py-2">
      <div className="font-medium">{p.symbol}</div>
      <div className="text-muted">{p.date}</div>
      <div className="num">
        {p.x.toFixed(2)}, {p.y.toFixed(2)}
      </div>
    </div>
  );
}

export function ScatterPlot({ groups, xLabel, yLabel, label, height = 380, showLegend = true }: { groups: ScatterGroup[]; xLabel: string; yLabel: string; label: string; height?: number; showLegend?: boolean }) {
  return (
    <div>
      {showLegend && <Legend items={groups.map((g) => ({ label: `${g.name} (${g.points.length})`, color: g.color }))} />}
      <div className="mt-3">
        <ChartBox height={height} label={label}>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid stroke="#e8ecef" />
            <XAxis type="number" dataKey="x" name={xLabel} {...AXIS} domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toFixed(0)} label={{ value: xLabel, position: "insideBottom", offset: -12, fill: "#48525f", fontSize: 12 }} />
            <YAxis type="number" dataKey="y" name={yLabel} {...AXIS} width={44} domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toFixed(0)} label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "#48525f", fontSize: 12 }} />
            <ZAxis range={[26, 26]} />
            <Tooltip content={<PointTip />} cursor={{ strokeDasharray: "3 3" }} />
            {groups.map((g) => (
              <Scatter key={g.name} data={g.points} fill={g.color} fillOpacity={g.opacity ?? 0.55} isAnimationActive={false} />
            ))}
          </ScatterChart>
        </ChartBox>
      </div>
    </div>
  );
}
