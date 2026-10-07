"use client";
import type { ReactElement } from "react";
import { ResponsiveContainer } from "recharts";

/** Fixed-height box that lets a Recharts chart fill the width and resize with the screen. */
export function ChartBox({ height = 300, label, children }: { height?: number; label: string; children: ReactElement }) {
  return (
    <div role="img" aria-label={label} style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        {children}
      </ResponsiveContainer>
    </div>
  );
}

export const AXIS = { stroke: "#66717d", fontSize: 12, tickLine: false } as const;
export const GRID = { stroke: "#e8ecef", vertical: false } as const;
export const TOOLTIP_STYLE = { border: "1px solid #dde2e6", borderRadius: 6, fontSize: 13, boxShadow: "none" } as const;
