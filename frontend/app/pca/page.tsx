"use client";
import { useMemo, useState } from "react";
import { LoadingsTable } from "@/components/analysis/LoadingsTable";
import { ScatterPlot, type ScatterGroup } from "@/components/charts/ScatterPlot";
import { ScreePlot } from "@/components/charts/ScreePlot";
import { SelectField } from "@/components/common/fields";
import { ChartSkeleton, ErrorState, LoadingBlock } from "@/components/common/states";
import { Notice, PageHeader, Panel, StatCard } from "@/components/common/ui";
import { getPCA, getUniverse } from "@/lib/api";
import { CAP_COLORS, CAP_ORDER, capColor } from "@/lib/colors";
import { useRequest } from "@/lib/hooks";
import type { PcaResponse, UniverseResponse } from "@/lib/types";
import { describeError, formatInteger, formatPercent } from "@/lib/utils";

type ColourBy = "none" | "cap" | "stock";
const HIGHLIGHT = "#eb6834";
const MUTED_POINTS = "#8a94a0";

function buildGroups(pca: PcaResponse, universe: UniverseResponse | undefined, mode: ColourBy, stock: string): ScatterGroup[] {
  const pts = pca.observations.map((o) => ({ x: o.scores.PC1, y: o.scores.PC2, symbol: o.symbol, date: o.date }));
  if (mode === "cap" && universe) {
    const capOf = new Map(universe.stocks.map((s) => [s.symbol, s.cap_category as string]));
    const names = [...CAP_ORDER, ...new Set(universe.stocks.map((s) => s.cap_category as string))].filter((n, i, a) => a.indexOf(n) === i);
    return names.map((name) => ({ name, color: capColor(name), points: pts.filter((p) => capOf.get(p.symbol) === name) })).filter((g) => g.points.length > 0);
  }
  if (mode === "stock") {
    return [
      { name: "Other stocks", color: MUTED_POINTS, opacity: 0.3, points: pts.filter((p) => p.symbol !== stock) },
      { name: stock, color: HIGHLIGHT, opacity: 0.85, points: pts.filter((p) => p.symbol === stock) },
    ];
  }
  return [{ name: "All stocks", color: "#2a6fb5", points: pts }];
}

function PcaContent({ pca }: { pca: PcaResponse }) {
  const universe = useRequest(getUniverse, "universe");
  const [mode, setMode] = useState<ColourBy>("none");
  const symbols = useMemo(() => [...new Set(pca.observations.map((o) => o.symbol))].sort(), [pca]);
  const [stock, setStock] = useState<string>("");
  const picked = symbols.includes(stock) ? stock : symbols[0];
  const groups = useMemo(() => buildGroups(pca, universe.data, mode === "cap" && !universe.data ? "none" : mode, picked), [pca, universe.data, mode, picked]);

  const pc1 = pca.explained_variance[0];
  const pc2 = pca.explained_variance[1];
  return (
    <div className="space-y-5">
      <Notice>The exploratory PCA analysis is fitted on the complete available synthetic dataset. It is not a walk-forward estimate.</Notice>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Components kept" value={pca.components} />
        <StatCard label="Cumulative variance" value={formatPercent(pca.cumulative_variance[pca.cumulative_variance.length - 1])} hint={`across ${pca.components} components`} />
        <StatCard label="PC1 explained variance" value={formatPercent(pc1)} />
        <StatCard label="PC2 explained variance" value={formatPercent(pc2)} />
      </div>

      <Panel title="Explained variance" description="How much of the variation in the engineered features each component captures.">
        <ScreePlot names={pca.component_names} explained={pca.explained_variance} cumulative={pca.cumulative_variance} />
      </Panel>

      <Panel
        title="PC1 against PC2"
        description={`A sample of ${formatInteger(pca.returned_observations)} of ${formatInteger(pca.total_observations)} stock-day observations, as returned by the API. Each point is one stock on one day.`}
      >
        <div className="mb-4 grid max-w-xl gap-4 sm:grid-cols-2">
          <SelectField id="pca-colour" label="Colour points by" value={mode} onChange={(e) => setMode(e.target.value as ColourBy)}>
            <option value="none">None</option>
            <option value="cap">Cap category</option>
            <option value="stock">Highlight a stock</option>
          </SelectField>
          {mode === "stock" && (
            <SelectField id="pca-stock" label="Stock" value={picked} onChange={(e) => setStock(e.target.value)}>
              {symbols.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </SelectField>
          )}
        </div>
        {mode === "cap" && universe.error !== undefined && <p className="mb-3 text-sm text-danger-ink">Cap categories could not be loaded, so points are shown without colour.</p>}
        <ScatterPlot groups={groups} xLabel={`PC1 (${formatPercent(pc1)})`} yLabel={`PC2 (${formatPercent(pc2)})`} label="Scatter plot of PC1 against PC2" showLegend={groups.length > 1 || mode !== "none"} />
        {mode === "cap" && <p className="mt-2 text-sm text-muted">Cap colours: {Object.entries(CAP_COLORS).map(([k]) => k).join(", ")}.</p>}
      </Panel>

      <Panel title="Feature loadings" description="The weight each engineered feature carries in each component. Scroll sideways on small screens.">
        <LoadingsTable rows={pca.loadings} />
      </Panel>

      <Panel title="About this analysis">
        <p className="text-ink-2">PCA is used here to reduce the dimensionality of the engineered market features and identify latent behavioural structure.</p>
        <p className="mt-2 text-sm text-muted">{pca.notice}</p>
      </Panel>
    </div>
  );
}

export default function PcaPage() {
  const { data, error, loading, reload } = useRequest(() => getPCA(), "pca");
  return (
    <>
      <PageHeader title="PCA Analysis" subtitle="Principal component analysis of the engineered stock features." />
      {loading && (
        <div className="space-y-5">
          <LoadingBlock message="Loading PCA analysis..." />
          <ChartSkeleton message="Preparing charts..." />
        </div>
      )}
      {!loading && (error !== undefined || !data) && <ErrorState message={describeError(error, "PCA analysis")} onRetry={reload} />}
      {data && <PcaContent pca={data} />}
    </>
  );
}
