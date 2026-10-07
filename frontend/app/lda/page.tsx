"use client";
import { useMemo } from "react";
import { LoadingsTable } from "@/components/analysis/LoadingsTable";
import { ScatterPlot } from "@/components/charts/ScatterPlot";
import { ChartSkeleton, ErrorState, LoadingBlock } from "@/components/common/states";
import { Notice, PageHeader, Panel, StatCard, Swatch } from "@/components/common/ui";
import { getLDA } from "@/lib/api";
import { BEHAVIOR_ORDER, behaviorColor } from "@/lib/colors";
import { useRequest } from "@/lib/hooks";
import type { LdaResponse } from "@/lib/types";
import { describeError, formatInteger, formatPercent, formatPercentPoints } from "@/lib/utils";

function LdaContent({ lda }: { lda: LdaResponse }) {
  const groups = useMemo(() => {
    const names = [...BEHAVIOR_ORDER, ...lda.classes.map((c) => c.name as string)].filter((n, i, a) => a.indexOf(n) === i);
    return names
      .map((name) => ({
        name,
        color: behaviorColor(name),
        points: lda.points.filter((p) => p.behavior_class === name).map((p) => ({ x: p.LD1, y: p.LD2, symbol: p.symbol, date: p.date })),
      }))
      .filter((g) => g.points.length > 0);
  }, [lda]);

  const ld1 = lda.explained_variance.LD1;
  const ld2 = lda.explained_variance.LD2;
  return (
    <div className="space-y-5">
      <Notice tone="warn" title="Read the accuracy with care">
        The behavioural classes are constructed from historical volatility, beta and return characteristics. Because LDA is trained on these constructed labels using the same feature set, this accuracy is an in-sample diagnostic rather than an independent predictive metric.
      </Notice>

      <div className="grid gap-3 sm:grid-cols-3">
        {lda.classes.map((c) => (
          <div key={c.name} className="rounded-md border border-line bg-surface p-4">
            <div className="flex items-center gap-2 text-sm text-muted">
              <Swatch color={behaviorColor(c.name)} />
              {c.name}
            </div>
            <div className="num mt-1 font-serif text-3xl font-medium">{formatInteger(c.count)}</div>
            <div className="text-sm text-ink-2">{formatPercentPoints(c.percentage)} of stock-days</div>
          </div>
        ))}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <StatCard label="Training accuracy" value={formatPercent(lda.training_accuracy)} hint="Training accuracy on constructed behavioural classes" />
        <StatCard label="Majority-class baseline" value={formatPercent(lda.majority_baseline)} hint="Accuracy of always guessing the most common class" />
      </div>

      <Panel
        title="LD1 against LD2"
        description={`A sample of ${formatInteger(lda.returned_points)} of ${formatInteger(lda.total_points)} stock-day observations, as returned by the API. Axes show the share of between-class separation each discriminant carries.`}
      >
        <ScatterPlot groups={groups} xLabel={`LD1 (${formatPercent(ld1)})`} yLabel={`LD2 (${formatPercent(ld2)})`} label="Scatter plot of LD1 against LD2, coloured by behaviour class" />
      </Panel>

      <Panel title="Discriminant loadings" description="The weight each engineered feature carries in each discriminant.">
        <LoadingsTable rows={lda.loadings} />
      </Panel>

      <Panel title="About this analysis">
        <p className="text-ink-2">LDA finds the directions that best separate the three behaviour classes. Here it is a way to view and describe the classes, not a way to forecast prices.</p>
        <p className="mt-2 text-sm text-muted">{lda.notice}</p>
      </Panel>
    </div>
  );
}

export default function LdaPage() {
  const { data, error, loading, reload } = useRequest(() => getLDA(), "lda");
  return (
    <>
      <PageHeader title="LDA Analysis" subtitle="Linear discriminant analysis of the Defensive, Balanced and Aggressive behaviour classes." />
      {loading && (
        <div className="space-y-5">
          <LoadingBlock message="Loading LDA analysis..." />
          <ChartSkeleton message="Preparing charts..." />
        </div>
      )}
      {!loading && (error !== undefined || !data) && <ErrorState message={describeError(error, "LDA analysis")} onRetry={reload} />}
      {data && <LdaContent lda={data} />}
    </>
  );
}
