"use client";
import { ErrorState, LoadingBlock } from "@/components/common/states";
import { Panel, StatCard, Swatch } from "@/components/common/ui";
import { getDataset, getLDA, getPCA, getUniverse } from "@/lib/api";
import { BEHAVIOR_ORDER, CAP_ORDER, behaviorColor } from "@/lib/colors";
import { useRequest } from "@/lib/hooks";
import { describeError, formatDate, formatInteger, formatPercent, formatPercentPoints } from "@/lib/utils";

// The dashboard only needs the numbers, so it asks for a single sample point and avoids a large download.
export const usePcaSummary = () => useRequest(() => getPCA({ maxPoints: 1 }), "pca-summary");
export const useLdaSummary = () => useRequest(() => getLDA({ maxPoints: 1 }), "lda-summary");

function Row({ label, value }: { label: React.ReactNode; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-2 text-sm last:border-0">
      <span className="text-ink-2">{label}</span>
      <span className="num text-right font-medium">{value}</span>
    </div>
  );
}

export function DatasetCard() {
  const dataset = useRequest(getDataset, "dataset");
  const pca = usePcaSummary();
  return (
    <Panel title="Dataset">
      {dataset.loading && <LoadingBlock message="Loading dataset..." />}
      {!dataset.loading && (dataset.error !== undefined || !dataset.data) && <ErrorState message={describeError(dataset.error, "the dataset")} onRetry={dataset.reload} />}
      {dataset.data && (
        <div>
          <Row label="Stocks" value={dataset.data.stock_count} />
          <Row label="Trading days" value={formatInteger(dataset.data.trading_days)} />
          <Row label="Date range" value={`${formatDate(dataset.data.start_date)} to ${formatDate(dataset.data.end_date)}`} />
          <Row label="Observations" value={pca.data ? formatInteger(pca.data.total_observations) : pca.loading ? "Loading..." : "n/a"} />
          <Row label="Data source" value={dataset.data.is_synthetic ? "Synthetic Research Dataset" : dataset.data.source} />
          <p className="mt-3 text-sm text-muted">Observations are stock-day feature rows after the warm-up period needed to compute the features.</p>
        </div>
      )}
    </Panel>
  );
}

export function UniverseCard() {
  const { data, error, loading, reload } = useRequest(getUniverse, "universe");
  return (
    <Panel title="Stock universe">
      {loading && <LoadingBlock message="Loading universe..." />}
      {!loading && (error !== undefined || !data) && <ErrorState message={describeError(error, "the universe")} onRetry={reload} />}
      {data && (
        <div className="grid grid-cols-3 gap-3">
          {[...CAP_ORDER, ...Object.keys(data.counts).filter((k) => !CAP_ORDER.includes(k))]
            .filter((k) => k in data.counts)
            .map((k) => (
              <div key={k}>
                <div className="num font-serif text-3xl font-medium">{data.counts[k]}</div>
                <div className="text-sm text-ink-2">{k}</div>
              </div>
            ))}
        </div>
      )}
    </Panel>
  );
}

export function PcaSummaryCard() {
  const { data, error, loading, reload } = usePcaSummary();
  return (
    <Panel title="PCA summary" description="Exploratory, fitted on the full synthetic history.">
      {loading && <LoadingBlock message="Loading PCA analysis..." />}
      {!loading && (error !== undefined || !data) && <ErrorState message={describeError(error, "PCA analysis")} onRetry={reload} />}
      {data && (
        <div className="grid grid-cols-2 gap-3">
          <StatCard label="Components" value={data.components} />
          <StatCard label="Cumulative variance" value={formatPercent(data.cumulative_variance[data.cumulative_variance.length - 1])} />
          <StatCard label="PC1 explained" value={formatPercent(data.explained_variance[0])} />
          <StatCard label="PC2 explained" value={formatPercent(data.explained_variance[1])} />
        </div>
      )}
    </Panel>
  );
}

export function LdaSummaryCard() {
  const { data, error, loading, reload } = useLdaSummary();
  return (
    <Panel title="LDA summary" description="Three constructed behaviour classes.">
      {loading && <LoadingBlock message="Loading LDA analysis..." />}
      {!loading && (error !== undefined || !data) && <ErrorState message={describeError(error, "LDA analysis")} onRetry={reload} />}
      {data && (
        <div>
          <Row label="Classes" value={data.classes.length} />
          {[...BEHAVIOR_ORDER].map((name) => {
            const c = data.classes.find((x) => x.name === name);
            return (
              c && (
                <Row
                  key={name}
                  label={
                    <span className="inline-flex items-center gap-1.5">
                      <Swatch color={behaviorColor(name)} />
                      {name}
                    </span>
                  }
                  value={`${formatInteger(c.count)} (${formatPercentPoints(c.percentage)})`}
                />
              )
            );
          })}
          <Row label="Training accuracy" value={formatPercent(data.training_accuracy)} />
          <p className="mt-3 text-sm text-muted">Training accuracy on constructed behavioural classes. It is an in-sample diagnostic, not a forecast.</p>
        </div>
      )}
    </Panel>
  );
}
