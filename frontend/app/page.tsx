"use client";
import Link from "next/link";
import { ScreePlot } from "@/components/charts/ScreePlot";
import { useDataset } from "@/components/layout/useDataset";
import { usePcaSummary } from "@/components/dashboard/Summaries";
import { Skeleton } from "@/components/common/states";
import { describeDataSource } from "@/lib/dataSource";
import { formatPercent } from "@/lib/utils";

export default function LandingPage() {
  const dataset = useDataset();
  const pca = usePcaSummary();
  const sourceText = dataset.data ? (dataset.data.is_synthetic ? "Synthetic" : describeDataSource(dataset.data).label) : dataset.loading ? "checking..." : "unavailable (backend not reachable)";

  return (
    <div className="grid items-center gap-10 py-4 lg:grid-cols-2 lg:py-10">
      <div>
        <h1 className="font-serif text-4xl font-medium leading-[1.1] tracking-tight sm:text-5xl">
          Intelligent Multi-Cap
          <br />
          Stock Basket Construction
        </h1>
        <p className="mt-5 max-w-xl text-lg text-ink-2">Learning the latent structure of financial markets using PCA, LDA and behavioural similarity.</p>
        <p className="mt-4 text-sm text-ink-2">
          Research &amp; Educational System. Current dataset: <span className="font-medium text-ink">{sourceText}</span>. It does not predict prices and is not financial advice.
        </p>
        <Link href="/dashboard" className="mt-7 inline-flex items-center rounded-md bg-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-[#173a57]">
          Open Research Dashboard
        </Link>
      </div>

      <figure className="rounded-md border border-line bg-surface p-5">
        <figcaption className="mb-3">
          <div className="font-serif text-lg font-medium">What PCA found in the features</div>
          <div className="text-sm text-muted">Share of variation captured by each component, read live from the API.</div>
        </figcaption>
        {pca.loading && <Skeleton className="h-64 w-full" />}
        {pca.data && (
          <>
            <ScreePlot names={pca.data.component_names} explained={pca.data.explained_variance} cumulative={pca.data.cumulative_variance} height={240} />
            <p className="mt-2 text-sm text-ink-2">
              {pca.data.components} components together hold {formatPercent(pca.data.cumulative_variance[pca.data.cumulative_variance.length - 1])} of the variation.
            </p>
          </>
        )}
        {!pca.loading && pca.error !== undefined && <p className="text-sm text-muted">Start the backend to see live analysis here. Run: uvicorn api.main:app --reload</p>}
      </figure>
    </div>
  );
}
