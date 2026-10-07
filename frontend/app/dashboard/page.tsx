"use client";
import { SimilarityExplorer } from "@/components/analysis/SimilarityExplorer";
import { DatasetCard, LdaSummaryCard, PcaSummaryCard, UniverseCard } from "@/components/dashboard/Summaries";
import { QuickBacktest } from "@/components/dashboard/QuickBacktest";
import { QuickBasket } from "@/components/dashboard/QuickBasket";
import { PageHeader, Panel } from "@/components/common/ui";

export default function DashboardPage() {
  return (
    <>
      <PageHeader title="Research Dashboard" subtitle="Everything below is read from the research API. Nothing is calculated in the browser." />
      <div className="space-y-5">
        <div className="grid gap-5 lg:grid-cols-2">
          <DatasetCard />
          <UniverseCard />
        </div>
        <div className="grid gap-5 lg:grid-cols-2">
          <PcaSummaryCard />
          <LdaSummaryCard />
        </div>
        <Panel title="Stock similarity" description="Pick a stock to see the five stocks that have behaved most like it.">
          <SimilarityExplorer showChart={false} />
        </Panel>
        <QuickBasket />
        <QuickBacktest />
      </div>
    </>
  );
}
