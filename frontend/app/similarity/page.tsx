"use client";
import { SimilarityExplorer } from "@/components/analysis/SimilarityExplorer";
import { PageHeader, Panel } from "@/components/common/ui";

export default function SimilarityPage() {
  return (
    <>
      <PageHeader title="Stock Behavioural Similarity" subtitle="Which stocks have behaved most like the one you choose, measured in PCA space." />
      <Panel>
        <SimilarityExplorer showChart />
      </Panel>
    </>
  );
}
