"use client";
import { BacktestForm } from "@/components/backtest/BacktestForm";
import { BacktestResults } from "@/components/backtest/BacktestResults";
import { EmptyState, ErrorState } from "@/components/common/states";
import { Notice, PageHeader, Panel } from "@/components/common/ui";
import { ApiError, getDataset, getUniverse, runBacktest } from "@/lib/api";
import { useAction, useRequest } from "@/lib/hooks";
import { describeError } from "@/lib/utils";
import { mapServerErrors } from "@/lib/validation";

export default function BacktestPage() {
  const universe = useRequest(getUniverse, "universe");
  const dataset = useRequest(getDataset, "dataset");
  const { state, run, cancel, elapsed } = useAction(runBacktest);
  const serverErrors = state.status === "error" && state.error instanceof ApiError ? mapServerErrors(state.error.details) : undefined;

  return (
    <>
      <PageHeader title="Walk-Forward Backtest" subtitle="Rebuild the basket on a schedule using only the data available at each date, then follow the portfolio forward." />
      <div className="space-y-5">
        <Panel title="Backtest settings">
          <BacktestForm
            onSubmit={run}
            onCancel={cancel}
            loading={state.status === "loading"}
            elapsed={elapsed}
            maxSize={universe.data?.stocks.length}
            minDate={dataset.data?.start_date}
            maxDate={dataset.data?.end_date}
            serverErrors={serverErrors}
          />
        </Panel>

        {state.status === "idle" && <EmptyState>No backtest has been run yet. Configure your parameters and run a backtest to see results.</EmptyState>}
        {state.status === "loading" && <Notice>The backtest refits the models at every rebalance date, so it is much slower than the other pages. Please keep this page open.</Notice>}
        {state.status === "error" && <ErrorState message={describeError(state.error, "the backtest")} />}
        {state.status === "success" && <BacktestResults result={state.data} />}
      </div>
    </>
  );
}
