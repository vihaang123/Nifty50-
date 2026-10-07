"use client";
import { BasketForm } from "@/components/basket/BasketForm";
import { BasketResult } from "@/components/basket/BasketResult";
import { ErrorState, LoadingBlock, EmptyState } from "@/components/common/states";
import { PageHeader, Panel } from "@/components/common/ui";
import { ApiError, generateBasket, getUniverse } from "@/lib/api";
import { useAction, useRequest } from "@/lib/hooks";
import { describeError } from "@/lib/utils";
import { mapServerErrors } from "@/lib/validation";

export default function BasketPage() {
  const universe = useRequest(getUniverse, "universe");
  const { state, run } = useAction(generateBasket);
  const maxSize = universe.data?.stocks.length;
  const serverErrors = state.status === "error" && state.error instanceof ApiError ? mapServerErrors(state.error.details) : undefined;

  return (
    <>
      <PageHeader title="Basket Generator" subtitle="Build a diversified multi-cap basket from the stock universe using behaviour classes and similarity." />
      <div className="space-y-5">
        <Panel title="Basket settings" description="Constructed on the full synthetic history (exploratory), not a walk-forward estimate.">
          <BasketForm onSubmit={run} loading={state.status === "loading"} maxSize={maxSize} serverErrors={serverErrors} />
        </Panel>

        {universe.error !== undefined && <ErrorState message={describeError(universe.error, "the stock universe")} onRetry={universe.reload} />}
        {state.status === "idle" && <EmptyState>No basket yet. Choose your settings and generate a basket to see the stocks, weights and diagnostics.</EmptyState>}
        {state.status === "loading" && <LoadingBlock message="Generating basket..." lines={4} />}
        {state.status === "error" && <ErrorState message={describeError(state.error, "the basket")} />}
        {state.status === "success" && <BasketResult basket={state.data} universeSize={maxSize} />}
      </div>
    </>
  );
}
