import { vi } from "vitest";
import backtest from "./fixtures/backtest.json";
import basket from "./fixtures/basket.json";
import dataset from "./fixtures/dataset.json";
import lda from "./fixtures/lda.json";
import pca from "./fixtures/pca.json";
import similarity from "./fixtures/similarity.json";
import universe from "./fixtures/universe.json";
import type { BacktestResponse, BasketResponse, DatasetResponse, LdaResponse, PcaResponse, SimilarityResponse, UniverseResponse } from "@/lib/types";

// The fixtures are real responses captured from the FastAPI backend (not hand-written numbers).
export const fx = {
  dataset: dataset as DatasetResponse,
  universe: universe as UniverseResponse,
  pca: pca as unknown as PcaResponse,
  lda: lda as unknown as LdaResponse,
  similarity: similarity as unknown as SimilarityResponse,
  basket: basket as unknown as BasketResponse,
  backtest: backtest as unknown as BacktestResponse,
};

export const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
export const apiError = (status: number, code: string, message: string, details?: Array<{ field: string; message: string }>) => json({ error: { status, code, message, details } }, status);

export type Handler = (url: URL, init?: RequestInit) => Response | undefined | Promise<Response | undefined>;

/** Stubs fetch. Handlers are tried in order; the default answers each endpoint with its fixture. */
export function mockFetch(...handlers: Handler[]) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    for (const h of handlers) {
      const r = await h(url, init);
      if (r) return r;
    }
    const p = url.pathname;
    if (p === "/api/dataset") return json(fx.dataset);
    if (p === "/api/universe") return json(fx.universe);
    if (p === "/api/analysis/pca") return json(fx.pca);
    if (p === "/api/analysis/lda") return json(fx.lda);
    if (p.startsWith("/api/similarity/")) return json(fx.similarity);
    if (p === "/api/basket/generate") return json(fx.basket);
    if (p === "/api/backtest") return json(fx.backtest);
    if (p === "/api/health") return json({ status: "ok", service: "stock-basket-api", version: "1.0.0" });
    return apiError(404, "not_found", "Route not found.");
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

export const callsTo = (fn: ReturnType<typeof mockFetch>, path: string) => fn.mock.calls.filter(([u]) => new URL(String(u)).pathname === path);
