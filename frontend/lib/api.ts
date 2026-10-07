/**
 * The only place the frontend talks to the backend. A small typed fetch client over the FastAPI endpoints.
 * The base URL comes from NEXT_PUBLIC_API_URL (never hard-coded). Nothing here calculates anything.
 */
import type {
  BacktestRequest,
  BacktestResponse,
  BasketRequest,
  BasketResponse,
  DatasetResponse,
  ErrorDetail,
  HealthResponse,
  LdaResponse,
  PcaResponse,
  SimilarityResponse,
  UniverseResponse,
} from "./types";

export type ApiErrorKind = "config" | "network" | "http" | "invalid_response";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly code?: string;
  readonly details: ErrorDetail[];

  constructor(kind: ApiErrorKind, message: string, extra: { status?: number; code?: string; details?: ErrorDetail[] } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = extra.status;
    this.code = extra.code;
    this.details = extra.details ?? [];
  }
}

export function getApiBaseUrl(): string {
  const url = process.env.NEXT_PUBLIC_API_URL; // inlined at build time by Next.js
  if (!url || !url.trim()) {
    throw new ApiError(
      "config",
      "The backend address is not configured. Set NEXT_PUBLIC_API_URL (see frontend/.env.example) and restart the frontend.",
    );
  }
  return url.trim().replace(/\/+$/, "");
}

interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const base = getApiBaseUrl();
  let response: Response;
  try {
    response = await fetch(`${base}${path}`, {
      method: options.method ?? "GET",
      headers: options.body === undefined ? { Accept: "application/json" } : { Accept: "application/json", "Content-Type": "application/json" },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError("network", "Cannot reach the backend. Please check that it is running.");
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    payload = undefined;
  }

  if (!response.ok) {
    const body = (payload as { error?: { status?: number; code?: string; message?: string; details?: ErrorDetail[] | null } } | undefined)?.error;
    throw new ApiError("http", body?.message ?? "The backend could not complete the request.", {
      status: response.status,
      code: body?.code,
      details: body?.details ?? [],
    });
  }
  if (payload === undefined) throw new ApiError("invalid_response", "The backend returned a response the app could not read.", { status: response.status });
  return payload as T;
}

// GET results never change while the backend runs, so identical GETs share one request and one answer.
// Failures are not cached, so Retry really asks again.
const cache = new Map<string, Promise<unknown>>();

function cachedGet<T>(path: string): Promise<T> {
  const hit = cache.get(path);
  if (hit) return hit as Promise<T>;
  const pending = request<T>(path).catch((error) => {
    cache.delete(path);
    throw error;
  });
  cache.set(path, pending);
  return pending;
}

export function clearApiCache(): void {
  cache.clear();
}

export const getHealth = () => request<HealthResponse>("/api/health");
export const getDataset = () => cachedGet<DatasetResponse>("/api/dataset");
export const getUniverse = () => cachedGet<UniverseResponse>("/api/universe");

export function getPCA(options: { components?: number; maxPoints?: number } = {}) {
  const query = new URLSearchParams();
  if (options.components !== undefined) query.set("components", String(options.components));
  if (options.maxPoints !== undefined) query.set("max_points", String(options.maxPoints));
  const text = query.toString();
  return cachedGet<PcaResponse>(`/api/analysis/pca${text ? `?${text}` : ""}`);
}

export function getLDA(options: { maxPoints?: number } = {}) {
  const text = options.maxPoints === undefined ? "" : `?max_points=${options.maxPoints}`;
  return cachedGet<LdaResponse>(`/api/analysis/lda${text}`);
}

export const getSimilarity = (symbol: string, topN = 5) =>
  cachedGet<SimilarityResponse>(`/api/similarity/${encodeURIComponent(symbol)}?top_n=${topN}`);

export const generateBasket = (payload: BasketRequest, signal?: AbortSignal) =>
  request<BasketResponse>("/api/basket/generate", { method: "POST", body: payload, signal });

export const runBacktest = (payload: BacktestRequest, signal?: AbortSignal) =>
  request<BacktestResponse>("/api/backtest", { method: "POST", body: payload, signal });
