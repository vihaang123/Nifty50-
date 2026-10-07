import { describe, expect, it, vi } from "vitest";
import { ApiError, generateBasket, getDataset, getPCA, getSimilarity, runBacktest } from "@/lib/api";
import { apiError, callsTo, fx, mockFetch } from "./helpers";

describe("API client", () => {
  it("builds URLs from NEXT_PUBLIC_API_URL, never a hard-coded host", async () => {
    const fetchMock = mockFetch();
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://backend.example.com/");
    await getDataset();
    expect(String(fetchMock.mock.calls[0][0])).toBe("https://backend.example.com/api/dataset");
  });

  it("reports a clear configuration error when the URL is not set", async () => {
    mockFetch();
    vi.stubEnv("NEXT_PUBLIC_API_URL", "");
    await expect(getDataset()).rejects.toMatchObject({ kind: "config" });
  });

  it("sends query parameters for PCA and similarity", async () => {
    const fetchMock = mockFetch();
    await getPCA({ components: 3, maxPoints: 1 });
    await getSimilarity("SBIN", 5);
    const urls = fetchMock.mock.calls.map(([u]) => String(u));
    expect(urls).toContain("http://api.test/api/analysis/pca?components=3&max_points=1");
    expect(urls).toContain("http://api.test/api/similarity/SBIN?top_n=5");
  });

  it("shares one request between identical GETs and returns the same data", async () => {
    const fetchMock = mockFetch();
    const [a, b] = await Promise.all([getDataset(), getDataset()]);
    expect(a).toEqual(fx.dataset);
    expect(b).toBe(a);
    expect(callsTo(fetchMock, "/api/dataset")).toHaveLength(1);
  });

  it("does not cache a failed GET, so a retry asks the backend again", async () => {
    let first = true;
    const fetchMock = mockFetch((url) => {
      if (url.pathname === "/api/dataset" && first) {
        first = false;
        return apiError(500, "internal_error", "Something went wrong.");
      }
    });
    await expect(getDataset()).rejects.toBeInstanceOf(ApiError);
    await expect(getDataset()).resolves.toEqual(fx.dataset);
    expect(callsTo(fetchMock, "/api/dataset")).toHaveLength(2);
  });

  it("turns a network failure into a friendly error with no internals", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const error = await getDataset().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.kind).toBe("network");
    expect(error.message).toBe("Cannot reach the backend. Please check that it is running.");
  });

  it("keeps the backend's error code, status and field details", async () => {
    mockFetch(() => apiError(422, "validation_error", "Check the highlighted fields.", [{ field: "body.basket_size", message: "Input should be greater than or equal to 3" }]));
    const error = await generateBasket({ capital: 50000, basket_size: 2, similarity_threshold: 0.9 }).catch((e) => e);
    expect(error).toMatchObject({ kind: "http", status: 422, code: "validation_error" });
    expect(error.details[0].field).toBe("body.basket_size");
  });

  it("posts JSON bodies and sends no credentials or secret headers", async () => {
    const fetchMock = mockFetch();
    const payload = { capital: 50000, basket_size: 6, frequency: "quarterly" as const, similarity_threshold: 0.9 };
    await runBacktest(payload);
    const [, init] = fetchMock.mock.calls[0];
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual(payload);
    expect(Object.keys(init?.headers ?? {}).sort()).toEqual(["Accept", "Content-Type"]);
    expect(init?.credentials).toBeUndefined();
  });

  it("handles an unreadable success response", async () => {
    mockFetch(() => new Response("not json", { status: 200 }));
    await expect(getDataset()).rejects.toMatchObject({ kind: "invalid_response" });
  });

  it("handles an error response that is not JSON", async () => {
    mockFetch(() => new Response("<html>bad gateway</html>", { status: 502 }));
    const error = await getDataset().catch((e) => e);
    expect(error).toMatchObject({ kind: "http", status: 502 });
    expect(error.message).not.toContain("<html>");
  });
});
