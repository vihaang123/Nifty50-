import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import LdaPage from "@/app/lda/page";
import PcaPage from "@/app/pca/page";
import SimilarityPage from "@/app/similarity/page";
import { AppShell } from "@/components/layout/AppShell";
import { apiError, callsTo, mockFetch } from "./helpers";

describe("error handling", () => {
  it("explains a backend outage in plain words and recovers on Retry", async () => {
    let down = true;
    const fetchMock = mockFetch((url) => {
      if (down && url.pathname === "/api/analysis/pca") throw new TypeError("Failed to fetch");
    });
    render(<PcaPage />);
    expect(screen.getByText("Loading PCA analysis...")).toBeInTheDocument();
    expect(await screen.findByText("Unable to load PCA analysis. Please check that the backend is running.")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/TypeError|Failed to fetch|stack/i);

    down = false;
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Feature loadings")).toBeInTheDocument();
    expect(callsTo(fetchMock, "/api/analysis/pca")).toHaveLength(2);
  });

  it("shows a server error without leaking details", async () => {
    mockFetch((url) => (url.pathname === "/api/analysis/lda" ? apiError(500, "internal_error", "Internal Server Error: Traceback (most recent call last)") : undefined));
    render(<LdaPage />);
    expect(await screen.findByText("The backend hit a problem while loading LDA analysis. Please try again.")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("Traceback");
  });

  it("shows the backend's message for an unknown stock", async () => {
    mockFetch((url) => (url.pathname.startsWith("/api/similarity/") ? apiError(404, "not_found", "Stock 'XYZ' is not in the universe.") : undefined));
    render(<SimilarityPage />);
    expect(await screen.findByText("Stock 'XYZ' is not in the universe.")).toBeInTheDocument();
  });

  it("loads the shell and navigation even when the backend is down", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    render(<AppShell>page body</AppShell>);
    expect((await screen.findAllByText("Backend unavailable")).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Backtest" }).length).toBeGreaterThan(0);
    expect(screen.getByText("page body")).toBeInTheDocument();
  });

  it("always labels the data source as synthetic when the API says so", async () => {
    mockFetch();
    render(<AppShell>x</AppShell>);
    expect((await screen.findAllByText("Synthetic Dataset")).length).toBeGreaterThan(0);
    expect(screen.getByText("Synthetic Research Dataset")).toBeInTheDocument();
    expect(screen.getByText("Financial Market Intelligence")).toBeInTheDocument();
  });
});
