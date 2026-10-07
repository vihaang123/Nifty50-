import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import DashboardPage from "@/app/dashboard/page";
import { callsTo, fx, mockFetch } from "./helpers";

describe("Dashboard", () => {
  it("shows loading states first, then the numbers returned by the API", async () => {
    mockFetch();
    render(<DashboardPage />);
    expect(screen.getByText("Loading dataset...")).toBeInTheDocument();

    expect(await screen.findByText("Synthetic Research Dataset")).toBeInTheDocument();
    expect(screen.getByText(String(fx.dataset.trading_days).replace(/\B(?=(\d{2})*\d{3}$)/g, ","))).toBeInTheDocument();
    expect(screen.getByText("Large Cap")).toBeInTheDocument();
    expect(await screen.findByText(/Training accuracy on constructed behavioural classes/)).toBeInTheDocument();
    expect(screen.getAllByText(/PC1 explained/).length).toBeGreaterThan(0);
  });

  it("does not run the expensive backtest until the button is pressed, then sends the stated settings", async () => {
    const fetchMock = mockFetch();
    render(<DashboardPage />);
    await screen.findByText("Synthetic Research Dataset");
    expect(callsTo(fetchMock, "/api/backtest")).toHaveLength(0);

    await userEvent.click(screen.getByRole("button", { name: "Run Quick Backtest" }));
    await waitFor(() => expect(callsTo(fetchMock, "/api/backtest")).toHaveLength(1));
    const body = JSON.parse(String(callsTo(fetchMock, "/api/backtest")[0][1]?.body));
    expect(body).toEqual({ capital: 50000, basket_size: 6, frequency: "quarterly", similarity_threshold: 0.9 });
    expect(await screen.findByText("Final Portfolio Value")).toBeInTheDocument();
  });

  it("keeps working when one card fails: the others still render", async () => {
    mockFetch((url) => (url.pathname === "/api/analysis/lda" ? new Response("{}", { status: 500 }) : undefined));
    render(<DashboardPage />);
    expect(await screen.findByText("Synthetic Research Dataset")).toBeInTheDocument();
    expect(await screen.findByText(/The backend hit a problem while loading LDA analysis/)).toBeInTheDocument();
    expect(screen.getByText("Stock universe")).toBeInTheDocument();
  });
});
