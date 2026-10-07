import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import BacktestPage from "@/app/backtest/page";
import { BacktestResults } from "@/components/backtest/BacktestResults";
import { validateBacktestForm } from "@/lib/validation";
import { apiError, callsTo, fx, mockFetch } from "./helpers";

const run = () => userEvent.click(screen.getByRole("button", { name: "Run Backtest" }));

describe("Backtest page", () => {
  it("starts with an empty state and runs nothing automatically", async () => {
    const fetchMock = mockFetch();
    render(<BacktestPage />);
    expect(screen.getByText("No backtest has been run yet. Configure your parameters and run a backtest to see results.")).toBeInTheDocument();
    await screen.findByLabelText("Start date");
    expect(callsTo(fetchMock, "/api/backtest")).toHaveLength(0);
  });

  it("shows a loading state and disables the button while a slow backtest runs", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    mockFetch(async (url) => {
      if (url.pathname === "/api/backtest") {
        await gate;
        return undefined; // fall through to the default fixture
      }
    });
    render(<BacktestPage />);
    await run();

    expect(await screen.findByText(/Running walk-forward backtest\.\.\./)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Running backtest..." })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();
    expect(screen.queryByText("Final Portfolio Value")).not.toBeInTheDocument();

    release();
    expect(await screen.findByText("Final Portfolio Value")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Backtest" })).toBeEnabled();
  });

  it("sends the chosen parameters to the backend", async () => {
    const fetchMock = mockFetch();
    render(<BacktestPage />);
    await userEvent.type(screen.getByLabelText("Start date"), "2019-01-01");
    await userEvent.type(screen.getByLabelText("End date"), "2020-06-30");
    await userEvent.selectOptions(screen.getByLabelText("Rebalance frequency"), "semiannual");
    const size = screen.getByLabelText("Basket size");
    await userEvent.clear(size);
    await userEvent.type(size, "5");
    await run();
    await waitFor(() => expect(callsTo(fetchMock, "/api/backtest")).toHaveLength(1));
    expect(JSON.parse(String(callsTo(fetchMock, "/api/backtest")[0][1]?.body))).toEqual({
      capital: 100000,
      basket_size: 5,
      frequency: "semiannual",
      similarity_threshold: 0.9,
      start_date: "2019-01-01",
      end_date: "2020-06-30",
    });
  });

  it("blocks an end date before the start date without calling the backend", async () => {
    const fetchMock = mockFetch();
    render(<BacktestPage />);
    await userEvent.type(screen.getByLabelText("Start date"), "2020-01-01");
    await userEvent.type(screen.getByLabelText("End date"), "2019-01-01");
    await run();
    expect(screen.getByText("End date must be on or after the start date.")).toBeInTheDocument();
    expect(callsTo(fetchMock, "/api/backtest")).toHaveLength(0);
  });

  it("shows the backend's message when a backtest is rejected", async () => {
    mockFetch((url) => (url.pathname === "/api/backtest" ? apiError(400, "invalid_request", "No rebalance is possible with this date range.") : undefined));
    render(<BacktestPage />);
    await run();
    expect(await screen.findByText("No rebalance is possible with this date range.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Backtest" })).toBeEnabled();
  });
});

describe("BacktestResults", () => {
  it("renders the six metrics exactly as the API reports them", () => {
    render(<BacktestResults result={fx.backtest} />);
    const s = fx.backtest.summary;
    for (const label of ["Final Portfolio Value", "Cumulative Return", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Maximum Drawdown"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.getByText(`+${(s.cumulative_return * 100).toFixed(1)}%`)).toBeInTheDocument();
    expect(screen.getByText(s.sharpe_ratio!.toFixed(2))).toBeInTheDocument();
    expect(screen.getByText(`${(s.max_drawdown * 100).toFixed(1)}%`)).toBeInTheDocument();
    expect(screen.getByText(/synthetic research dataset/)).toBeInTheDocument();
  });

  it("lists every rebalance and reveals the stocks bought on demand", async () => {
    render(<BacktestResults result={fx.backtest} />);
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("row").length - 1).toBeGreaterThanOrEqual(fx.backtest.rebalances.length);
    const first = fx.backtest.rebalances[0];
    const firstHolding = fx.backtest.rebalance_history.find((h) => h.rebalance_date === first.rebalance_date)!;
    expect(screen.queryByText(firstHolding.reason)).not.toBeInTheDocument();
    await userEvent.click(screen.getAllByRole("button", { name: "Show" })[0]);
    expect(screen.getAllByText(firstHolding.symbol).length).toBeGreaterThan(0);
  });

  it("shows skipped rebalances and notices when the backend reports them", () => {
    render(<BacktestResults result={{ ...fx.backtest, notices: ["Costs are zero."], skipped_rebalances: [{ rebalance_date: "2019-04-01", reason: "Basket size larger than available stocks." }] }} />);
    expect(screen.getByText("Costs are zero.")).toBeInTheDocument();
    expect(screen.getByText(/Basket size larger than available stocks\./)).toBeInTheDocument();
  });

  it("handles an empty result without crashing", () => {
    render(<BacktestResults result={{ ...fx.backtest, equity_curve: [], drawdown: [] }} />);
    expect(screen.getByText(/returned no results/)).toBeInTheDocument();
  });
});

describe("backtest validation", () => {
  it("passes blank dates as null so the backend uses the full range", () => {
    const r = validateBacktestForm({ capital: "100000", basketSize: "10", threshold: "0.9", frequency: "quarterly", startDate: "", endDate: "" }, 10);
    expect(r.payload).toMatchObject({ start_date: null, end_date: null });
  });
});
