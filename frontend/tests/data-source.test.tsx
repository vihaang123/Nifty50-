import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AppShell } from "@/components/layout/AppShell";
import { DatasetCard } from "@/components/dashboard/Summaries";
import { describeDataSource } from "@/lib/dataSource";
import type { DatasetResponse } from "@/lib/types";
import { fx, json, mockFetch } from "./helpers";

const variant = (changes: Partial<DatasetResponse>): DatasetResponse => ({ ...fx.dataset, ...changes });
const ANGEL_ONE = variant({ provider: "angel_one", source: "angel_one", is_synthetic: false });
const LOCAL_REAL = variant({ provider: "local", source: "local", is_synthetic: false });

describe("describeDataSource", () => {
  it("calls the synthetic development data a Synthetic Research Dataset", () => {
    expect(describeDataSource(fx.dataset)).toEqual({ label: "Synthetic Research Dataset", badge: "Synthetic Dataset", synthetic: true });
  });

  it("is architecturally ready to say Angel One, using only what the API reports", () => {
    expect(describeDataSource(ANGEL_ONE)).toEqual({ label: "Angel One", badge: "Angel One", synthetic: false });
  });

  it("describes non-synthetic local files as a local dataset", () => {
    expect(describeDataSource(LOCAL_REAL).label).toBe("Local Dataset");
  });

  it("makes an unfamiliar provider readable instead of failing", () => {
    expect(describeDataSource(variant({ provider: "some_new_feed", source: "some_new_feed", is_synthetic: false })).label).toBe("Some New Feed");
    expect(describeDataSource(variant({ provider: "", source: "", is_synthetic: false })).label).toBe("Unknown source");
  });

  it("says synthetic whenever the API says so, whatever the provider is", () => {
    expect(describeDataSource(variant({ provider: "angel_one", is_synthetic: true })).synthetic).toBe(true);
  });

  it("never claims the data is live", () => {
    for (const d of [fx.dataset, ANGEL_ONE, LOCAL_REAL]) {
      const { label, badge } = describeDataSource(d);
      expect(`${label} ${badge}`).not.toMatch(/live/i);
    }
  });
});

describe("data-source indicator in the app shell", () => {
  it("shows Synthetic Dataset and Synthetic Research Dataset for the current synthetic provider", async () => {
    mockFetch();
    render(<AppShell>x</AppShell>);
    expect((await screen.findAllByText("Synthetic Dataset")).length).toBeGreaterThan(0);
    expect(screen.getByText("Synthetic Research Dataset")).toBeInTheDocument();
  });

  it("would read Angel One if the backend reported that provider (not enabled today)", async () => {
    mockFetch((url) => (url.pathname === "/api/dataset" ? json(ANGEL_ONE) : undefined));
    render(<AppShell>x</AppShell>);
    expect((await screen.findAllByText("Angel One")).length).toBeGreaterThan(0);
    expect(screen.queryByText(/synthetic/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/live/i)).not.toBeInTheDocument();
  });

  it("shows a neutral local label for non-synthetic local data", async () => {
    mockFetch((url) => (url.pathname === "/api/dataset" ? json(LOCAL_REAL) : undefined));
    render(<AppShell>x</AppShell>);
    expect((await screen.findAllByText("Local Dataset")).length).toBeGreaterThan(0);
    expect(screen.queryByText(/live/i)).not.toBeInTheDocument();
  });
});

describe("dataset card", () => {
  it("shows the data source row from the API response", async () => {
    mockFetch();
    render(<DatasetCard />);
    expect(await screen.findByText("Data source")).toBeInTheDocument();
    expect(screen.getByText("Synthetic Research Dataset")).toBeInTheDocument();
  });

  it("shows Angel One in the data source row when the API reports it", async () => {
    mockFetch((url) => (url.pathname === "/api/dataset" ? json(ANGEL_ONE) : undefined));
    render(<DatasetCard />);
    expect(await screen.findByText("Angel One")).toBeInTheDocument();
    expect(screen.queryByText("Synthetic Research Dataset")).not.toBeInTheDocument();
  });
});
