import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import BasketPage from "@/app/basket/page";
import { BasketForm } from "@/components/basket/BasketForm";
import { BasketResult } from "@/components/basket/BasketResult";
import { validateBasketForm } from "@/lib/validation";
import { apiError, callsTo, fx, mockFetch } from "./helpers";

describe("basket validation", () => {
  const ok = { capital: "50000", basketSize: "6", threshold: "0.90" };
  it("accepts good values and builds the request payload", () => {
    expect(validateBasketForm(ok, 10).payload).toEqual({ capital: 50000, basket_size: 6, similarity_threshold: 0.9 });
  });
  it.each([
    [{ ...ok, basketSize: "2" }, "basketSize", "Basket size must be between 3 and 10."],
    [{ ...ok, basketSize: "11" }, "basketSize", "Basket size must be between 3 and 10."],
    [{ ...ok, basketSize: "4.5" }, "basketSize", "Basket size must be between 3 and 10."],
    [{ ...ok, capital: "0" }, "capital", "Capital must be a number greater than 0."],
    [{ ...ok, capital: "" }, "capital", "Capital must be a number greater than 0."],
    [{ ...ok, threshold: "1.5" }, "threshold", "Similarity threshold must be between -1 and 1."],
  ])("rejects %j", (values, field, message) => {
    const result = validateBasketForm(values, 10);
    expect(result.payload).toBeUndefined();
    expect(result.errors[field]).toBe(message);
  });
});

describe("BasketForm", () => {
  it("shows the validation message and does not submit an invalid basket size", async () => {
    const onSubmit = vi.fn();
    render(<BasketForm onSubmit={onSubmit} maxSize={10} />);
    const size = screen.getByLabelText("Basket size");
    await userEvent.clear(size);
    await userEvent.type(size, "2");
    await userEvent.click(screen.getByRole("button", { name: "Generate Basket" }));
    expect(screen.getByText("Basket size must be between 3 and 10.")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits the typed values", async () => {
    const onSubmit = vi.fn();
    render(<BasketForm onSubmit={onSubmit} maxSize={10} />);
    await userEvent.click(screen.getByRole("button", { name: "Generate Basket" }));
    expect(onSubmit).toHaveBeenCalledWith({ capital: 50000, basket_size: 6, similarity_threshold: 0.9 });
  });

  it("warns that a basket the size of the universe makes selection meaningless", async () => {
    render(<BasketForm onSubmit={vi.fn()} maxSize={10} />);
    const size = screen.getByLabelText("Basket size");
    await userEvent.clear(size);
    await userEvent.type(size, "10");
    expect(screen.getByText(/whole universe/)).toBeInTheDocument();
  });

  it("disables the button while a basket is being generated", () => {
    render(<BasketForm onSubmit={vi.fn()} loading />);
    expect(screen.getByRole("button", { name: "Generating basket..." })).toBeDisabled();
  });
});

describe("BasketResult", () => {
  it("renders every stock row, the weights and the pairwise diagnostics from the API", () => {
    render(<BasketResult basket={fx.basket} universeSize={10} />);
    const table = screen.getByRole("table");
    for (const stock of fx.basket.stocks) {
      const row = within(table).getByText(stock.symbol).closest("tr")!;
      expect(within(row).getByText(stock.cap_category)).toBeInTheDocument();
      expect(within(row).getByText(stock.behavior_class)).toBeInTheDocument();
      expect(within(row).getByText(stock.reason)).toBeInTheDocument();
    }
    expect(screen.getByText("Average pairwise similarity")).toBeInTheDocument();
    expect(screen.getByText(fx.basket.statistics.average_similarity.toFixed(2))).toBeInTheDocument();
    expect(screen.getByText(fx.basket.statistics.maximum_pair.join(" and "))).toBeInTheDocument();
    expect(screen.getByText("Cap distribution")).toBeInTheDocument();
    expect(screen.getByText("Behaviour distribution")).toBeInTheDocument();
  });

  it("always shows backend notes instead of hiding them", () => {
    render(<BasketResult basket={{ ...fx.basket, notes: ["Only 2 Small Cap stocks passed the similarity limit."] }} universeSize={10} />);
    expect(screen.getByText("Only 2 Small Cap stocks passed the similarity limit.")).toBeInTheDocument();
  });

  it("explains when the basket is simply the whole universe", () => {
    render(<BasketResult basket={fx.basket} universeSize={fx.basket.stocks.length} />);
    expect(screen.getByText(/contains every stock in the universe/)).toBeInTheDocument();
  });

  it("handles an empty basket without crashing", () => {
    render(<BasketResult basket={{ ...fx.basket, stocks: [] }} universeSize={10} />);
    expect(screen.getByText(/empty basket/)).toBeInTheDocument();
  });
});

describe("Basket page", () => {
  it("starts empty, generates through the API and shows the result", async () => {
    const fetchMock = mockFetch();
    render(<BasketPage />);
    expect(screen.getByText(/No basket yet/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Generate Basket" }));
    expect(await screen.findByText("Average pairwise similarity")).toBeInTheDocument();
    expect(JSON.parse(String(callsTo(fetchMock, "/api/basket/generate")[0][1]?.body))).toEqual({ capital: 50000, basket_size: 6, similarity_threshold: 0.9 });
  });

  it("sends changed parameters to the backend", async () => {
    const fetchMock = mockFetch();
    render(<BasketPage />);
    const capital = screen.getByLabelText("Capital (₹)");
    await userEvent.clear(capital);
    await userEvent.type(capital, "125000");
    const size = screen.getByLabelText("Basket size");
    await userEvent.clear(size);
    await userEvent.type(size, "4");
    await userEvent.click(screen.getByRole("button", { name: "Generate Basket" }));
    await waitFor(() => expect(callsTo(fetchMock, "/api/basket/generate")).toHaveLength(1));
    expect(JSON.parse(String(callsTo(fetchMock, "/api/basket/generate")[0][1]?.body))).toMatchObject({ capital: 125000, basket_size: 4 });
  });

  it("shows the backend's message for a rejected basket, with no internals", async () => {
    mockFetch((url) => (url.pathname === "/api/basket/generate" ? apiError(400, "invalid_request", "Basket size 8 is larger than the 6 stocks with data.") : undefined));
    render(<BasketPage />);
    await userEvent.click(screen.getByRole("button", { name: "Generate Basket" }));
    expect(await screen.findByText("Basket size 8 is larger than the 6 stocks with data.")).toBeInTheDocument();
  });

  it("maps a 422 field error onto the form field", async () => {
    mockFetch((url) => (url.pathname === "/api/basket/generate" ? apiError(422, "validation_error", "Check the highlighted fields.", [{ field: "body.similarity_threshold", message: "Input should be less than or equal to 1" }]) : undefined));
    render(<BasketPage />);
    await userEvent.click(screen.getByRole("button", { name: "Generate Basket" }));
    expect(await screen.findByText("Input should be less than or equal to 1")).toBeInTheDocument();
  });
});
