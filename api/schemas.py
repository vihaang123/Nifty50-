"""
Request and response contracts (Pydantic). The Phase 8B frontend is built against exactly these shapes.

Conventions: dates are 'YYYY-MM-DD' strings; fractions are fractions (0.25 = 25%) unless a field says percentage;
nothing is NaN or Infinity (undefined numbers are null).
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


# ----------------------------------------------------------------------------- shared
class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    status: int
    code: str = Field(description="Short machine-readable code, e.g. unknown_symbol")
    message: str = Field(description="Human-readable explanation")
    details: Optional[list[ErrorDetail]] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ----------------------------------------------------------------------------- health, dataset, universe
class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class DatasetResponse(BaseModel):
    source: str = Field(description="'synthetic' for the development data, otherwise 'local'")
    is_synthetic: bool = Field(description="True while the data is the generated development data, not real market prices")
    start_date: str
    end_date: str
    stock_count: int
    stocks: list[str]
    trading_days: int
    market_index: str = Field(description="Name of the benchmark series in the data")
    notice: str


class UniverseStock(BaseModel):
    symbol: str
    cap_category: Literal["Large Cap", "Mid Cap", "Small Cap"]


class UniverseResponse(BaseModel):
    stocks: list[UniverseStock]
    counts: dict[str, int] = Field(description="Number of stocks per cap category")
    is_synthetic: bool
    notice: str


# ----------------------------------------------------------------------------- analysis
class LoadingRow(BaseModel):
    feature: str
    loadings: dict[str, float] = Field(description="Component name -> loading, e.g. {'PC1': 0.31, 'PC2': -0.12}")


class PcaObservation(BaseModel):
    date: str
    symbol: str
    scores: dict[str, float] = Field(description="Component name -> score for this stock on this date")


class PcaResponse(BaseModel):
    components: int
    component_names: list[str]
    explained_variance: list[float] = Field(description="Share of total variance per component (fractions)")
    cumulative_variance: list[float]
    loadings: list[LoadingRow]
    observations: list[PcaObservation] = Field(description="A reproducible sample of the stock-day scores (see total_observations)")
    total_observations: int
    returned_observations: int
    n_training_rows: int
    mode: str
    notice: str


class LdaClass(BaseModel):
    name: Literal["Defensive", "Balanced", "Aggressive"]
    count: int
    percentage: float = Field(description="0-100")


class LdaPoint(BaseModel):
    date: str
    symbol: str
    behavior_class: str
    LD1: float
    LD2: float


class LdaResponse(BaseModel):
    components: int
    classes: list[LdaClass]
    training_accuracy: float = Field(description="Accuracy on the data the model was fitted on, NOT out-of-sample performance")
    majority_baseline: float = Field(description="Accuracy of always guessing the largest class")
    explained_variance: dict[str, float]
    loadings: list[LoadingRow]
    points: list[LdaPoint] = Field(description="A reproducible sample of the stock-day LDA scores")
    total_points: int
    returned_points: int
    mode: str
    notice: str


class StockInfo(BaseModel):
    symbol: str
    cap_category: Optional[str] = Field(None, description="From the universe table; null if the stock is not in it")
    behavior_class: Optional[str] = Field(None, description="The stock's most frequent LDA class over the history (what the basket generator uses)")


class SimilarStock(StockInfo):
    rank: int
    similarity: float = Field(description="Cosine similarity of the average PCA profiles, -1 to 1")


class SimilarityResponse(BaseModel):
    symbol: str
    selected: StockInfo
    top_n: int
    similar_stocks: list[SimilarStock]
    mode: str
    notice: str


# ----------------------------------------------------------------------------- basket
class BasketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capital: float = Field(100_000, gt=0, le=1e12, allow_inf_nan=False, description="Rupees to allocate")
    basket_size: StrictInt = Field(10, ge=3, le=500, description="Number of stocks (at least 3, at most the number available)")
    similarity_threshold: float = Field(0.90, ge=-1.0, le=1.0, allow_inf_nan=False, description="Pairs more similar than this are penalised")


class BasketStock(BaseModel):
    symbol: str
    cap_category: str
    behavior_class: str
    weight: float
    allocation: float
    reason: str


class Distribution(BaseModel):
    count: int
    percentage: float = Field(description="0-100")


class BasketStatistics(BaseModel):
    behavior_distribution: dict[str, Distribution]
    cap_distribution: dict[str, Distribution]
    average_similarity: float
    maximum_similarity: float
    maximum_pair: list[str]
    pairs_above_threshold: int
    n_pairs: int


class BasketResponse(BaseModel):
    capital: float
    basket_size: int
    similarity_threshold: float
    total_weight: float
    total_allocation: float = Field(description="Sum of the allocations, rounded to 6 decimals")
    stocks: list[BasketStock]
    statistics: BasketStatistics
    target_allocation: dict[str, int] = Field(description="Wanted number of stocks per cap category (40/30/30)")
    planned_allocation: dict[str, int] = Field(description="Number per cap category after limits on supply")
    notes: list[str]
    n_eligible: int
    is_synthetic: bool
    mode: str
    notice: str


# ----------------------------------------------------------------------------- backtest
class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capital: float = Field(100_000, gt=0, le=1e12, allow_inf_nan=False)
    basket_size: StrictInt = Field(10, ge=3, le=500)
    frequency: Literal["monthly", "quarterly", "semiannual", "annual"] = "quarterly"
    similarity_threshold: float = Field(0.90, ge=-1.0, le=1.0, allow_inf_nan=False)
    start_date: Optional[date] = Field(None, description="First allowed rebalance date (training still uses all earlier data)")
    end_date: Optional[date] = Field(None, description="Last day of the test")

    @model_validator(mode="after")
    def _dates_in_order(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        return self


class StrategySummary(BaseModel):
    initial_capital: float
    final_portfolio_value: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: Optional[float] = Field(description="Risk-free rate = 0. Null if volatility is 0")
    max_drawdown: float


class BenchmarkSummary(BaseModel):
    final_value: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: Optional[float]
    max_drawdown: float


class EquityPoint(BaseModel):
    date: str
    portfolio_value: float
    daily_return: float
    benchmark_value: float
    benchmark_return: float


class DrawdownPoint(BaseModel):
    date: str
    portfolio_drawdown: float
    benchmark_drawdown: float


class RebalanceRow(BaseModel):
    rebalance_date: str
    symbol: str
    cap_category: str
    behavior_class: str
    weight: float
    allocation: float
    reason: str


class RebalanceInfo(BaseModel):
    rebalance_date: str
    period_start: str
    period_end: str
    status: str = Field(description="rebalanced, skipped_kept_previous_basket or skipped_no_basket")
    n_holdings: int
    symbols: list[str]
    last_training_date: Optional[str] = None
    n_training_rows: Optional[int] = None
    capital: Optional[float] = None
    reason: Optional[str] = Field(None, description="Why a rebalance was skipped")


class BacktestResponse(BaseModel):
    summary: StrategySummary
    benchmark: BenchmarkSummary
    equity_curve: list[EquityPoint]
    drawdown: list[DrawdownPoint]
    rebalance_history: list[RebalanceRow]
    rebalances: list[RebalanceInfo]
    skipped_rebalances: list[dict]
    missing_data_events: list[dict]
    data_info: dict
    settings: dict
    is_synthetic: bool
    benchmark_name: str
    notices: list[str]
