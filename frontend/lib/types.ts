/**
 * TypeScript types for the FastAPI contracts. They mirror api/schemas.py (Phase 8A, plus the cap_category /
 * behavior_class fields on the similarity response) and the real JSON the backend returns.
 * Dates are 'YYYY-MM-DD' strings. Fractions are fractions (0.25 = 25%) unless the field says percentage.
 */

// ---- errors ---------------------------------------------------------------
export interface ErrorDetail {
  field: string;
  message: string;
}
export interface ErrorBody {
  status: number;
  code: string;
  message: string;
  details?: ErrorDetail[] | null;
}
export interface ErrorResponse {
  error: ErrorBody;
}

// ---- health, dataset, universe ---------------------------------------------
export interface DataHealth {
  provider: string;
  available: boolean;
  is_synthetic: boolean | null;
  message: string | null;
}

export interface HealthResponse {
  /** "ok" when the API and its data are ready, "degraded" when the API is up but the data provider is not. */
  status: "ok" | "degraded";
  service: string;
  version: string;
  environment: "development" | "production";
  data: DataHealth;
}

export interface DatasetResponse {
  /** The data provider: "local" today, "angel_one" from Phase 8E. */
  provider: string;
  source: string;
  is_synthetic: boolean;
  start_date: string;
  end_date: string;
  stock_count: number;
  stocks: string[];
  trading_days: number;
  /** Stock-day price rows loaded (before the feature warm-up rows are removed). */
  observations: number;
  market_index: string;
  notice: string;
}

export type CapCategory = "Large Cap" | "Mid Cap" | "Small Cap";
export type BehaviorClass = "Defensive" | "Balanced" | "Aggressive";

export interface UniverseStock {
  symbol: string;
  cap_category: CapCategory;
}
export interface UniverseResponse {
  stocks: UniverseStock[];
  counts: Record<string, number>;
  is_synthetic: boolean;
  notice: string;
}

// ---- analysis -------------------------------------------------------------
export interface LoadingRow {
  feature: string;
  loadings: Record<string, number>;
}

export interface PcaObservation {
  date: string;
  symbol: string;
  scores: Record<string, number>;
}
export interface PcaResponse {
  components: number;
  component_names: string[];
  explained_variance: number[];
  cumulative_variance: number[];
  loadings: LoadingRow[];
  observations: PcaObservation[];
  total_observations: number;
  returned_observations: number;
  n_training_rows: number;
  mode: string;
  notice: string;
}

export interface LdaClass {
  name: BehaviorClass;
  count: number;
  percentage: number; // 0-100
}
export interface LdaPoint {
  date: string;
  symbol: string;
  behavior_class: string;
  LD1: number;
  LD2: number;
}
export interface LdaResponse {
  components: number;
  classes: LdaClass[];
  training_accuracy: number;
  majority_baseline: number;
  explained_variance: Record<string, number>;
  loadings: LoadingRow[];
  points: LdaPoint[];
  total_points: number;
  returned_points: number;
  mode: string;
  notice: string;
}

// ---- similarity -----------------------------------------------------------
export interface StockInfo {
  symbol: string;
  cap_category: string | null;
  behavior_class: string | null;
}
export interface SimilarStock extends StockInfo {
  rank: number;
  similarity: number;
}
export interface SimilarityResponse {
  symbol: string;
  selected: StockInfo;
  top_n: number;
  similar_stocks: SimilarStock[];
  mode: string;
  notice: string;
}

// ---- basket ---------------------------------------------------------------
export interface BasketRequest {
  capital: number;
  basket_size: number;
  similarity_threshold: number;
}
export interface BasketStock {
  symbol: string;
  cap_category: string;
  behavior_class: string;
  weight: number;
  allocation: number;
  reason: string;
}
export interface Distribution {
  count: number;
  percentage: number; // 0-100
}
export interface BasketStatistics {
  behavior_distribution: Record<string, Distribution>;
  cap_distribution: Record<string, Distribution>;
  average_similarity: number;
  maximum_similarity: number;
  maximum_pair: string[];
  pairs_above_threshold: number;
  n_pairs: number;
}
export interface BasketResponse {
  capital: number;
  basket_size: number;
  similarity_threshold: number;
  total_weight: number;
  total_allocation: number;
  stocks: BasketStock[];
  statistics: BasketStatistics;
  target_allocation: Record<string, number>;
  planned_allocation: Record<string, number>;
  notes: string[];
  n_eligible: number;
  is_synthetic: boolean;
  mode: string;
  notice: string;
}

// ---- backtest -------------------------------------------------------------
export type RebalanceFrequency = "monthly" | "quarterly" | "semiannual" | "annual";

export interface BacktestRequest {
  capital: number;
  basket_size: number;
  frequency: RebalanceFrequency;
  similarity_threshold: number;
  start_date?: string | null;
  end_date?: string | null;
}
export interface StrategySummary {
  initial_capital: number;
  final_portfolio_value: number;
  cumulative_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number | null;
  max_drawdown: number;
}
export interface BenchmarkSummary {
  final_value: number;
  cumulative_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number | null;
  max_drawdown: number;
}
export interface EquityPoint {
  date: string;
  portfolio_value: number;
  daily_return: number;
  benchmark_value: number;
  benchmark_return: number;
}
export interface DrawdownPoint {
  date: string;
  portfolio_drawdown: number;
  benchmark_drawdown: number;
}
export interface RebalanceRow {
  rebalance_date: string;
  symbol: string;
  cap_category: string;
  behavior_class: string;
  weight: number;
  allocation: number;
  reason: string;
}
export interface RebalanceInfo {
  rebalance_date: string;
  period_start: string;
  period_end: string;
  status: "rebalanced" | "skipped_kept_previous_basket" | "skipped_no_basket" | string;
  n_holdings: number;
  symbols: string[];
  last_training_date?: string | null;
  n_training_rows?: number | null;
  capital?: number | null;
  reason?: string | null;
}
export interface SkippedRebalance {
  rebalance_date: string;
  reason: string;
}
export interface MissingDataEvent {
  rebalance_date: string;
  symbol: string;
  stop_date: string;
}
export interface BacktestDataInfo {
  start_date: string;
  end_date: string;
  trading_days: number;
  n_stocks: number;
  possible_periods: number;
  n_rebalance_dates: number;
  first_rebalance: string;
  last_rebalance: string;
  backtest_start: string;
  backtest_end: string;
  baseline_date: string;
  years: number;
  days_without_holdings: number;
}
export interface BacktestSettings {
  capital: number;
  basket_size: number;
  frequency: RebalanceFrequency;
  n_components: number;
  similarity_threshold: number;
  min_history_days: number;
  transaction_cost: number;
  slippage: number;
  risk_free_rate: number;
}
export interface BacktestResponse {
  summary: StrategySummary;
  benchmark: BenchmarkSummary;
  equity_curve: EquityPoint[];
  drawdown: DrawdownPoint[];
  rebalance_history: RebalanceRow[];
  rebalances: RebalanceInfo[];
  skipped_rebalances: SkippedRebalance[];
  missing_data_events: MissingDataEvent[];
  data_info: BacktestDataInfo;
  settings: BacktestSettings;
  is_synthetic: boolean;
  benchmark_name: string;
  notices: string[];
}
