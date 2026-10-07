# Learning the Latent Structure of Financial Markets for Multi-Cap Stock Basket Construction (PCA + LDA)

BTech Data Science final project. We study Indian Large, Mid and Small Cap stocks and ask:

> Can PCA and LDA help us understand the behavioural structure of stocks and build a more diversified stock basket than simpler approaches?

We do **not** predict prices, generate buy/sell signals, or give financial advice. If a simple baseline beats PCA/LDA, we report that honestly.

## Pipeline

```
Historical data -> Features -> PCA -> LDA -> Similarity -> Selection -> Basket -> Backtest -> FastAPI -> Next.js dashboard
```

## Progress

- [x] Phase 1: data loader, validation, development dataset, tests
- [x] Phase 2: feature engineering
- [x] Phase 3: PCA
- [x] Phase 4: LDA and behavioural classes
- [x] Phase 5: stock similarity
- [x] Phase 6: basket construction
- [x] Phase 7: walk-forward backtesting
- [x] Phase 8A: FastAPI backend
- [x] Phase 8B: Next.js frontend
- [x] Phase 8C: Production deployment and integration
- [ ] Phase 8D: Production hardening / next deployment phase
- [ ] Phase 8E: Angel One / live market data integration
- [ ] Phase 9: testing and cleanup

Phase 8C is frozen: this repository's `main` branch is the source of truth for what is deployed.

## Production deployment (Phase 8C)

| Part | Technology | Hosted on | URL |
|---|---|---|---|
| Frontend | Next.js (App Router, React, TypeScript) | Vercel | https://nifty50-frontend.vercel.app |
| Backend | FastAPI (Python) | Vercel | https://nifty50-api.vercel.app |

Both are separate Vercel projects built from this one repository. The frontend is a presentation layer: it calls the backend and draws the result. All PCA, LDA, similarity, basket and backtest work happens in the existing Python research engine (`src/`) behind the API.

- **Current data source: Synthetic Research Dataset.** Every number in the deployed app comes from simulated prices (10 tickers, 2018 to 2025). It says nothing about real companies or real markets, and the UI labels it as synthetic.
- **Angel One integration: not yet implemented.** There is no live market data, no broker connection, no orders, no authentication and no database.
- This is a research and educational project. It is not a live trading system, does not predict prices and is not financial advice.

### Architecture

```
User
  |
Next.js Frontend            (frontend/, Vercel)
  |  HTTPS, JSON
Vercel
  |
FastAPI                     (api/, entry point index.py, Vercel)
  |
Existing Python Research Engine   (src/, unchanged since Phase 7)
  |
Features
  |
PCA / LDA
  |
Behavioural Similarity
  |
Basket Construction
  |
Walk-Forward Backtesting
```

The data source feeding the engine is currently the synthetic dataset in `data/raw/`.

### Vercel project settings

| | Frontend project (`nifty50-frontend`) | Backend project (`nifty50-api`) |
|---|---|---|
| Git repository | `vihaang123/Nifty50-`, branch `main` | same |
| Root Directory | `frontend` | `.` (repository root) |
| Framework | Next.js | FastAPI |
| Entry point | Next.js app in `frontend/app/` | `index.py`, which exposes `api.main:app` |
| Python version | not used | `.python-version` (3.13) |
| Dependencies | `frontend/package.json` | `requirements.txt` (runtime only) |

`.vercelignore` keeps tests, notebooks and generated files out of the backend bundle. It must **not** list `frontend/`: the frontend project builds from that folder, and hiding it breaks the frontend build.

### Environment variables

Set these in each Vercel project's settings. They are never committed, and the frontend never holds a secret.

| Project | Variable | Production value |
|---|---|---|
| Frontend | `NEXT_PUBLIC_API_URL` | `https://nifty50-api.vercel.app` |
| Backend | `ENVIRONMENT` | `production` |
| Backend | `FRONTEND_ORIGIN` | `https://nifty50-frontend.vercel.app` |

`NEXT_PUBLIC_API_URL` is read at build time, so changing it needs a frontend redeploy. In production the backend refuses a wildcard origin and will not start without `FRONTEND_ORIGIN`. If the frontend moves to another domain, update `FRONTEND_ORIGIN` to that exact origin (no trailing slash) and redeploy the backend. For local development use the values in `.env.example` and `frontend/.env.example`.

### Requirements files

| File | Contains | Use it for |
|---|---|---|
| `requirements.txt` | runtime packages only (pandas, numpy, pyyaml, scikit-learn, matplotlib, fastapi, uvicorn) | the deployed API, or running the API locally |
| `requirements-dev.txt` | `-r requirements.txt` plus pytest, httpx, pyarrow | running the test suite and working on the code |

Vercel installs `requirements.txt` only. `uvicorn` stays in it because it is how the API is run locally; the other test-only packages are kept out of the deployment.

### Known limitations of the deployment

- The data is synthetic and the benchmark is a synthetic market index.
- A full backtest takes about 13 to 20 seconds, and the first request after idle time is slower (serverless cold start).
- CORS allows only the exact frontend origin above, so a custom domain needs a `FRONTEND_ORIGIN` update.
- Vercel's team-scoped preview URLs are behind Vercel Authentication; the two production URLs above are public.
- Light theme only.

## Quick start

```bash
pip install -r requirements-dev.txt   # runtime + tests; the API alone needs only requirements.txt

python -m src.sample_data      # creates the synthetic dev data in data/raw/ (already included)
python -m src.data_loader      # loads, validates, prints statistics
python -m src.features         # builds the Phase 2 features, prints a report
python -m src.pca_model        # exploratory PCA: prints results, saves CSV + plots (--components 3|5|10)
python -m src.lda_model        # exploratory LDA on constructed classes: prints results, saves CSV + plot
python -m src.similarity       # exploratory stock similarity from PCA profiles (--symbol X --top-n 5)
python -m src.basket           # exploratory multi-cap basket (--basket-size 6 --capital 50000)
python -m src.backtest         # walk-forward backtest (--capital 50000 --basket-size 6)
uvicorn api.main:app --reload  # the web API: http://localhost:8000/docs
pytest -q                      # runs the tests
```

## Folder layout

```
data/raw/          input price files (CSV or Parquet)
data/processed/    files produced by later phases (git-ignored)
src/               the research engine, one file per pipeline step
api/               the FastAPI web layer over src/ (Phase 8A)
frontend/          the Next.js / React app (Phase 8B)
index.py           Vercel entry point for the API (exposes api.main:app)
tests/             automated tests
notebooks/         exploration
config.yaml        settings (data file paths, provider, date window)
requirements.txt   runtime dependencies (what the deployed API installs)
requirements-dev.txt  runtime plus test dependencies
.python-version    Python version used on Vercel
.vercelignore      files kept out of the backend deployment bundle
.env.example       API settings (allowed origin) and empty Angel One placeholders
```

## Data format

Every data source must produce one table with these columns:

| column | meaning |
|---|---|
| date | trading day |
| symbol | stock ticker, e.g. `TCS` (stored upper-case) |
| open, high, low, close | daily prices, all > 0 |
| volume | shares traded, >= 0 |

To use your own data, put a CSV or Parquet file with these columns in `data/raw/` and change `prices_file` in `config.yaml`. The stock universe is simply whatever symbols the file contains.

### What the loader does

Reads the file, checks the columns, parses dates, sorts by symbol and date, then cleans:

- duplicate (symbol, date) rows: keep the last
- row with no close price: dropped (we never invent a close)
- missing open/high/low: rebuilt from open and close
- missing volume: set to 0
- it never forward-fills or back-fills prices (back-filling would leak future prices into the past)

Then it validates and **raises an error** (it does not silently fix) if it finds impossible candles, such as high below low, non-positive prices or negative volume. The cleaning step prints a report of every change it made.

## About the development data

`data/raw/dev_prices_synthetic.csv` is **synthetic**: random numbers from a small simulation (`src/sample_data.py`), not real prices. The ticker names are only labels. Nothing computed from it says anything about the real companies, and no result from it should be reported as a finding. Real data comes in Phase 8E.

The 10 development tickers are all Large Caps in reality, so the Large/Mid/Small Cap logic needs a wider universe later (Phase 6). The dev dataset also has a synthetic market index (`dev_market_index_synthetic.csv`), which Phase 2 will use for beta and correlation.

## Phase 2: feature engineering (`src/features.py`)

### What is feature engineering?

A model cannot learn from a raw price table directly. Feature engineering turns each stock-day into a short row of numbers that describe how the stock has behaved recently (how much it moved, how bumpy it was, how it relates to the market). Those numbers are what PCA and LDA will work on in later phases.

```python
features = build_features(stock_data, market_data)   # one row per stock-day, 13 feature columns
clean    = clean_feature_data(features)              # drops the rows that lack a full set of features
```

### Why these features?

They are the standard, easy-to-explain ways of describing a stock's behaviour, and between them they cover the aspects a basket should balance: trend, risk, market sensitivity and trading activity. They overlap on purpose (for example 20-day and 60-day volatility are strongly related). That overlap is exactly what PCA is meant to compress in Phase 3.

**Important:** these features **describe past behaviour**. We do not claim they predict future returns.

### What each feature means

All "N-day" windows count trading rows of data, not calendar days. Returns and drawdown are decimals (0.05 = +5%).

| Group | Feature | What it represents | Formula |
|---|---|---|---|
| Returns | `return_5d`, `return_20d`, `return_60d` | How much the price changed over about a week, a month, a quarter | `close today / close N days ago - 1` |
| Volatility | `volatility_20d`, `volatility_60d` | How bumpy the daily moves were (higher = riskier) | standard deviation of the last N daily returns, times `sqrt(252)` to express it per year |
| Price position | `price_ma20_ratio`, `price_ma50_ratio` | Is the price above (>1) or below (<1) its recent average | `close / average of last 20 (or 50) closes` |
| Momentum | `rsi_14` | 0 to 100 gauge of recent rises vs falls. Above about 70 = mostly rising, below about 30 = mostly falling | `100 - 100/(1 + avg gain / avg loss)` over 14 days (plain averages) |
| Drawdown | `max_drawdown` | Worst fall from a peak inside the last 60 days (0 = no fall, -0.2 = fell 20%) | for each day in the window: `price / highest price so far in the window - 1`; take the lowest |
| Market link | `beta_60d` | If the market moves 1%, the stock tends to move beta % | `covariance(stock, market) / variance(market)` over 60 days |
| Market link | `market_correlation_60d` | How closely the stock moves with the market, from -1 to +1 | correlation of daily returns over 60 days |
| Volume | `avg_volume_20d` | Typical number of shares traded recently | average volume over 20 days |
| Volume | `volume_change` | Is today busier (>0) or quieter (<0) than usual | `volume today / avg_volume_20d - 1` |

Choices made where the definition could vary (all easy to change at the top of `src/features.py`):
- Volatility is annualised with 252 trading days per year.
- `max_drawdown` is measured over a rolling 60-day window (`DRAWDOWN_WINDOW`), so it describes recent risk. Measured from the very first day of history, it could never improve and would depend on where the data happens to start.
- `rsi_14` uses plain 14-day averages. The textbook Wilder version smooths differently and gives slightly different numbers.
- The market index is matched to each stock **by date**, and the market's return is taken over the same interval as the stock's return, even if the stock is missing a day.

### Why the first 60 days of each stock are missing

A 60-day feature cannot exist until 60 days of history do. `return_60d` needs the close from 60 days earlier. `volatility_60d`, `beta_60d` and `market_correlation_60d` need 60 daily returns, and 60 returns need 61 closes. So the first valid row of each stock is its 61st, and the first 60 rows contain missing values (`NaN`). Shorter features become valid earlier (for example `rsi_14` on row 15), but a row is only usable once all 13 exist.

We do **not** fill these gaps. Back-filling would copy later information backwards in time, and forward-filling or averaging would invent history. Instead the NaN values are counted and reported, and `clean_feature_data()` removes those rows. A stock with fewer than 61 rows is removed entirely, and the demo warns about it. If volume was missing, Phase 1 stored it as 0, which makes `volume_change` look like a 100% drop on that day. That is a data artefact, not real trading.

### How leakage is prevented

- Every feature on date `t` uses only data from `t` or earlier. All windows look backwards, nothing is shifted to the future, and nothing is filled forward or backward.
- Each stock is processed on its own, so one stock's history never leaks into another's.
- The tests prove it, not just the code review:
  - Changing a **future** stock price, volume or market price leaves every **earlier** feature exactly unchanged.
  - Cutting the data off at date `t` and recomputing gives the same features as the full run at `t`.
  - The tests were checked against deliberately broken versions of the code (a centred window, a moving average pulled from the future); they failed as they should.
- Fitting a scaler or PCA only on training data is a Phase 3 concern. Phase 2 only creates features and fits nothing.

### About the data

The development dataset is **synthetic** (random numbers), as described above. The feature values you see from `python -m src.features` describe simulated history only. They say nothing about the real companies whose names are used as labels, and they must not be reported as findings.

## Phase 3: PCA (`src/pca_model.py`)

```python
model  = fit_pca(training_features, n_components=5)   # LEARN the scaler and the PCA weights
scores = transform_pca(any_features, model)           # APPLY them; nothing is re-learned
get_explained_variance(model)                         # how much each component captures
get_feature_loadings(model)                           # the weights, one row per original feature
```

### What is PCA?

Imagine 13 dashboard gauges where several move together: when an engine works harder, speed, temperature and noise all rise at once. PCA notices which gauges move together and builds a few new gauges (the **principal components**, PC1, PC2, ...), each summarising one pattern. PC1 captures as much of the overall variation as a single new gauge can. PC2 captures as much as possible of what is left, while being uncorrelated with PC1, and so on.

Each principal component is a weighted combination of the original features:

```
PC1 = w1 x feature_1 + w2 x feature_2 + ... + w13 x feature_13
```

PCA is unsupervised: it never sees future returns or any label.

### Why PCA here?

Several of our 13 features overlap (20-day and 60-day volatility, or the 5, 20 and 60-day returns). PCA turns them into a smaller number of **uncorrelated** components while keeping as much of the information as possible. This is the compact stock representation that Phase 4 (LDA) and Phase 5 (similarity) will use.

### The pipeline (the order matters)

```
13 features -> select the 13 columns -> log1p(avg_volume_20d) -> StandardScaler -> PCA -> PC1 ... PCk
```

### Why log-transform volume?

Average trading volume is a raw scale variable and is highly skewed: it is in the millions, and a few very busy stocks sit far above the rest. The logarithm shrinks those extreme values before standardising. We use `log1p(x) = log(1 + x)` because it is safe when volume is 0 (`log(0)` would be minus infinity). This happens **only inside the PCA preprocessing, on a copy**. The `avg_volume_20d` column in the Phase 2 table is never changed.

### Why standardise?

The features are on very different scales: volatility is a fraction around 0.25, RSI runs from 0 to 100, beta is near 1, and volume is in the millions. PCA looks for directions of largest **variance**, so without standardising, whichever feature has the biggest numbers would dominate every component just because of its units. `StandardScaler` rescales each feature to mean 0 and standard deviation 1 so each starts with an equal vote. It runs **before** PCA. (A test shows the difference: without it, RSI alone would swallow almost all the variance.)

### What are loadings?

> A principal component is a weighted combination of the original features. The loading tells us how strongly each feature contributes to that component.

How to read the loadings table (rows = features, columns = components):
- A **larger size** (ignoring the sign) means the feature contributes more to that component.
- Features with the **same sign** push the component the same way; **opposite signs** push in opposite directions.
- Every component's weights have total squared size 1, so loadings lie between -1 and +1.
- The **overall sign of a component is arbitrary**: flipping every sign in a column describes the same component. Read the pattern of signs, not whether PC1 is "positive".
- Do not over-interpret. The demo prints the three biggest loadings per component, taken from the fitted model. Nothing is labelled "momentum" or "risk" in advance.

### How many components?

`explained variance` is the share of the total variation a component captures; the cumulative figure is the running total. Keeping fewer components gives a simpler representation but discards more. Start with 5 (`n_components` in `config.yaml`) and try 3 or 10 with `python -m src.pca_model --components 10`. Comparing 3, 5 and 10 components properly (by how well the resulting baskets do) is a later experiment; Phase 3 only looks at the structure.

### Leakage rule

The scaler (feature means and spreads) and PCA (the weights) are **learned from data**. They must be learned only from the data given to `fit_pca()` and then reused unchanged by `transform_pca()`, which never re-learns anything.

> This demo is an **exploratory** PCA analysis, so it fits on the complete dataset. The later backtesting pipeline must fit preprocessing and PCA only on the training period to avoid look-ahead bias.

Tests demonstrate this. They fit on a training period, change future rows, and show earlier PCA values do not move. They also show, on purpose, that fitting on all the data would let the future change earlier values. The fitted model records the period it was trained on (`fit_start`, `fit_end`).

### What PCA does NOT do

- It does **not** predict stock prices or returns.
- It does **not** guarantee better returns or a better basket. Whether this representation helps is exactly what the later experiments test, and a simple baseline may win.
- It does not decide basket weights. Portfolio construction stays a separate step.

### Things to keep in mind

- Each row is one stock on one day, and neighbouring days share most of their history (a 60-day window overlaps 59 of the next day's 60 days). Rows are therefore not independent. That is fine for describing structure, and worth knowing before drawing statistical conclusions.
- Components depend on the data they were fitted on, so their weights (and signs) can change between training periods. Keep this in mind when we refit per period in the backtest.
- Component sizes depend on the features: if a feature is nearly uncorrelated with the others, it tends to get a component of its own rather than being compressed.

### Outputs

Created by `python -m src.pca_model` (git-ignored, regenerated on demand):

| File | Contents |
|---|---|
| `data/processed/pca_features.csv` | `date, symbol, PC1 ... PCk` for every valid feature row |
| `data/processed/plots/pca_explained_variance.png` | bar chart: variance explained by each component |
| `data/processed/plots/pca_pc1_pc2_scatter.png` | PC1 vs PC2: a random sample of stock-days (for readability only) plus each stock's average position, labelled |

The development data is synthetic, so the percentages, loadings and plots it produces describe simulated data only and must not be reported as findings about real stocks.

## Phase 4: LDA and behavioural classes (`src/labels.py`, `src/lda_model.py`)

### What is LDA?

LDA (Linear Discriminant Analysis) is a **supervised dimensionality-reduction technique that finds directions that separate predefined classes as much as possible.** We hand it the 13 standardised features plus a class for every observation, and it returns two new columns, `LD1` and `LD2`. Each is a weighted mix of the features, chosen so that stocks of different classes land far apart and stocks of the same class land close together.

LDA does **not** discover the classes. We create them first, with a transparent rule.

### PCA vs LDA

| PCA | LDA |
|---|---|
| Unsupervised | Supervised |
| Does not require classes | Requires predefined classes |
| Maximizes variance | Maximizes class separation |
| Finds latent directions | Finds discriminant directions |

They are two different views of the **same** 13 features: both use the same preprocessing (select the 13, `log1p` on volume, standardise).

### Behavioural classes

Every stock-day gets one of three constructed classes describing how the stock has behaved over the recent past:

| Class | Meaning |
|---|---|
| **Defensive** | lower volatility, lower beta, weaker recent return |
| **Balanced** | in between |
| **Aggressive** | higher volatility, higher beta, stronger recent return |

How they are made (`create_behavior_labels`):

1. Rank `volatility_60d`, `beta_60d` and `return_60d` against a reference set (0 = lowest, 1 = highest).
2. Average the three ranks into one score. Higher score = more aggressive. Return counts with a plus sign, as specified for this project.
3. Cut the score at its 1/3 and 2/3 percentiles. Because the cuts are percentiles, the classes come out nearly equal in size instead of depending on arbitrary thresholds.

The labels use only the same date's features. They never use future returns.

### Important limitation: the classes are constructed, and the setup is circular

- The classes are **not ground truth**, not official classifications, not investment advice, and say nothing about future performance.
- They are built from `volatility_60d`, `beta_60d` and `return_60d`, and LDA then uses those same features (plus ten others). So LDA is only showing **whether the selected features can separate the categories we constructed from them**. It is **not** proving that LDA found naturally occurring stock categories. A high training accuracy is expected and proves little.
- Our classes are three slices of one ordered score, so almost all the separation sits on `LD1` and very little on `LD2`. That is a property of how we built the classes, not a market finding.
- LDA assumes each class is roughly bell-shaped with a similar spread. Slices of one distribution only roughly satisfy that, so treat the numbers as descriptive.

### Why LDA?

It lets us examine whether the selected financial features produce separable behavioural groups, and gives a 2-D view (`LD1`, `LD2`) that can be set beside the PCA view.

### Why exactly 2 components?

With K classes, LDA can produce at most **K - 1** discriminant directions. We have 3 classes, so the maximum is 2: `LD1` and `LD2`. (Asking for 3 raises an error; a test checks this.)

### Leakage: fit on the training period, apply to the future

The class rule, the scaler and the LDA directions are all **learned from data**, so all three follow the same fit/apply discipline:

```text
training period
      |
fit scaler
      |
create training labels      fit_label_rules(train) -> create_behavior_labels(train, rules)
      |
fit LDA                     fit_lda(train, labels)
      |
transform future rows       transform_lda(future, model)   (nothing is re-learned)
```

The label rule is a fit/apply pair because its percentile cut points come from data. If they were recomputed on all the data, future observations could move the cut points and change **earlier** labels. `create_behavior_labels(df)` without rules learns them from `df` itself, which is acceptable for this exploratory demo only.

> This demo is **exploratory**, so rules, scaler and LDA are learned from the complete dataset. A backtest must learn them from the training period only.

Tests demonstrate this: with frozen rules or a frozen model, changing future rows does not move earlier labels or earlier `LD1`/`LD2`, and, on purpose, learning on all the data does let the future change earlier values.

### Reading the coefficients

`get_lda_loadings` returns one coefficient per feature for each of `LD1` and `LD2` (sklearn's `scalings_`, the discriminant directions; not the classifier weights `coef_`). Larger absolute coefficients indicate that the feature contributes more strongly to the corresponding discriminant direction, although interpretation should consider the sign and scaling. The features are standardised, so coefficients are comparable, but correlated features can share or trade off a contribution, and the overall sign of a discriminant is arbitrary. Do not over-interpret individual coefficients.

### Training accuracy

The demo prints a training accuracy and a confusion table. The model is evaluated on the same observations it was fitted on, so this is **only a diagnostic**, not an out-of-sample performance measure. Proper temporal evaluation comes later if needed.

### Outputs

Created by `python -m src.lda_model` (git-ignored, regenerated on demand):

| File | Contents |
|---|---|
| `data/processed/lda_features.csv` | `date, symbol, behavior_class, LD1, LD2` for every valid feature row |
| `data/processed/plots/lda_scatter.png` | LD1 vs LD2 coloured by class: a random sample of stock-days (for readability only) plus each class's average position, labelled |

The development data is synthetic and was generated with behavioural differences between stocks. Everything it produces here only verifies that the implementation works; it says nothing about real stocks or real markets. The whole analysis is rerun on real data after Angel One is connected.

## Phase 5: stock behavioural similarity (`src/similarity.py`)

The goal: **given a stock, find the other stocks with the most similar historical behaviour, using the PCA representation.**

This is **not** stock price prediction, return prediction, a buy/sell signal, or a recommendation system, and it is not proof that similar stocks will perform similarly in the future.

### What is stock similarity?

Each stock is represented by its PCA coordinates (PC1 ... PC5), so every stock is one point in PCA space. Two stocks are "similar" when their points are close in direction. That answers one question only: *which stocks have behaved similarly, according to the financial features used in this project?*

### Why PCA space?

The PCA representation summarises the 13 original behaviour features (several of which overlap) into a handful of uncorrelated dimensions. Comparing 5 numbers is simpler and less repetitive than comparing 13 overlapping ones.

### How is one profile created?

The PCA output has many rows per stock (one per day). For a single behavioural profile per stock we average:

> For exploratory similarity analysis, each stock is represented by the mean of its PCA component scores across its available historical observations.

`create_stock_profiles` groups by `symbol` and takes the mean of each PC. The result has exactly one row per stock and no dates. Missing or infinite PC values are rejected rather than skipped, because skipping would average different stocks over different days.

### Why cosine similarity?

```text
cosine_similarity(A, B) = (A . B) / (||A|| x ||B||)

 1.0  same direction     0.0  unrelated     -1.0  opposite directions
```

It is a simple way to compare the **direction** of two profiles. It ignores their length, so a profile and the same profile scaled by 3 score exactly 1.0. We use `sklearn.metrics.pairwise.cosine_similarity`. Two practical consequences:

- PCA scores are centred (they average to zero over all stock-days), so a stock's mean profile is its *deviation from the typical stock-day*. Cosine similarity asks whether two stocks deviate in the same direction, not by how much.
- An all-zero profile has no direction, so cosine similarity is undefined for it. The code raises an error instead of returning a made-up number.

### Functions

| Function | What it does |
|---|---|
| `create_stock_profiles(pca_data)` | one row per stock: `symbol, PC1 ... PCk` (means) |
| `calculate_similarity(stock_profiles)` | N x N matrix, square, symmetric, diagonal 1 |
| `find_similar_stocks(stock_profiles, symbol, top_n=5)` | `rank, symbol, similarity`, best first, never includes the stock itself; ties ordered by symbol; fewer than `top_n` stocks available returns all of them |

An unknown symbol raises an error that lists the available symbols.

### What does similarity mean?

**Historical behavioural similarity in PCA space.** Example: if stock A and stock B have a cosine similarity of 0.95, their historical PCA profiles point in very similar directions. It does **not** mean "Stock B will return 95% of Stock A's return".

### What does similarity NOT mean?

It does not mean:

- higher expected return
- lower risk
- a better investment
- future correlation
- guaranteed diversification

(Whether *dissimilar* stocks make a more diversified basket is a question for the later experiments, not something this phase shows.)

### Leakage limitation

> The current similarity profiles use the full historical period for each stock. This is acceptable for exploratory analysis, but future backtesting must construct stock profiles using only information available up to the relevant rebalance date.

Two things must be point-in-time in a backtest: the **PCA** that produces the scores (fitted on the training period only, Phase 3) and the **averaging** that builds the profile (only rows up to the rebalance date). A test shows the intended use: cut the table at a date first, then build profiles, and later rows cannot influence them. This engine is **not** suitable for live trading as it stands.

### Run it

```bash
python -m src.similarity                        # first available symbol, top 5
python -m src.similarity --symbol SYMBOL --top-n 5
```

It reads the PCA output (`python -m src.pca_model` creates it), builds profiles, saves the matrix, prints the neighbours and saves one bar chart.

| File | Contents |
|---|---|
| `data/processed/similarity_matrix.csv` | the N x N cosine similarity matrix |
| `data/processed/plots/similarity_<symbol>.png` | horizontal bars: the selected stock's most similar stocks |

The development data is synthetic, so the numbers it produces only verify that the implementation works and say nothing about real stocks or real markets.

## Phase 6: intelligent multi-cap basket (`src/universe.py`, `src/basket.py`)

### What is basket construction?

The earlier phases produce analysis: PCA describes latent behaviour, LDA assigns Defensive / Balanced / Aggressive classes, and the similarity engine says which stocks behave alike. Basket construction turns those outputs into one concrete result: **a small set of stocks, each with a weight and a rupee allocation, and a plain-English reason for including it.**

It is deliberately simple and rule-based. The explanation for the viva:

> First, we divide the stocks into Large, Mid and Small Cap categories. Then we use the LDA behavioural classes to understand whether stocks are Defensive, Balanced or Aggressive. We also use PCA-based similarity to avoid selecting too many stocks with almost identical historical behaviour. Finally, we use a simple greedy selection process to create a multi-cap, behaviourally diversified basket and assign equal weights.

### Important limitation: the development cap classes are synthetic

> Synthetic development cap classifications used only for testing and demonstration.

The 10 development symbols are all effectively Large Cap stocks, so a multi-cap basket cannot be shown with their real classifications. `src/universe.py` therefore assigns them to categories **alphabetically, on purpose** (4 Large, 3 Mid, 3 Small), so nothing about the assignment looks like a real classification. For example RELIANCE and TCS appear as "Small Cap" only because they come late in the alphabet. This says nothing about any real company. When the real universe is connected (Phase 8E) the mapping is replaced; the basket code only needs a `symbol, cap_category` table.

### Multi-cap diversification

Large, Mid and Small Cap stocks tend to behave differently, so including all three is a simple way to avoid a basket that is secretly all one size of company. The target split is **40% Large / 30% Mid / 30% Small**, turned into whole numbers by the largest-remainder method (any tie goes to the larger cap):

| Basket size | Large | Mid | Small |
|---|---|---|---|
| 3 | 1 | 1 | 1 |
| 6 | 2 | 2 | 2 |
| 10 | 4 | 3 | 3 |
| 15 | 6 | 5 | 4 |

If a category has fewer eligible stocks than its target, the basket takes what exists and gives each missing slot to the category with the most unused eligible stocks (ties: Large, Mid, Small). This **never crashes and never hides the shortfall**. It is printed, for example: *"Target was 3 Small Cap stocks, but only 2 eligible Small Cap stocks were available. 1 additional stock was selected from Mid Cap."*

### Behavioural diversification

Each stock gets one behaviour class: its **most frequent LDA class across its history** (ties go to the first of Defensive, Balanced, Aggressive). The selection prefers the class that is least represented in the basket so far, so a basket is not all one class when alternatives exist. Equal counts are not required: if the universe has no Aggressive stock, the basket simply has none and the notes say so.

### Similarity control

If three stocks have pairwise similarity above 0.95, holding all three adds little behavioural variety compared with holding one of them plus a less similar stock. So a candidate whose cosine similarity with any already-selected stock is **above the threshold (default 0.90, configurable)** is penalised: it is chosen only if every remaining candidate also exceeds the limit. It is a penalty, not a ban, because the basket still has to reach its size; when that happens the notes name the stock and the pair.

### The greedy algorithm

1. **Eligible** stocks are in the universe and also have a behaviour label and a row in the similarity matrix. Others are left out and reported.
2. Work out the cap targets and, from the eligible supply, the cap quotas.
3. Pick one stock at a time from the cap categories that still have room, ranking candidates by:
   1. not above the similarity limit with any selected stock;
   2. its behaviour class is the least represented so far;
   3. lowest similarity to the selected stocks;
   4. symbol (alphabetical), so the result is always identical.
4. Equal weights, `weight = 1 / basket_size` and `allocation = capital x weight`.
5. Write a reason for each stock **from what actually happened at its pick**: which cap slot it filled, how it changed the behaviour balance, and the real similarity number and partner.

### Equal weighting

Every stock gets `1 / basket_size`. This is deliberate: any risk-based weighting would add choices (how to measure risk, how to scale it) that would blur what the selection rules themselves contribute. Equal weights keep the basket easy to read, and it is the natural baseline against which any later weighting could be compared. The weights sum to 1 and the allocations sum to the capital; both are checked on every basket.

### Why no optimizer?

The first version is intentionally rule-based and transparent, so every choice can be traced and explained. Markowitz, Sharpe optimisation, risk parity, Black-Litterman and genetic algorithms are outside the current scope.

### Input checks

The generator raises a clear error (and never silently repairs) for: `basket_size` below 3, not a whole number, or larger than the number of eligible stocks; non-positive or non-numeric `capital`; a threshold outside -1 to 1; an invalid universe (duplicate symbols, unknown cap category); invalid or missing behaviour labels; and a similarity matrix that is not square, symmetric and finite within [-1, 1]. The finished basket is re-checked: size, no duplicate stock, every stock in the universe, valid class, weights sum to 1, allocations sum to the capital, a reason for every row.

### What changed from the original function signature

`generate_basket(lda_data, similarity_matrix, universe, basket_size, capital, similarity_threshold)` has **no `feature_data` argument**. Everything the basket needs from the features has already been summarised by PCA, LDA and the similarity matrix, and an input that nothing reads would only be misleading. The generator is also given no prices and no returns.

### What this does and does not show

The algorithm constructs a basket designed to provide market-cap and behavioural diversification while reducing excessive similarity between selected stocks. **It has not been tested against any benchmark**, so nothing here claims a better portfolio. Whether it is better is the question for Phase 7. It is not an investment recommendation.

Things to keep in mind:

- With a 10-stock development universe, a 10-stock basket is the **entire universe**, so there is no real selection and any similar pairs cannot be avoided; the notes list them. Smaller baskets (for example 6) show the selection logic.
- The similarity penalty only changes the result in some cases, because the cap quotas, the behaviour balance and the "least similar" ranking already steer away from near-duplicates. It does not always lower the basket's average similarity.
- A stock's single behaviour class is a simplification: it may be Defensive on 60% of its days and Balanced on the rest. The reason text quotes that share.
- The class, the similarity profiles and the PCA behind them currently use the full synthetic history.

### Leakage

No returns or prices are used anywhere in selection, so the generator cannot look at the future. The exploratory demo uses the full synthetic history, which is acceptable only for exploration. A backtest must follow this order:

```text
Historical training data
        |
Features
        |
PCA fitted on training data
        |
LDA fitted using training data / rules
        |
Similarity profiles built using training history
        |
Basket generated
        |
Future period evaluated
```

### Run it

```bash
python -m src.universe                                  # the synthetic development universe
python -m src.basket                                    # settings from config.yaml (10 stocks, ₹100,000)
python -m src.basket --basket-size 6 --capital 50000
python -m src.basket --similarity-threshold 0.95
```

It reads the LDA output (`python -m src.lda_model`) and the similarity matrix (`python -m src.similarity`).

| File | Contents |
|---|---|
| `data/processed/basket.csv` | `symbol, cap_category, behavior_class, weight, allocation, reason` (rows ordered Large, Mid, Small, then by symbol) |
| `data/processed/plots/basket_allocation.png` | horizontal bars of the rupee allocation per stock, coloured by behaviour class |

The development data and the cap classes are synthetic, so the baskets it produces only verify that the implementation works and say nothing about real stocks or real markets.

## Phase 7: walk-forward backtesting (`src/backtest.py`)

### What is a backtest?

A backtest replays the method through the past. At each rebalance date it builds a basket exactly as the live method would have, holds it, records what happened, then repeats. It answers: "how would this method have behaved on this data?" It does **not** say how it will behave in future, and a good-looking curve is not proof that the method works.

### What is walk-forward testing?

The method learns from data (PCA, label rules, LDA, similarity profiles), so it has to be re-learned at every rebalance, using only the past:

```
rebalance date R
   training = every row BEFORE R      ->  features, scaler, PCA, label rules, LDA, profiles, similarity, basket
   test     = R until the next rebalance  ->  hold that basket, record daily returns
then move to the next rebalance and start again (nothing is reused from the earlier fit)
```

### Quarterly rebalancing

Rebalance dates are the **first trading day of each quarter that appears in the dataset** (not a calendar guess, so holidays are handled). A date only counts when at least `min_history_days` (default 252, one trading year) of history exists before it: 60 days are consumed by the feature warm-up and the rest give PCA and LDA enough rows to fit. A quarter whose first day is too early is skipped; we never rebalance on a later day of the quarter. Monthly, semiannual and annual are also available (`--frequency`), for the future backtest form.

### Look-ahead bias, and how it is prevented

Look-ahead bias means using information that did not exist yet at the decision time. It makes any backtest look better than it could have been. At each rebalance:

- the price tables are **cut at the rebalance date before any feature is computed** (`date < R`), so even a rolling-window mistake could not reach the future;
- the scaler, PCA, label rules (percentile cuts), LDA, stock profiles and similarity matrix are all fitted on that training slice only;
- the Phase 6 generator receives only training-period LDA output and a training-period similarity matrix: no prices and no returns, so no future return can influence selection;
- a stock is only eligible if it has a real close on the last training day.

Timing: the basket is decided from data up to the last training day (the close before R) and earns the return of every day from R onward. In effect it is bought at the close before R, which is when its information became available.

Eight look-ahead tests prove this by experiment: two copies of the dataset are identical before a date and wildly different after it, and the baskets, PCA, label rules, LDA, profiles and similarity matrix must come out identical. A chronology test checks that training rows are always strictly before the rebalance date and never overlap the test period. Mutation testing confirmed that fitting any of these on future data, or cutting the slice a day late, makes a test fail.

### Holding the basket

Equal weights. The daily basket return is the mean of the held stocks' daily returns, with no weight changes during the holding period. At each rebalance the rupee allocation uses the portfolio value at that moment, so the money compounds. Transaction costs and slippage are **zero** (no cost model).

### Missing prices (conservative, deterministic)

Nothing is forward-filled, back-filled or invented. If a held stock has no usable return on some day (its close or the previous close is missing), it is **stopped for the rest of that holding period** and the remaining stocks stay equally weighted. If every stock is stopped the return is 0 (cash) and the days are counted. Stopped stocks are listed in the result (`missing_data_events`). A rebalance that cannot be built (too little history, too few tradable stocks) is skipped and recorded: the previous basket is kept, or, if there is none yet, those days are left out of the backtest. The market index must have every price it needs, otherwise the backtest stops with an error.

### The benchmark

The benchmark is the **market index in the supplied data**, with the same dates and the same starting capital. For the development data, **the synthetic market index is the benchmark**. No other data source is used.

### The five metrics (strategy and benchmark, side by side)

| Metric | Definition |
|---|---|
| Cumulative return | final value / initial capital - 1 |
| Annualised return | (1 + cumulative) ^ (1 / years) - 1, years = real calendar days / 365.25 |
| Annualised volatility | standard deviation of daily returns x sqrt(252) |
| Sharpe ratio | annualised return / annualised volatility, **risk-free rate = 0** (undefined, shown as n/a, if volatility is 0) |
| Maximum drawdown | smallest value of (value / running peak - 1); the starting capital counts as the first peak |

### Run it

```bash
python -m src.backtest                                       # settings from config.yaml (10 stocks, ₹100,000, quarterly)
python -m src.backtest --capital 100000 --basket-size 10
python -m src.backtest --capital 50000 --basket-size 6
python -m src.backtest --start-date 2021-01-01 --end-date 2023-12-31 --frequency quarterly
```

| File | Contents |
|---|---|
| `data/processed/backtest_results.csv` | `date, portfolio_value, daily_return, benchmark_value, benchmark_return` |
| `data/processed/rebalance_history.csv` | `rebalance_date, symbol, cap_category, behavior_class, weight, allocation, reason` |
| `data/processed/backtest_summary.json` | initial capital, final value and the metrics above, for strategy and benchmark |
| `data/processed/plots/backtest_equity_curve.png` | portfolio value vs benchmark value |
| `data/processed/plots/backtest_drawdown.png` | portfolio drawdown (benchmark as a thin line) |

### Reading the development-data results

The development data is synthetic, so the run only validates that the pipeline works ("Synthetic pipeline validation only"). Two things to know when you read it:

- With 10 stocks in the universe, a basket of 10 is the **whole universe**, so selection changes nothing: that run is simply an equal-weight portfolio of all ten stocks.
- With 6 of 10, the baskets barely change from quarter to quarter on this data, and the strategy trails the synthetic index. That is a description of random data, not a finding about the method.

The method claims diversification, not higher returns. Nothing here shows that features, PCA, LDA, similarity or the baskets predict returns or make a better portfolio. Judging that needs real data and the baselines (random, equal-weight, market-cap) from the project plan.

### Limitations

- Synthetic data, synthetic cap classes and a synthetic benchmark.
- Zero transaction costs and zero slippage; no cost model.
- Zero risk-free rate in the Sharpe ratio.
- Simplified basket construction: one behaviour class per stock, a fixed cap split, a greedy selection, no optimiser.
- The daily return is the mean of the stocks' daily returns (equal weights each day), which is a simplification of a buy-and-hold basket.
- Possible survivorship bias on real data: the universe and cap classes are fixed, not historical index membership.
- No live execution, no orders, no guarantee of any outcome. Not investment advice.

### Built to be exposed through an API later

`src/backtest.py` returns plain Python data and never prints (only the command line does): `run_backtest(...)` gives `summary`, `equity_curve`, `drawdown`, `rebalance_history`, `rebalances`, `skipped_rebalances`, `missing_data_events`, `data_info` and `settings`. `result_to_jsonable(result)` converts it to JSON. The module imports no Angel One code and holds no credentials (a test checks this), and it reads the standard Phase 1 price tables.

## Architecture note

```
Current pipeline (this repository, Phases 1-7)
Data -> Features -> PCA -> LDA -> Similarity -> Basket -> Backtest

Future production architecture (NOT implemented)
Angel One SmartAPI -> FastAPI backend -> ML / basket / backtest engine -> JSON API -> Next.js / React frontend -> Vercel
```

The production frontend and backend were **not** implemented in Phase 7. As of Phase 7 there was no FastAPI app, no Next.js project, no Vercel deployment, no Angel One connection, no authentication and no database. Since then Phases 8A to 8C added the FastAPI backend, the Next.js frontend and the Vercel deployment; the Angel One connection, authentication and a database are still not built. Python stays the engine: the ML and financial pipeline runs in the backend, Angel One credentials live in the backend only, and the frontend never holds API keys, secrets, ML credentials, private keys or database credentials. It only calls the JSON API.

The code is shaped so that this can be added without a rewrite. The functions return structured data, so the planned screens map onto them:

| Future screen | Backed by |
|---|---|
| Dashboard | `run_backtest` summary and charts data |
| PCA / LDA analysis | `fit_pca`, `fit_lda` outputs and loadings |
| Stock similarity selector | `find_similar_stocks` |
| Basket generator form (size, capital) | `generate_basket` |
| Backtest form (start date, end date, basket size, capital, rebalance frequency) | `run_backtest(start_date, end_date, basket_size, capital, frequency)` |

The earlier "Streamlit dashboard" checklist item is superseded by this direction.

## Phase 8A: Backend API (`api/`)

Phase 8 turns the research project into a web application, step by step:

| Step | What | Status |
|---|---|---|
| **8A** | **Backend: FastAPI + API contracts** | **done (this section)** |
| 8B | Frontend: Next.js / React | done (see Phase 8B below) |
| 8C | Production deployment and integration (frontend and backend on Vercel) | done (see Production deployment above) |
| 8D | Production hardening / next deployment phase | not started |
| 8E | Angel One / live market data integration (backend only) | not started |

### How it fits together

```
Existing research engine (src/: data, features, PCA, LDA, similarity, basket, backtest)
        |
FastAPI API layer (api/: validate request -> call a service function -> return JSON)
        |
JSON responses (no DataFrames, NumPy values, CSV files or NaN ever reach the client)
        |
Next.js frontend (Phase 8B), which only draws what the API returns
```

The API is a thin layer. The PCA, LDA, similarity, basket and backtest logic stays in `src/` and is not copied: routes (`api/routes/`) only validate and call `api/services.py`, which calls the existing modules. A test checks that routes import no pandas, NumPy, scikit-learn or `src` code, and that only the service layer touches the engine. `src/` was not changed in this phase.

```
api/
  main.py            app factory, CORS, error handlers      settings.py   environment variables
  schemas.py         request / response contracts (Pydantic) errors.py     one error shape
  services.py        all real work, delegated to src/        serialization.py   JSON-safe output
  routes/            health, data, analysis, similarity, basket, backtest
```

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/dataset` | Dataset information (`is_synthetic`, dates, stocks, trading days) |
| GET | `/api/universe` | Stock universe with cap categories |
| GET | `/api/analysis/pca` | PCA analysis (`?components=5&max_points=2000`) |
| GET | `/api/analysis/lda` | LDA analysis (`?max_points=2000`) |
| GET | `/api/similarity/{symbol}` | Similar stocks (`?top_n=5`). Returns `selected` plus `cap_category` and `behavior_class` for each stock (added in 8B so the UI never recomputes them) |
| POST | `/api/basket/generate` | Generate a basket |
| POST | `/api/backtest` | Run a walk-forward backtest |

Interactive documentation with every schema is served at `/docs` (Swagger) and `/redoc`; the raw contract is `/openapi.json`. This is what Phase 8B is built against.

Requests:

```jsonc
POST /api/basket/generate   { "capital": 100000, "basket_size": 10, "similarity_threshold": 0.90 }
POST /api/backtest          { "capital": 100000, "basket_size": 10, "frequency": "quarterly",
                              "similarity_threshold": 0.90, "start_date": "2019-01-01", "end_date": "2025-12-31" }
                            // frequency: monthly | quarterly | semiannual | annual. Everything except the dates is optional.
```

The backtest response contains `summary`, `benchmark`, `equity_curve`, `drawdown`, `rebalance_history`, plus a rebalance overview, skipped rebalances, missing-data events, `data_info`, `settings`, `is_synthetic` and `notices`. The PCA and LDA endpoints return a reproducible sample of the stock-day scores (`max_points`, with `total_observations` giving the full count), so the payload stays small; the frontend draws the charts from these numbers.

### Two modes, stated in every response

- `exploratory_full_history`: PCA, LDA, similarity and the basket endpoints fit on the whole dataset, exactly like the command-line demos. They show structure; they are **not** a performance test.
- `walk_forward`: the backtest refits at every rebalance using only earlier data (Phase 7).

The LDA response labels its accuracy as a *training* accuracy on constructed classes, so it is not evidence of predictive power. The dataset is synthetic: `is_synthetic` is `true` in `/api/dataset`, the basket and backtest responses, and the notices say so.

### Errors

Every error has the same shape and never contains a traceback or internal detail:

```json
{ "error": { "status": 404, "code": "unknown_symbol", "message": "Unknown stock symbol 'NOPE'. Available symbols: ..." } }
```

| Status | When | Example `code` |
|---|---|---|
| 400 | A valid request that cannot be served: a basket larger than the available stocks, a window with no possible rebalance | `invalid_basket_size`, `invalid_backtest_request` |
| 404 | Unknown stock symbol, unknown route | `unknown_symbol`, `not_found` |
| 405 | Wrong HTTP method | `method_not_allowed` |
| 422 | Invalid parameters: capital <= 0, basket_size < 3 or not a whole number, threshold outside -1..1, bad frequency or date, unknown fields (with the offending field names in `details`) | `invalid_request` |
| 500 | Anything unexpected (the traceback is only in the server log) | `internal_error` |

### CORS and environment variables

The browser origin allowed to call the API comes from the environment, never from code:

| Variable | Meaning |
|---|---|
| `FRONTEND_ORIGIN` | Allowed origin(s), comma separated. Default `http://localhost:3000`. In production: the deployed frontend origin |
| `ENVIRONMENT` | `development` (default) or `production`. Production refuses `*` and requires `FRONTEND_ORIGIN` to be set |
| `CONFIG_PATH` | Optional path to a different `config.yaml` |

Only the listed origins get CORS headers; credentials are off. `.env.example` lists these plus the Angel One placeholders (empty, and **not read anywhere yet**). `.env` is git-ignored. Angel One credentials will exist only on the backend and never in the browser.

### Run it

```bash
pip install -r requirements-dev.txt   # runtime + tests; the API alone needs only requirements.txt
uvicorn api.main:app --reload          # http://localhost:8000/api/health  and  /docs
pytest -q tests/api                    # the API tests
```

### Performance

The loaded data and the fitted exploratory models are kept in memory after the first request (`functools.lru_cache`), because they cannot change while the server runs. The first PCA/LDA/similarity/basket call therefore takes about a second; later ones are instant. A full backtest is not cached and takes about 12 to 15 seconds, and returns about 600 KB. The heavy work is isolated in `api/services.py`, so caching or background jobs can be added later without touching the routes. Do not add them before they are needed.

### Tests

`tests/api/` covers every endpoint, error and CORS case. Most tests run the real research modules on the synthetic data: the API output is compared with the engine called directly (basket, similarity, PCA, LDA, backtest), so the API cannot be returning fixtures. Other tests check that responses are strict JSON (no NaN or Infinity), that nothing is hard-coded (a different config changes `/api/dataset`), and that the API holds no Angel One code or credentials.

### Not built yet (as of 8A)

No Next.js or React app (added in Phase 8B), no Vercel configuration, no deployment, no Angel One connection, no authentication, no database. The final architecture is unchanged:

```
Internet -> Next.js / React (Vercel) -> HTTPS -> FastAPI -> Data layer (Angel One) / PCA, LDA, similarity, basket / Backtest
```

## Phase 8B: Frontend (`frontend/`)

A Next.js (App Router), React, TypeScript and Tailwind CSS app with Recharts. It is a presentation layer only: it calls the FastAPI endpoints and draws what comes back. It never imports Python, reads `data/`, or computes PCA, LDA, similarity, baskets or backtests. `src/` was not changed; `api/` got one small additive change (see below).

```
Browser -> Next.js / React (frontend/) -> HTTP -> FastAPI (api/) -> research engine (src/)
```

### Run it (two terminals)

```bash
uvicorn api.main:app --reload          # http://localhost:8000
cd frontend
cp .env.example .env.local             # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev                            # http://localhost:3000
```

Other commands: `npm run build`, `npm run lint`, `npm run typecheck`, `npm test`.

### Environment

`NEXT_PUBLIC_API_URL` is the only setting. It is read in `frontend/lib/api.ts` and never hard-coded, and the app does not assume the API is on the same domain. Anything prefixed `NEXT_PUBLIC_` is visible in the browser, so the frontend holds no secrets. Angel One credentials will only ever live on the backend.

### Pages

| Route | What it shows |
|---|---|
| `/` | Landing page. The dataset label comes from the API, and a small scree chart is read live from `/api/analysis/pca` |
| `/dashboard` | Dataset and universe cards, PCA and LDA summaries, similarity selector, quick basket, quick backtest (runs only on click) |
| `/pca` | Explained variance, PC1 against PC2 sample (colour by none, cap category or one highlighted stock), feature loadings |
| `/lda` | Class cards, LD1 against LD2 coloured by behaviour class, training accuracy with the in-sample warning |
| `/similarity` | Stock selector, most similar stocks table, bar chart |
| `/basket` | Capital, size and threshold form, result table with reasons, allocation donut, cap and behaviour distributions, similarity statistics, backend notes |
| `/backtest` | Date range, capital, size, frequency, threshold; six metrics against the benchmark, equity curve, drawdown, expandable rebalance history, skipped rebalances and missing-price events |

Every API-driven block has its own loading, error (with Retry) and empty state, so one failing endpoint does not break a page.

### Structure

```
frontend/
  app/          one folder per route, plus layout.tsx and globals.css (design tokens)
  components/   layout, dashboard, charts, analysis, basket, backtest, common
  lib/          api.ts (typed client), types.ts (mirrors api/schemas.py), validation.ts, hooks.ts, utils.ts
  tests/        Vitest tests; fixtures/ holds real responses captured from the backend
```

### Honesty rules built into the UI

- Data is labelled "Synthetic Dataset" whenever the API says `is_synthetic`. Nothing claims to be live Angel One data.
- LDA accuracy is called "Training accuracy" and carries the in-sample warning. The majority-class baseline is shown next to it.
- PCA, LDA, similarity and basket pages are marked exploratory (fitted on the full history). Only the backtest is walk-forward.
- Similarity is described as historical behavioural similarity, not expected return.
- A basket as large as the whole universe triggers a notice, because no selection takes place.
- API notes, notices, skipped rebalances and missing-price events are shown, never hidden.
- The dashboard's quick backtest and quick basket use stated defaults (50,000 capital, 6 stocks, quarterly, threshold 0.90).

### API change made in 8B

`GET /api/similarity/{symbol}` now also returns `selected` and, for each similar stock, `cap_category` and `behavior_class`. The table needs them, and the alternative was to recompute them in the browser. The change is additive in `api/services.py` and `api/schemas.py`, reuses existing engine functions, and has two new tests.

### Tests

`npm test` runs 47 Vitest tests: the API client (URL from the environment, caching, error shapes, no credentials), dashboard rendering and the click-only backtest, basket validation and result rendering, backtest loading and result rendering, and API error handling (backend down, 422, 400, 404, 500). Fixtures are real backend responses. Tests were checked by breaking the code on purpose and confirming a test fails.

### Limitations

- Light theme only.
- The PCA and LDA charts show the sample of points the API returns (default 2,000 of 20,280), not every observation.
- The PCA scatter cannot colour by all 10 stocks at once (too many hues to tell apart), so it offers cap category or one highlighted stock.
- The backtest takes 10 to 20 seconds, and the page waits for the response (with a cancel button). There is no background job.
- Built and tested against `http://localhost:3000` and `http://localhost:8000`. The deployed version was verified in Phase 8C (see Production deployment above).
- No authentication, database or Angel One connection.

## Switching to Angel One later

All code gets data through one interface:

```python
provider = get_provider(config)                       # reads config.yaml
df = provider.get_historical_data(symbols, start, end)
```

`LocalDataProvider` works now. In Phase 8E we add `AngelOneDataProvider`, which returns the same table, then change `provider: local` to `provider: angelone` in `config.yaml`. No ML code changes. Credentials go in a git-ignored `.env` file (see `.env.example`); they are never needed before Phase 8E.

## Honest limitations (to be kept in the final report)

- **Survivorship bias.** The planned universe is today's Large, Mid and Small Cap stocks applied to the past. Stocks that were delisted or merged are missing, which inflates absolute returns, especially for small caps. It affects every strategy equally, so comparisons stay fair, but absolute numbers must not be presented as achievable.
- **Corporate actions.** Whether Angel One's candles are adjusted for splits and bonuses is unverified. We test this before trusting the real data.
- **Educational research project.** Not investment advice. Past behaviour does not imply future returns, and "similar stocks" means similar historical features only.
