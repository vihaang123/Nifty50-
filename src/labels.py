"""
Phase 4a: behavioural classes (Defensive / Balanced / Aggressive).

What these labels are
---------------------
Three CONSTRUCTED classes that describe how a stock has behaved over the recent past:

    Defensive   lower volatility, lower beta, weaker recent return
    Balanced    in between
    Aggressive  higher volatility, higher beta, stronger recent return

They are NOT ground truth, NOT official classifications, NOT investment recommendations, and
they say nothing about FUTURE performance. They exist so that LDA (a supervised method) has
predefined classes to try to separate.

The rule (simple and transparent)
---------------------------------
For each of the three label features  volatility_60d, beta_60d, return_60d:

  1. RANK it against a reference set of observations:
         rank = fraction of reference values that are <= this value     (0 = lowest, 1 = highest)
     Ranks need no scaling and are not thrown off by outliers.
  2. SCORE = the plain average of the three ranks.        Higher score = more aggressive.
  3. Cut the score at its 1/3 and 2/3 points (percentiles) of the reference set:
         score <= lower cut          -> Defensive
         lower cut < score <= upper  -> Balanced
         score > upper cut           -> Aggressive
     Because the cuts are percentiles, the classes come out roughly equal in size, instead of
     depending on arbitrary fixed thresholds that could leave one class nearly empty.

Return enters with a PLUS sign (stronger recent return = more aggressive), as specified for this
project. Note this mixes a risk idea (volatility, beta) with a trend idea (return).

LIMITATION: CIRCULAR BY CONSTRUCTION (read this)
------------------------------------------------
The labels are built from volatility_60d, beta_60d and return_60d, and LDA later uses those SAME
features (plus ten others). So LDA is only demonstrating whether the chosen features can separate
the categories WE constructed from them. It is NOT discovering naturally occurring stock categories,
and a high LDA accuracy is expected and proves little.

LEAKAGE: the reference set is "learned" data
--------------------------------------------
The ranks and the 1/3 and 2/3 cut points come from a reference set, so they are learned from data,
exactly like a scaler or PCA. That makes labelling a fit/apply pair:

    rules  = fit_label_rules(training_features)           learn the reference ranks and cuts
    labels = create_behavior_labels(any_features, rules)  apply them; nothing is re-learned

With frozen rules, a row's label depends only on that row and the training data, so changing future
observations can never change an earlier label. If you call create_behavior_labels(df) WITHOUT rules,
the rules are learned from df itself. That is fine for the exploratory demo on the full dataset, but a
backtest must learn the rules from the TRAINING period only. (Learning them on all the data would let
future observations move the cut points and so change earlier labels.)

Labels use only the features of the same date. They never use future returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The three features the rule is built from (all already in the Phase 2 table).
LABEL_FEATURES = ["volatility_60d", "beta_60d", "return_60d"]
# Ordered from least to most aggressive. This order is used everywhere (tables, plots).
CLASS_NAMES = ["Defensive", "Balanced", "Aggressive"]
LABEL_COLUMN = "behavior_class"
MIN_CLASS_SHARE = 0.05  # a class with less than 5% of the observations is considered unusable


@dataclass
class LabelRules:
    """What fit_label_rules() learned. Treat it as read-only after fitting."""

    reference: dict[str, np.ndarray]  # each label feature's training values, sorted (to compute ranks)
    score_cuts: tuple[float, float]  # the 1/3 and 2/3 percentiles of the training score
    n_training_rows: int
    fit_start: pd.Timestamp | None  # first/last date of the training data (None if no date column)
    fit_end: pd.Timestamp | None


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _label_inputs(feature_data: pd.DataFrame) -> pd.DataFrame:
    """The three label features as numbers, checked. Returns a new table; the input is not modified."""
    missing = [c for c in LABEL_FEATURES if c not in feature_data.columns]
    if missing:
        raise ValueError(f"feature_data is missing column(s): {missing}")
    if len(feature_data) == 0:
        raise ValueError("feature_data is empty.")
    values = feature_data[LABEL_FEATURES].astype("float64")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("The label features contain missing or infinite values. Run clean_feature_data() first.")
    return values


# ---------------------------------------------------------------------------
# Learn the rules, then apply them
# ---------------------------------------------------------------------------
def _ranks(values: pd.DataFrame, reference: dict[str, np.ndarray]) -> np.ndarray:
    """For each label feature: the fraction of reference values <= this value. Shape (rows, 3), each in [0, 1]."""
    columns = []
    for feature in LABEL_FEATURES:
        sorted_reference = reference[feature]
        columns.append(np.searchsorted(sorted_reference, values[feature].to_numpy(), side="right") / len(sorted_reference))
    return np.column_stack(columns)


def fit_label_rules(feature_data: pd.DataFrame) -> LabelRules:
    """
    LEARN the reference ranks and the 1/3 and 2/3 score cut points from `feature_data` (and only from it).

    LEAKAGE: when backtesting, pass training-period data only.
    """
    values = _label_inputs(feature_data)
    reference = {feature: np.sort(values[feature].to_numpy()) for feature in LABEL_FEATURES}
    training_score = _ranks(values, reference).mean(axis=1)
    lower, upper = np.quantile(training_score, [1 / 3, 2 / 3])
    if not lower < upper:
        raise ValueError("The behaviour score has too little variety to split into three classes.")

    dates = pd.to_datetime(feature_data["date"]) if "date" in feature_data.columns else None
    return LabelRules(
        reference=reference,
        score_cuts=(float(lower), float(upper)),
        n_training_rows=len(values),
        fit_start=dates.min() if dates is not None else None,
        fit_end=dates.max() if dates is not None else None,
    )


def behavior_score(feature_data: pd.DataFrame, rules: LabelRules) -> pd.Series:
    """The behaviour score of each row (0 to 1, higher = more aggressive) under the given rules."""
    score = _ranks(_label_inputs(feature_data), rules.reference).mean(axis=1)
    return pd.Series(score, index=feature_data.index, name="behavior_score")


def create_behavior_labels(feature_data: pd.DataFrame, rules: LabelRules | None = None) -> pd.Series:
    """
    Label every row Defensive, Balanced or Aggressive.

    rules: frozen rules from fit_label_rules() (use this when backtesting, with rules learned on the
           training period). If None, the rules are learned from `feature_data` itself, which is
           acceptable for exploratory analysis only (see the module notes on leakage).

    Returns a Series (same index as feature_data, name "behavior_class") of an ORDERED categorical:
    Defensive < Balanced < Aggressive. The input table is not modified.
    """
    if rules is None:
        rules = fit_label_rules(feature_data)
    score = behavior_score(feature_data, rules).to_numpy()
    lower, upper = rules.score_cuts
    names = np.where(score <= lower, CLASS_NAMES[0], np.where(score <= upper, CLASS_NAMES[1], CLASS_NAMES[2]))
    return pd.Series(pd.Categorical(names, categories=CLASS_NAMES, ordered=True), index=feature_data.index, name=LABEL_COLUMN)


# ---------------------------------------------------------------------------
# Class sizes
# ---------------------------------------------------------------------------
def label_counts(labels: pd.Series) -> pd.Series:
    """Number of observations in each class, in the order Defensive, Balanced, Aggressive (0 if absent)."""
    return labels.astype(str).value_counts().reindex(CLASS_NAMES, fill_value=0).astype(int)


def check_label_balance(labels: pd.Series, min_share: float = MIN_CLASS_SHARE) -> pd.Series:
    """
    Raise ValueError if any class holds less than `min_share` of the observations (default 5%).
    Returns the class counts when all classes are reasonably populated.
    """
    counts = label_counts(labels)
    shares = counts / counts.sum()
    too_small = [name for name in CLASS_NAMES if shares[name] < min_share]
    if too_small:
        summary = ", ".join(f"{name}: {counts[name]:,} ({shares[name]:.1%})" for name in CLASS_NAMES)
        raise ValueError(f"Unusable class balance (each class needs at least {min_share:.0%}): {summary}")
    return counts


def describe_label_rules(rules: LabelRules) -> str:
    """A plain-English statement of the rule that was learned (printed by the demo)."""
    lower, upper = rules.score_cuts
    period = f" from {rules.fit_start.date()} to {rules.fit_end.date()}" if rules.fit_start is not None else ""
    return "\n".join(
        [
            f"Class rule (learned from {rules.n_training_rows:,} observations{period}):",
            f"  rank {', '.join(LABEL_FEATURES)} (0 = lowest, 1 = highest), average the three ranks into a score",
            f"  score <= {lower:.3f}            -> {CLASS_NAMES[0]}",
            f"  {lower:.3f} < score <= {upper:.3f} -> {CLASS_NAMES[1]}",
            f"  score > {upper:.3f}             -> {CLASS_NAMES[2]}",
        ]
    )
