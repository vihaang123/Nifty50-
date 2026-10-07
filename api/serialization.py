"""Everything leaving the API must be plain JSON: no DataFrames, NumPy numbers, Timestamps, NaN or Infinity."""

from __future__ import annotations

from src.backtest import result_to_jsonable as to_jsonable  # DataFrame -> list of dicts, numpy -> python, Timestamp -> YYYY-MM-DD, NaN/inf -> null

__all__ = ["to_jsonable"]
