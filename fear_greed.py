"""
Fear & Greed Index — Data Loader + Feature Builder
====================================================
Source: https://api.alternative.me/fng/
Free API, updated daily, data back to 2018.

Features generated:
  fng_value      : Raw index 0-100
  fng_norm       : Normalized 0.0-1.0
  fng_7d_change  : Change in index over 7 days (momentum)
  fng_30d_zscore : Z-score vs 30-day mean (how extreme is the sentiment)
  fng_extreme    : 1 if < 20 (Extreme Fear) or > 80 (Extreme Greed), else 0
  fng_fear_zone  : 1 if value < 40, else 0
  fng_greed_zone : 1 if value > 60, else 0

Evidence:
  Extreme Fear  (< 25) → historically BTC bounces (contrarian buy signal)
  Extreme Greed (> 75) → historically BTC corrects (contrarian sell signal)
  The 7d CHANGE matters more than the absolute value for short-term prediction.
"""

import json
import os
import time
import datetime
import urllib.request
import pandas as pd
import numpy as np

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".fng_cache.json")
CACHE_TTL  = 3600 * 12   # re-fetch every 12 hours


class FearGreedLoader:
    def __init__(self):
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def get_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by date with F&G features.
        Dates not found in the API data are forward-filled.
        """
        raw = self._load_data()
        if raw is None or raw.empty:
            return self._empty_features(dates)

        # Align to requested dates
        raw = raw.reindex(dates.normalize())
        raw = raw.ffill().bfill()    # fill weekends / missing days

        return raw

    def get_latest(self) -> dict:
        """Returns today's F&G value as a dict."""
        raw = self._load_data()
        if raw is None or raw.empty:
            return {"value": 50, "classification": "Neutral", "date": "N/A"}
        row = raw.iloc[-1]
        return {
            "value":          int(row["fng_value"]),
            "classification": self._classify(int(row["fng_value"])),
            "date":           str(raw.index[-1].date())
        }

    # ------------------------------------------------------------------ #
    #  Loading & Caching
    # ------------------------------------------------------------------ #
    def _load_data(self) -> pd.DataFrame | None:
        if self._df is not None:
            return self._df

        # Try cache first
        cached = self._read_cache()
        if cached is not None:
            self._df = cached
            return self._df

        # Fetch from API
        self._df = self._fetch_from_api()
        if self._df is not None:
            self._write_cache(self._df)
        return self._df

    def _fetch_from_api(self, limit: int = 3000) -> pd.DataFrame | None:
        """Fetch up to `limit` days of F&G history from the API."""
        try:
            url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
            with urllib.request.urlopen(url, timeout=15) as r:
                payload = json.loads(r.read())

            rows = []
            for entry in payload["data"]:
                ts  = int(entry["timestamp"])
                val = int(entry["value"])
                dt  = datetime.datetime.utcfromtimestamp(ts).date()
                rows.append({"date": str(dt), "fng_value": val})

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")

            return self._build_features(df)

        except Exception as e:
            print(f"  ⚠️  F&G fetch failed: {e}")
            return None

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        v = df["fng_value"].astype(float)

        df["fng_norm"]       = v / 100.0
        df["fng_7d_change"]  = v.diff(7) / 100.0
        df["fng_14d_change"] = v.diff(14) / 100.0
        df["fng_30d_zscore"] = (v - v.rolling(30).mean()) / (v.rolling(30).std() + 1e-9)
        df["fng_7d_ma"]      = v.rolling(7).mean() / 100.0
        df["fng_extreme"]    = ((v < 20) | (v > 80)).astype(float)
        df["fng_fear_zone"]  = (v < 40).astype(float)
        df["fng_greed_zone"] = (v > 60).astype(float)

        return df

    # ------------------------------------------------------------------ #
    #  Cache helpers
    # ------------------------------------------------------------------ #
    def _read_cache(self) -> pd.DataFrame | None:
        try:
            if not os.path.exists(CACHE_FILE):
                return None
            mtime = os.path.getmtime(CACHE_FILE)
            if time.time() - mtime > CACHE_TTL:
                return None           # stale
            with open(CACHE_FILE) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            return df
        except Exception:
            return None

    def _write_cache(self, df: pd.DataFrame):
        try:
            records = df.reset_index().to_dict(orient="records")
            for r in records:
                r["date"] = str(r["date"])[:10]
            with open(CACHE_FILE, "w") as f:
                json.dump(records, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _classify(self, value: int) -> str:
        if value <= 24:   return "Extreme Fear"
        if value <= 44:   return "Fear"
        if value <= 55:   return "Neutral"
        if value <= 74:   return "Greed"
        return "Extreme Greed"

    def _empty_features(self, dates) -> pd.DataFrame:
        cols = ["fng_value", "fng_norm", "fng_7d_change", "fng_14d_change",
                "fng_30d_zscore", "fng_7d_ma", "fng_extreme",
                "fng_fear_zone", "fng_greed_zone"]
        df = pd.DataFrame(0.5, index=dates.normalize(), columns=cols)
        df["fng_value"] = 50
        return df


# Singleton for reuse across the predictor
_loader = FearGreedLoader()

def get_fng_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return _loader.get_features(dates)

def get_fng_latest() -> dict:
    return _loader.get_latest()
