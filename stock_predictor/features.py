from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .config import Config, HORIZONS


NON_FEATURE_COLUMNS: Set[str] = {
    "date",
    "symbol",
    "is_active",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
}


class FeatureBuilder:
    """Constructs leakage-aware point-in-time features and labels."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _engineer_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.sort_values(["symbol", "date"]).copy()
        grp = prices.groupby("symbol", group_keys=False)

        prices["ret_1d"] = grp["adj_close"].pct_change(1)
        prices["ret_5d"] = grp["adj_close"].pct_change(5)
        prices["ret_21d"] = grp["adj_close"].pct_change(21)

        prices["vol_21d"] = grp["ret_1d"].rolling(21, min_periods=10).std().reset_index(level=0, drop=True)
        prices["vol_63d"] = grp["ret_1d"].rolling(63, min_periods=20).std().reset_index(level=0, drop=True)

        prices["ma_10"] = grp["adj_close"].rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
        prices["ma_50"] = grp["adj_close"].rolling(50, min_periods=25).mean().reset_index(level=0, drop=True)
        prices["ma_ratio_10_50"] = prices["ma_10"] / (prices["ma_50"] + 1e-12)

        prices["dollar_vol"] = prices["adj_close"] * prices["volume"]
        prices["dollar_vol_21d"] = grp["dollar_vol"].rolling(21, min_periods=10).mean().reset_index(level=0, drop=True)
        prices["high_low_range"] = (prices["high"] - prices["low"]) / prices["open"].replace(0, np.nan)

        for name, days in HORIZONS.items():
            fwd = grp["adj_close"].shift(-days) / grp["adj_close"] - 1.0
            prices[f"fwd_ret_{name}"] = fwd
            prices[f"label_{name}"] = (fwd >= self.cfg.min_return_threshold).astype(float)

        return prices

    def build(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = self._engineer_price_features(tables["prices"])
        universe = tables["universe"].copy()

        # Point-in-time active universe to reduce survivorship bias.
        df = prices.merge(universe[["date", "symbol", "is_active"]], on=["date", "symbol"], how="inner")
        df = df[df["is_active"] == 1].copy()

        if not tables["macro"].empty:
            df = df.merge(tables["macro"].sort_values("date"), on="date", how="left")

        for optional_name in ["sentiment", "insider"]:
            table = tables[optional_name]
            if not table.empty:
                df = df.merge(table, on=["date", "symbol"], how="left")

        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        feature_cols = [
            c
            for c in df.columns
            if c not in NON_FEATURE_COLUMNS and not c.startswith("fwd_ret_") and not c.startswith("label_")
        ]

        # Fill gaps only with historical data for each symbol.
        df[feature_cols] = df.groupby("symbol")[feature_cols].ffill()

        label_cols = [f"label_{h}" for h in HORIZONS.keys()]
        return df.dropna(subset=label_cols)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    label_cols = {f"label_{h}" for h in HORIZONS.keys()}
    return [
        c for c in df.columns if c not in NON_FEATURE_COLUMNS and c not in label_cols and not c.startswith("fwd_ret_")
    ]
