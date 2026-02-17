from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd


class MarketDataLoader:
    """Loads required and optional market data tables from CSV files."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    @staticmethod
    def _read_csv(path: Path, required_cols: Sequence[str]) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_csv(path)
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load(self) -> Dict[str, pd.DataFrame]:
        prices = self._read_csv(
            self.data_dir / "prices.csv",
            ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"],
        )
        universe = self._read_csv(
            self.data_dir / "universe_membership.csv",
            ["date", "symbol", "is_active"],
        )

        tables = {
            "prices": prices,
            "universe": universe,
            "macro": pd.DataFrame(),
            "sentiment": pd.DataFrame(),
            "insider": pd.DataFrame(),
        }

        optional_specs = {
            "macro": ["date"],
            "sentiment": ["date", "symbol"],
            "insider": ["date", "symbol"],
        }

        for name, required_cols in optional_specs.items():
            path = self.data_dir / f"{name}.csv"
            if path.exists():
                tables[name] = self._read_csv(path, required_cols)

        return tables
