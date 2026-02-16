from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import Config, HORIZONS


@dataclass
class WindowedData:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    symbols: np.ndarray
    last_prices: np.ndarray


class WindowCacheBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> str:
        payload = {
            "n": len(df),
            "min_date": str(df["date"].min()),
            "max_date": str(df["date"].max()),
            "features": list(feature_cols),
            "sequence_length": self.cfg.sequence_length,
            "sequence_stride": self.cfg.sequence_stride,
            "threshold": self.cfg.min_return_threshold,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def build_or_load(self, df: pd.DataFrame, feature_cols: List[str]) -> WindowedData:
        cache_file = self.cache_dir / f"windows_{self._cache_key(df, feature_cols)}.npz"
        if cache_file.exists():
            arr = np.load(cache_file, allow_pickle=True)
            return WindowedData(
                X=arr["X"],
                y=arr["y"],
                dates=arr["dates"],
                symbols=arr["symbols"],
                last_prices=arr["last_prices"],
            )

        X_list, y_list, d_list, s_list, p_list = [], [], [], [], []
        y_cols = [f"label_{h}" for h in HORIZONS]

        for symbol, sdf in tqdm(df.groupby("symbol"), desc="Building windows"):
            sdf = sdf.sort_values("date").reset_index(drop=True)
            feat = sdf[feature_cols].to_numpy(dtype=np.float32)
            y = sdf[y_cols].to_numpy(dtype=np.float32)
            dates = sdf["date"].to_numpy(dtype="datetime64[ns]")
            prices = sdf["adj_close"].to_numpy(dtype=np.float32)

            for end_idx in range(self.cfg.sequence_length - 1, len(sdf), self.cfg.sequence_stride):
                start_idx = end_idx - self.cfg.sequence_length + 1
                X_list.append(feat[start_idx : end_idx + 1])
                y_list.append(y[end_idx])
                d_list.append(dates[end_idx])
                s_list.append(symbol)
                p_list.append(prices[end_idx])

        if not X_list:
            raise ValueError("No windows were generated. Check data coverage and sequence settings.")

        out = WindowedData(
            X=np.stack(X_list),
            y=np.stack(y_list),
            dates=np.array(d_list, dtype="datetime64[ns]"),
            symbols=np.array(s_list),
            last_prices=np.array(p_list, dtype=np.float32),
        )
        np.savez_compressed(cache_file, X=out.X, y=out.y, dates=out.dates, symbols=out.symbols, last_prices=out.last_prices)
        return out


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, symbols: np.ndarray, symbol_to_idx: Dict[str, int]):
        self.X = X
        self.y = y
        self.symbol_idx = np.array([symbol_to_idx[s] for s in symbols], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx]),
            torch.tensor(self.symbol_idx[idx], dtype=torch.long),
        )
