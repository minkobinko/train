from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int, purge_days: int, embargo_days: int):
        self.n_splits = n_splits
        self.purge_days = np.timedelta64(purge_days, "D")
        self.embargo_days = np.timedelta64(embargo_days, "D")

    def split(self, dates: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        order = np.argsort(dates)
        sorted_dates = dates[order]
        n = len(sorted_dates)

        fold_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            fold_sizes[i] += 1

        start = 0
        for fold_size in fold_sizes:
            stop = start + fold_size
            val_sorted_idx = np.arange(start, stop)

            val_start = sorted_dates[val_sorted_idx].min()
            val_end = sorted_dates[val_sorted_idx].max()
            left_limit = val_start - self.purge_days
            right_limit = val_end + self.embargo_days

            train_mask = (sorted_dates < left_limit) | (sorted_dates > right_limit)
            train_sorted_idx = np.where(train_mask)[0]

            yield order[train_sorted_idx], order[val_sorted_idx]
            start = stop


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class WalkForwardSplitter:
    def __init__(self, train_end_date: str, test_start_date: str, walkforward_months: int):
        self.train_end = np.datetime64(pd.to_datetime(train_end_date))
        self.test_start = np.datetime64(pd.to_datetime(test_start_date))
        self.walkforward_months = walkforward_months

    def split(self, dates: np.ndarray) -> List[Split]:
        unique_dates = np.array(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()), dtype="datetime64[ns]")
        test_dates = unique_dates[unique_dates >= self.test_start]
        if len(test_dates) == 0:
            raise ValueError("No test dates >= test_start_date.")

        splits: List[Split] = []
        cursor_start = test_dates.min()

        while cursor_start <= test_dates.max():
            cursor_end = np.datetime64(
                pd.Timestamp(cursor_start) + pd.DateOffset(months=self.walkforward_months) - pd.Timedelta(days=1)
            )

            train_mask = dates <= self.train_end
            val_mask = (dates > self.train_end) & (dates < cursor_start)
            test_mask = (dates >= cursor_start) & (dates <= cursor_end)

            if test_mask.sum() == 0:
                break

            splits.append(
                Split(
                    train_idx=np.where(train_mask)[0],
                    val_idx=np.where(val_mask)[0],
                    test_idx=np.where(test_mask)[0],
                )
            )

            self.train_end = cursor_end
            cursor_start = np.datetime64(pd.Timestamp(cursor_end) + pd.Timedelta(days=1))

        return splits
