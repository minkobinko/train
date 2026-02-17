from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .backtest import Backtester
from .config import Config, HORIZONS
from .data import MarketDataLoader
from .features import FeatureBuilder, get_feature_columns
from .model import MultiAssetTransformer
from .splits import PurgedKFold, WalkForwardSplitter
from .training import Trainer, infer_probabilities, set_seed, split_and_scale
from .windows import WindowCacheBuilder


def save_metrics(path: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_pipeline(cfg: Config) -> Dict[str, Dict[str, float]]:
    set_seed(cfg.seed)

    tables = MarketDataLoader(
        Path(cfg.data_dir),
        auto_download_data=cfg.auto_download_data,
        download_start_date=cfg.download_start_date,
    ).load(refresh_downloaded_features=cfg.refresh_downloaded_features)
    features_df = FeatureBuilder(cfg).build(tables)
    feature_cols = get_feature_columns(features_df)

    windows = WindowCacheBuilder(cfg).build_or_load(features_df, feature_cols)
    unique_symbols = sorted(np.unique(windows.symbols).tolist())
    symbol_to_idx = {s: i for i, s in enumerate(unique_symbols)}

    splitter = WalkForwardSplitter(cfg.train_end_date, cfg.test_start_date, cfg.walkforward_months)
    splits = splitter.split(windows.dates)

    trainer = Trainer(cfg)
    backtester = Backtester(cfg)
    all_metrics: Dict[str, Dict[str, float]] = {}

    for i, split in enumerate(splits):
        if len(split.train_idx) < cfg.min_train_samples:
            print(f"Skipping split {i}: insufficient train samples ({len(split.train_idx)}).")
            continue

        # Build purged folds from training dates to enforce overlap-aware CV partitions.
        _ = list(PurgedKFold(cfg.purged_kfold_splits, cfg.purge_days, cfg.embargo_days).split(windows.dates[split.train_idx]))

        train_loader, val_loader, test_loader = split_and_scale(
            windows,
            split,
            symbol_to_idx=symbol_to_idx,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        model = MultiAssetTransformer(
            n_features=windows.X.shape[-1],
            n_symbols=len(unique_symbols),
            n_horizons=len(HORIZONS),
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
            max_seq_len=cfg.sequence_length,
        )

        ckpt = Path(cfg.output_dir) / "checkpoints" / f"split_{i}.ckpt"
        model = trainer.train(model, train_loader, val_loader, ckpt)

        probs_test = infer_probabilities(model, test_loader)
        probs_full = np.zeros((len(windows.X), len(HORIZONS)), dtype=np.float32)
        probs_full[split.test_idx] = probs_test

        metrics = backtester.run(probs_full, windows, split.test_idx)
        all_metrics[f"split_{i}"] = metrics
        print(f"split={i} metrics={metrics}")

    save_metrics(Path(cfg.output_dir) / "metrics.json", all_metrics)
    return all_metrics
