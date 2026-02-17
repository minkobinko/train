from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "data"
    cache_dir: str = ".cache"
    output_dir: str = "outputs"

    sequence_length: int = 90
    sequence_stride: int = 1
    min_return_threshold: float = 0.05
    min_adjusted_prob: float = 0.58

    train_end_date: str = "2022-12-31"
    test_start_date: str = "2023-01-01"
    walkforward_months: int = 3

    max_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 4
    ff_mult: int = 4

    gradient_accumulation_steps: int = 2
    mixed_precision: bool = True
    seed: int = 42

    purged_kfold_splits: int = 4
    purge_days: int = 90
    embargo_days: int = 21

    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.18
    max_positions: int = 25
    per_trade_risk_fraction: float = 0.01
    entropy_penalty: float = 0.08

    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0

    min_train_samples: int = 4096
    num_workers: int = 0


HORIZONS = {
    "1w": 5,
    "1m": 21,
    "2m": 42,
    "3m": 63,
    "4m": 84,
    "5m": 105,
    "6m": 126,
}
