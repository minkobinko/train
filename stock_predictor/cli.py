from __future__ import annotations

import argparse

from .config import Config


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Multi-asset transformer stock return threshold model")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--auto-download-data", action="store_true", dest="auto_download_data")
    p.add_argument("--no-auto-download-data", action="store_false", dest="auto_download_data")
    p.set_defaults(auto_download_data=True)
    p.add_argument("--download-start-date", default="2000-01-01")
    p.add_argument("--refresh-downloaded-features", action="store_true")

    p.add_argument("--sequence-length", type=int, default=90)
    p.add_argument("--sequence-stride", type=int, default=1)
    p.add_argument("--min-return-threshold", type=float, default=0.05)
    p.add_argument("--min-adjusted-prob", type=float, default=0.58)

    p.add_argument("--train-end-date", default="2022-12-31")
    p.add_argument("--test-start-date", default="2023-01-01")
    p.add_argument("--walkforward-months", type=int, default=3)

    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)

    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--purged-kfold-splits", type=int, default=4)
    p.add_argument("--purge-days", type=int, default=90)
    p.add_argument("--embargo-days", type=int, default=21)

    p.add_argument("--stop-loss-pct", type=float, default=0.08)
    p.add_argument("--take-profit-pct", type=float, default=0.18)
    p.add_argument("--max-positions", type=int, default=25)
    p.add_argument("--per-trade-risk-fraction", type=float, default=0.01)
    p.add_argument("--entropy-penalty", type=float, default=0.08)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)

    p.add_argument("--min-train-samples", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=0)

    return Config(**vars(p.parse_args()))


def main() -> None:
    cfg = parse_args()
    # Lazy import so `--help` works even if heavy deps are not installed yet.
    from .pipeline import run_pipeline

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
