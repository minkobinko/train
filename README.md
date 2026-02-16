# Multi-Asset Transformer Stock Return Threshold Predictor

A modular, leakage-aware research program for training a **single model over an asset universe** (S&P 500 to start) to predict whether future returns exceed a minimum threshold at multiple horizons.

## Project structure

- `stock_transformer.py` — thin entrypoint.
- `stock_predictor/config.py` — configuration + horizon definitions.
- `stock_predictor/data.py` — data loading for prices/universe/macro/sentiment/insider.
- `stock_predictor/features.py` — point-in-time feature engineering + labels.
- `stock_predictor/windows.py` — rolling window generation + cache.
- `stock_predictor/splits.py` — purged k-fold and walk-forward split logic.
- `stock_predictor/model.py` — multi-asset transformer.
- `stock_predictor/training.py` — scaling, dataloaders, training loop, checkpoints.
- `stock_predictor/backtest.py` — portfolio simulation with trade filters and risk controls.
- `stock_predictor/pipeline.py` — orchestration.
- `stock_predictor/cli.py` — command-line interface.

## Highlights

- Multi-asset training (S&P 500 and broader universe support).
- Multi-source features: price/volume, macro, sentiment, insider.
- Multi-horizon targets: `1w, 1m, 2m, 3m, 4m, 5m, 6m`.
- 90-day windows with 1-day stride.
- Walk-forward evaluation from `2023-01-01` to present by rolling time blocks.
- Purged + embargoed fold construction for overlap-aware CV.
- Point-in-time universe membership filtering for survivorship-bias management.
- Cached window datasets for speed.
- Crash-resumable checkpoints per walk-forward split.
- Trade simulation with min adjusted probability, expected-edge filtering, stop-loss/take-profit, and costs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Required:

- `data/prices.csv` columns: `date,symbol,open,high,low,close,adj_close,volume`
- `data/universe_membership.csv` columns: `date,symbol,is_active`

Optional:

- `data/macro.csv` (`date,feature...`)
- `data/sentiment.csv` (`date,symbol,feature...`)
- `data/insider.csv` (`date,symbol,feature...`)

All joins are point-in-time on `date` and (where relevant) `symbol`.

## Run (directly from a `.py` file, no compile step)

```bash
python run_stock_predictor.py \
  --data-dir data \
  --cache-dir .cache \
  --output-dir outputs \
  --train-end-date 2022-12-31 \
  --test-start-date 2023-01-01 \
  --sequence-length 90 \
  --min-return-threshold 0.05 \
  --min-adjusted-prob 0.58
```

Alternative entrypoint (same behavior):

```bash
python stock_transformer.py ...
```

## Leakage controls implemented

- Backward-looking feature transforms only.
- Labels computed from forward returns and excluded from feature set.
- Train-only scaler fit and application to val/test.
- Validation/test maintained in chronological order.
- Purge + embargo logic for overlap-aware fold creation.

## Notes

- This is a research framework, not investment advice.
- Extend backtest realism further with borrow fees, partial fills, and corporate action handling if needed.
