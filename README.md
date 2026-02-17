# Multi-Asset Transformer Stock Return Threshold Predictor

A modular, leakage-aware research program for training a **single model over an asset universe** (S&P 500 to start) to predict whether future returns exceed a minimum threshold at multiple horizons.

## Project structure

- `run_stock_predictor.py` — direct launcher (`python run_stock_predictor.py ...`).
- `stock_transformer.py` — alternative thin entrypoint.
- `stock_predictor/config.py` — configuration + horizon definitions.
- `stock_predictor/data.py` — data loading + automatic data bootstrap.
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

## Automatic data downloads (expanded)

If files are missing, the program now bootstraps **all major inputs automatically**:

1. `prices.csv` (S&P 500 OHLCV from Yahoo Finance)
2. `universe_membership.csv` (point-in-time rows from available symbol/date coverage)
3. `macro.csv` with many variables, including:
   - volatility: VIX, VVIX
   - rates: 3m/5y/10y/30y yields
   - FX/commodities: DXY, WTI, Brent, NatGas, Gold, Silver, Copper
   - indices/ETFs: SP500, NDX, RUT, DJI, SPY, QQQ, IWM
   - credit & duration: HYG, LQD, TLT, IEF
   - sector ETFs: XLE, XLF, XLK, XLI, XLP, XLV, XLU, XLY, XLC
   - derived features: returns, rolling vol, yield-curve spread, risk-on/risk-off ratios
4. `sentiment.csv` per symbol (proxy sentiment variables):
   - return/trend/volume z-scores
   - buy-pressure proxy
   - merged market-regime sentiment/risk columns
5. `insider.csv` per symbol (proxy insider-like flow variables):
   - buy/sell pressure imbalance proxies from abnormal signed volume
   - momentum/volume context features

This gives you a broad feature base with no manual CSV prep required.
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
  --download-start-date 2000-01-01 \
  --refresh-downloaded-features \
  --train-end-date 2022-12-31 \
  --test-start-date 2023-01-01 \
  --sequence-length 90 \
  --min-return-threshold 0.05 \
  --min-adjusted-prob 0.58
```

To disable auto-download and require local CSVs:

```bash
python run_stock_predictor.py --no-auto-download-data
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

## Survivorship-bias note

The auto-built `universe_membership.csv` is best-effort from historical price availability. For strict institutional survivorship-bias control, replace with dedicated point-in-time index membership history.

## Notes

- This is a research framework, not investment advice.
- Extend backtest realism further with borrow fees, partial fills, and corporate action handling if needed.


Tip: use `--refresh-downloaded-features` to force regeneration of `macro.csv`, `sentiment.csv`, and `insider.csv` even if they already exist.


## Troubleshooting

- If auto-download fails, verify internet/proxy access and that dependencies are installed:

```bash
pip install -r requirements.txt
```

- You can always bypass downloading and use your own local CSVs:

```bash
python run_stock_predictor.py --no-auto-download-data
```
