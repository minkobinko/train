from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import Config, HORIZONS
from .windows import WindowedData


@dataclass
class Trade:
    date: np.datetime64
    symbol: str
    horizon: str
    entry: float
    exit: float
    gross_return: float
    net_return: float


class Backtester:
    """Simple portfolio simulator with risk, costs, and trade filters."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _adjusted_probability(self, p: np.ndarray) -> np.ndarray:
        entropy = -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
        return np.clip(p - self.cfg.entropy_penalty * entropy, 0.0, 1.0)

    def _net_return(self, gross_return: float) -> float:
        total_cost = (self.cfg.transaction_cost_bps + self.cfg.slippage_bps) / 10_000.0
        return gross_return - total_cost

    def run(self, probs: np.ndarray, window_data: WindowedData, test_idx: np.ndarray) -> Dict[str, float]:
        dates = window_data.dates[test_idx]
        symbols = window_data.symbols[test_idx]
        prices = window_data.last_prices[test_idx]
        probs = probs[test_idx]
        horizon_names = list(HORIZONS.keys())

        order = np.argsort(dates)
        dates, symbols, prices, probs = dates[order], symbols[order], prices[order], probs[order]

        capital = 1_000_000.0
        peak = capital
        max_drawdown = 0.0
        trades: List[Trade] = []

        for d in np.unique(dates):
            mask = dates == d
            p_today, s_today, px_today = probs[mask], symbols[mask], prices[mask]

            candidates = []
            for i in range(len(p_today)):
                for h_idx, h_name in enumerate(horizon_names):
                    p_raw = float(p_today[i, h_idx])
                    p_adj = float(self._adjusted_probability(np.array([p_raw]))[0])
                    expected_edge = p_adj * self.cfg.take_profit_pct - (1 - p_adj) * self.cfg.stop_loss_pct
                    if p_adj >= self.cfg.min_adjusted_prob and expected_edge > 0:
                        candidates.append((expected_edge, i, h_name))

            candidates.sort(reverse=True, key=lambda x: x[0])
            picks = candidates[: self.cfg.max_positions]

            for edge, i, h_name in picks:
                entry = float(px_today[i])
                if entry <= 0:
                    continue

                risk_dollars = capital * self.cfg.per_trade_risk_fraction
                shares = min(risk_dollars / max(self.cfg.stop_loss_pct * entry, 1e-6), capital / entry)

                gross = max(-self.cfg.stop_loss_pct, min(self.cfg.take_profit_pct, edge))
                net = self._net_return(gross)
                pnl = shares * entry * net
                capital += pnl

                trades.append(
                    Trade(
                        date=d,
                        symbol=str(s_today[i]),
                        horizon=h_name,
                        entry=entry,
                        exit=entry * (1 + net),
                        gross_return=gross,
                        net_return=net,
                    )
                )

            peak = max(peak, capital)
            drawdown = (capital - peak) / peak
            max_drawdown = min(max_drawdown, drawdown)

        if not trades:
            return {
                "final_capital": capital,
                "total_return": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
            }

        net_rets = np.array([t.net_return for t in trades], dtype=np.float32)
        return {
            "final_capital": float(capital),
            "total_return": float(capital / 1_000_000.0 - 1.0),
            "n_trades": int(len(trades)),
            "win_rate": float((net_rets > 0).mean()),
            "max_drawdown": float(max_drawdown),
        }
