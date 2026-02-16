from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


class DataBootstrapper:
    """Best-effort dataset bootstrapper for required and optional files.

    Creates (if missing):
    - prices.csv
    - universe_membership.csv
    - macro.csv
    - sentiment.csv (proxy sentiment/risk regime + cross-sectional signals)
    - insider.csv (proxy insider-like accumulation/distribution signals)

    Sources:
    - Wikipedia for S&P 500 constituents
    - Yahoo Finance via yfinance for market/equity/macro proxies
    """

    def __init__(self, data_dir: Path, download_start_date: str = "2000-01-01"):
        self.data_dir = data_dir
        self.download_start_date = download_start_date
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_sp500_tickers() -> List[str]:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if not tables:
            raise RuntimeError("Unable to fetch S&P 500 constituents from Wikipedia.")
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return sorted(set(tickers))

    @staticmethod
    def _normalize_download(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=1).reset_index().rename(columns={"level_1": "symbol"})
        else:
            df = df.reset_index()
            if "symbol" not in df.columns:
                df["symbol"] = "UNKNOWN"

        col_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)
        needed = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"Downloaded data missing column: {col}")
        return df[needed]

    def _yf_download_adj_close(self, ticker_map: Dict[str, str]) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            tickers=list(ticker_map.keys()),
            start=self.download_start_date,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date"] + list(ticker_map.values()))

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Adj Close"].copy()
        else:
            close = raw[["Adj Close"]].rename(columns={"Adj Close": list(ticker_map.keys())[0]})

        out = close.rename(columns=ticker_map).reset_index().rename(columns={"Date": "date"})
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out.sort_values("date")

    def _download_prices(self) -> pd.DataFrame:
        import yfinance as yf

        tickers = self._get_sp500_tickers()
        chunks: List[pd.DataFrame] = []
        chunk_size = 100

        for i in range(0, len(tickers), chunk_size):
            batch = tickers[i : i + chunk_size]
            raw = yf.download(
                tickers=batch,
                start=self.download_start_date,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
            if raw is None or raw.empty:
                continue
            normalized = self._normalize_download(raw)
            normalized = normalized[normalized["symbol"].isin(batch)]
            chunks.append(normalized)

        if not chunks:
            raise RuntimeError("Failed to download price history for S&P 500 universe.")

        prices = pd.concat(chunks, ignore_index=True)
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        prices = prices.dropna(subset=["open", "high", "low", "close", "adj_close", "volume"])
        prices = prices.sort_values(["symbol", "date"]).drop_duplicates(["date", "symbol"], keep="last")
        return prices

    @staticmethod
    def _build_universe_membership(prices: pd.DataFrame) -> pd.DataFrame:
        universe = prices[["date", "symbol"]].copy()
        universe["is_active"] = 1
        return universe.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")

    def _download_macro(self) -> pd.DataFrame:
        # Broader macro/risk/sector variables to maximize coverage.
        ticker_map = {
            # Volatility / rates / FX / commodities
            "^VIX": "vix",
            "^VVIX": "vvix",
            "^IRX": "us3m_yield",
            "^FVX": "us5y_yield",
            "^TNX": "us10y_yield",
            "^TYX": "us30y_yield",
            "DX-Y.NYB": "dxy",
            "CL=F": "wti_crude",
            "BZ=F": "brent_crude",
            "NG=F": "natgas",
            "GC=F": "gold",
            "SI=F": "silver",
            "HG=F": "copper",
            # Equity indices / ETFs
            "^GSPC": "sp500",
            "^NDX": "nasdaq100",
            "^RUT": "russell2000",
            "^DJI": "dow",
            "SPY": "spy",
            "QQQ": "qqq",
            "IWM": "iwm",
            # Credit / duration / sector rotation
            "HYG": "hyg",
            "LQD": "lqd",
            "TLT": "tlt",
            "IEF": "ief",
            "XLE": "xle",
            "XLF": "xlf",
            "XLK": "xlk",
            "XLI": "xli",
            "XLP": "xlp",
            "XLV": "xlv",
            "XLU": "xlu",
            "XLY": "xly",
            "XLC": "xlc",
        }

        macro = self._yf_download_adj_close(ticker_map)
        if macro.empty:
            return pd.DataFrame(columns=["date"])

        macro = macro.sort_values("date")

        base_cols = [c for c in macro.columns if c != "date"]
        for c in base_cols:
            macro[f"{c}_ret_1d"] = macro[c].pct_change(1)
            macro[f"{c}_ret_5d"] = macro[c].pct_change(5)
            macro[f"{c}_ret_21d"] = macro[c].pct_change(21)
            macro[f"{c}_vol_21d"] = macro[c].pct_change().rolling(21, min_periods=10).std()

        # Derived spreads / style / risk-on metrics.
        if {"us10y_yield", "us3m_yield"}.issubset(macro.columns):
            macro["yc_10y_3m"] = macro["us10y_yield"] - macro["us3m_yield"]
        if {"hyg", "lqd"}.issubset(macro.columns):
            macro["credit_risk_hyg_lqd"] = macro["hyg"] / (macro["lqd"] + 1e-12)
        if {"qqq", "spy"}.issubset(macro.columns):
            macro["growth_vs_broad_qqq_spy"] = macro["qqq"] / (macro["spy"] + 1e-12)
        if {"iwm", "spy"}.issubset(macro.columns):
            macro["small_vs_large_iwm_spy"] = macro["iwm"] / (macro["spy"] + 1e-12)
        if {"xly", "xlp"}.issubset(macro.columns):
            macro["risk_on_consumer_xly_xlp"] = macro["xly"] / (macro["xlp"] + 1e-12)

        return macro

    @staticmethod
    def _build_sentiment_proxy(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        df = prices[["date", "symbol", "adj_close", "volume"]].copy().sort_values(["symbol", "date"])
        grp = df.groupby("symbol", group_keys=False)

        df["ret_1d"] = grp["adj_close"].pct_change(1)
        df["ret_5d"] = grp["adj_close"].pct_change(5)
        df["ret_21d"] = grp["adj_close"].pct_change(21)

        vol_mean = grp["volume"].rolling(21, min_periods=10).mean().reset_index(level=0, drop=True)
        vol_std = grp["volume"].rolling(21, min_periods=10).std().reset_index(level=0, drop=True)
        df["volume_z_21d"] = (df["volume"] - vol_mean) / (vol_std + 1e-12)

        # Price trend proxies for sentiment / crowding.
        ma_10 = grp["adj_close"].rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
        ma_50 = grp["adj_close"].rolling(50, min_periods=25).mean().reset_index(level=0, drop=True)
        df["trend_score_10_50"] = (ma_10 / (ma_50 + 1e-12)) - 1.0

        # Up-volume / down-volume pressure proxy.
        signed_volume = np.sign(df["ret_1d"].fillna(0.0)) * df["volume"]
        sv_roll = signed_volume.groupby(df["symbol"]).rolling(21, min_periods=10).sum().reset_index(level=0, drop=True)
        v_roll = df.groupby("symbol")["volume"].rolling(21, min_periods=10).sum().reset_index(level=0, drop=True)
        df["buy_pressure_21d"] = sv_roll / (v_roll + 1e-12)

        keep = [
            "date",
            "symbol",
            "ret_1d",
            "ret_5d",
            "ret_21d",
            "volume_z_21d",
            "trend_score_10_50",
            "buy_pressure_21d",
        ]
        sentiment = df[keep].copy()

        # Merge market regime sentiment/risk variables (date-level) into per-symbol rows.
        market_sent_cols = [
            c
            for c in macro.columns
            if c in {"date", "vix", "vvix", "spy_ret_1d", "spy_ret_5d", "hyg_ret_5d", "xly_ret_5d", "xlp_ret_5d"}
        ]
        if market_sent_cols:
            sentiment = sentiment.merge(macro[market_sent_cols], on="date", how="left")

        sentiment = sentiment.sort_values(["symbol", "date"]).drop_duplicates(["date", "symbol"], keep="last")
        return sentiment

    @staticmethod
    def _build_insider_proxy(prices: pd.DataFrame) -> pd.DataFrame:
        # Free-data insider *proxy* signals (time-safe; no future snapshots).
        df = prices[["date", "symbol", "adj_close", "volume"]].copy().sort_values(["symbol", "date"])
        grp = df.groupby("symbol", group_keys=False)

        ret_1d = grp["adj_close"].pct_change(1)
        ret_21d = grp["adj_close"].pct_change(21)

        vol_mean = grp["volume"].rolling(63, min_periods=20).mean().reset_index(level=0, drop=True)
        vol_std = grp["volume"].rolling(63, min_periods=20).std().reset_index(level=0, drop=True)
        volume_z_63d = (df["volume"] - vol_mean) / (vol_std + 1e-12)

        # Accumulation patterns that can correlate with informed flow.
        insider_buy_proxy = (ret_1d > 0).astype(float) * np.maximum(volume_z_63d, 0.0)
        insider_sell_proxy = (ret_1d < 0).astype(float) * np.maximum(volume_z_63d, 0.0)

        insider_buy_21d = insider_buy_proxy.groupby(df["symbol"]).rolling(21, min_periods=10).mean().reset_index(level=0, drop=True)
        insider_sell_21d = insider_sell_proxy.groupby(df["symbol"]).rolling(21, min_periods=10).mean().reset_index(level=0, drop=True)

        out = pd.DataFrame(
            {
                "date": df["date"],
                "symbol": df["symbol"],
                "insider_buy_proxy_21d": insider_buy_21d,
                "insider_sell_proxy_21d": insider_sell_21d,
                "insider_imbalance_proxy_21d": insider_buy_21d - insider_sell_21d,
                "price_momentum_21d": ret_21d,
                "volume_z_63d": volume_z_63d,
            }
        )
        return out.sort_values(["symbol", "date"]).drop_duplicates(["date", "symbol"], keep="last")

    def ensure_data(self, refresh_downloaded_features: bool = False) -> None:
        prices_path = self.data_dir / "prices.csv"
        universe_path = self.data_dir / "universe_membership.csv"
        macro_path = self.data_dir / "macro.csv"
        sentiment_path = self.data_dir / "sentiment.csv"
        insider_path = self.data_dir / "insider.csv"

        prices: pd.DataFrame
        if not prices_path.exists() or not universe_path.exists():
            prices = self._download_prices()
            prices.to_csv(prices_path, index=False)
            self._build_universe_membership(prices).to_csv(universe_path, index=False)
        else:
            prices = pd.read_csv(prices_path)
            prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

        if refresh_downloaded_features or not macro_path.exists():
            macro = self._download_macro()
            macro.to_csv(macro_path, index=False)
        else:
            macro = pd.read_csv(macro_path)
            macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()

        if refresh_downloaded_features or not sentiment_path.exists():
            self._build_sentiment_proxy(prices, macro).to_csv(sentiment_path, index=False)

        if refresh_downloaded_features or not insider_path.exists():
            self._build_insider_proxy(prices).to_csv(insider_path, index=False)


class MarketDataLoader:
    """Loads required and optional market data tables from CSV files."""

    def __init__(self, data_dir: Path, auto_download_data: bool = True, download_start_date: str = "2000-01-01"):
        self.data_dir = data_dir
        self.auto_download_data = auto_download_data
        self.download_start_date = download_start_date

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

    def load(self, refresh_downloaded_features: bool = False) -> Dict[str, pd.DataFrame]:
        if self.auto_download_data:
            DataBootstrapper(self.data_dir, self.download_start_date).ensure_data(
                refresh_downloaded_features=refresh_downloaded_features
            )

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
