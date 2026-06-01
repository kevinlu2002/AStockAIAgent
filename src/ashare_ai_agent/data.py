from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from .config import AppConfig


COLUMN_MAP = {
    "日期": "date",
    "股票代码": "symbol",
    "代码": "symbol",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}


def parse_symbols(symbols: str | Iterable[str] | None, fallback: list[str]) -> list[str]:
    if symbols is None:
        return fallback
    if isinstance(symbols, str):
        parts = [p.strip() for p in symbols.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in symbols if str(p).strip()]
    return [p.zfill(6) for p in parts]


def normalize_history(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns}).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "symbol" not in df.columns:
        if symbol is None:
            raise ValueError("History data needs a symbol column or a symbol argument.")
        df["symbol"] = str(symbol).zfill(6)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    keep = ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover"]
    for col in ["amount", "turnover"]:
        if col not in df.columns:
            df[col] = 0.0

    out = df[keep].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(out["symbol"].astype(str)).str.zfill(6)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turnover"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return out.reset_index(drop=True)


def read_local_history(raw_dir: Path, symbol: str) -> pd.DataFrame | None:
    candidates = [
        raw_dir / f"{symbol}.csv",
        raw_dir / f"{symbol}_daily.csv",
        raw_dir / f"sh{symbol}.csv",
        raw_dir / f"sz{symbol}.csv",
    ]
    candidates.extend(sorted(raw_dir.glob(f"*{symbol}*.csv")))
    for path in candidates:
        if path.exists():
            return normalize_history(pd.read_csv(path), symbol=symbol)
    return None


def _effective_end_date(cfg: AppConfig) -> str:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    configured = str(cfg.data.end_date)
    return max(configured, today)


def fetch_history_with_akshare(symbol: str, cfg: AppConfig) -> pd.DataFrame:
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AkShare is not installed. Install it with `python -m pip install akshare`, "
            "or place CSV files in data/raw."
        ) from exc

    errors: list[str] = []
    try:
        end_date = _effective_end_date(cfg)
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=cfg.data.start_date,
            end_date=end_date,
            adjust=cfg.data.adjust,
        )
        return normalize_history(df, symbol=symbol)
    except Exception as exc:
        errors.append(f"eastmoney={exc!r}")

    market_symbol = f"sh{symbol}" if str(symbol).startswith("6") else f"sz{symbol}"
    try:
        df = ak.stock_zh_a_daily(
            symbol=market_symbol,
            start_date=cfg.data.start_date,
            end_date=end_date,
            adjust=cfg.data.adjust,
        )
        return normalize_history(df, symbol=symbol)
    except Exception as exc:
        errors.append(f"sina={exc!r}")

    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=market_symbol,
            start_date=cfg.data.start_date,
            end_date=end_date,
            adjust=cfg.data.adjust,
        )
        if "volume" not in df.columns and "amount" in df.columns:
            df["volume"] = pd.to_numeric(df["amount"], errors="coerce") * 100
            df["amount"] = df["volume"] * pd.to_numeric(df["close"], errors="coerce")
        return normalize_history(df, symbol=symbol)
    except Exception as exc:
        errors.append(f"tencent={exc!r}")

    raise RuntimeError(f"Failed to fetch {symbol}: {'; '.join(errors)}")


def save_raw_history(df: pd.DataFrame, raw_dir: Path, symbol: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{symbol}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_history(cfg: AppConfig, symbols: list[str] | None = None) -> pd.DataFrame:
    selected = symbols or cfg.data.symbols
    frames: list[pd.DataFrame] = []
    for symbol in selected:
        local = read_local_history(cfg.data.raw_dir, symbol)
        if local is None:
            if not cfg.data.allow_download:
                raise FileNotFoundError(f"No local CSV found for {symbol} in {cfg.data.raw_dir}")
            local = fetch_history_with_akshare(symbol, cfg)
            save_raw_history(local, cfg.data.raw_dir, symbol)
        frames.append(local)

    if not frames:
        raise ValueError("No symbols were provided.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
