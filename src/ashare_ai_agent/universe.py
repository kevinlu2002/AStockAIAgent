from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_BUCKETS = [
    ("0-10", 0.0, 10.0),
    ("10-100", 10.0, 100.0),
    ("100+", 100.0, np.inf),
]


@dataclass(frozen=True)
class UniverseBuildResult:
    spot_path: Path
    selected_path: Path
    index_path: Path | None
    selected: pd.DataFrame


def price_bucket(price: float) -> str:
    for name, lo, hi in PRICE_BUCKETS:
        if price >= lo and price < hi:
            return name
    return "unknown"


def add_price_bucket(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    out = df.copy()
    out["price_bucket"] = out[price_col].astype(float).map(price_bucket)
    return out


def fetch_spot_universe() -> pd.DataFrame:
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("AkShare is required for live A-share universe fetch.") from exc

    raw = ak.stock_zh_a_spot_em()
    col_map = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "pct_chg",
        "涨跌额": "chg",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "最高": "high",
        "最低": "low",
        "今开": "open",
        "昨收": "prev_close",
        "量比": "volume_ratio",
        "换手率": "turnover",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb",
        "总市值": "market_cap",
        "流通市值": "float_market_cap",
        "涨速": "rise_speed",
        "5分钟涨跌": "pct_chg_5m",
        "60日涨跌幅": "pct_chg_60d",
        "年初至今涨跌幅": "pct_chg_ytd",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns}).copy()
    if "symbol" not in df.columns or "price" not in df.columns:
        raise ValueError(f"Unexpected spot columns: {list(raw.columns)}")

    df["symbol"] = df["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    for col in [
        "price",
        "amount",
        "volume",
        "turnover",
        "pct_chg",
        "market_cap",
        "float_market_cap",
        "pe_ttm",
        "pb",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["name"] = df.get("name", "").astype(str)
    return add_price_bucket(df, price_col="price")


def fetch_index_constituent_universe(index_symbols: list[str] | None = None) -> pd.DataFrame:
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("AkShare is required for index constituent universe fetch.") from exc

    index_symbols = index_symbols or ["000300", "000905", "000852"]
    frames: list[pd.DataFrame] = []
    for index_symbol in index_symbols:
        try:
            raw = ak.index_stock_cons_sina(symbol=index_symbol)
        except Exception:
            try:
                raw = ak.index_stock_cons(symbol=index_symbol).rename(columns={"品种代码": "code", "品种名称": "name"})
            except Exception:
                continue

        df = raw.copy()
        if "symbol" in df.columns and "code" in df.columns:
            df = df.rename(columns={"symbol": "market_symbol"})
        rename = {
            "code": "symbol",
            "trade": "price",
            "changepercent": "pct_chg",
            "turnoverratio": "turnover",
            "mktcap": "market_cap",
            "nmc": "float_market_cap",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "symbol" not in df.columns and "品种代码" in df.columns:
            df["symbol"] = df["品种代码"]
        if "name" not in df.columns and "品种名称" in df.columns:
            df["name"] = df["品种名称"]
        df["index_symbol"] = index_symbol
        frames.append(df)

    if not frames:
        raise RuntimeError("No index constituent universe source is available.")
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    out["name"] = out.get("name", "").astype(str)
    for col in ["price", "amount", "volume", "turnover", "pct_chg", "market_cap", "float_market_cap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "price" not in out.columns:
        out["price"] = np.nan
    if "amount" not in out.columns:
        out["amount"] = 0.0
    out = out.sort_values("amount", ascending=False).drop_duplicates("symbol", keep="first")
    return add_price_bucket(out, price_col="price").reset_index(drop=True)


def fetch_index_snapshot() -> pd.DataFrame | None:
    try:
        import akshare as ak  # type: ignore

        frames = []
        for symbol in ["沪深重要指数", "上证系列指数", "深证系列指数"]:
            try:
                frame = ak.stock_zh_index_spot_em(symbol=symbol)
                frame["指数分组"] = symbol
                frames.append(frame)
            except Exception:
                continue
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)
    except Exception:
        return None


def filter_tradeable_universe(spot: pd.DataFrame, include_beijing: bool = False) -> pd.DataFrame:
    df = spot.copy()
    prefixes = ("0", "3", "6", "8", "4") if include_beijing else ("0", "3", "6")
    df = df[df["symbol"].str.startswith(prefixes)]
    df = df[df["price"].notna() & (df["price"] > 0)]
    if "amount" in df.columns:
        df = df[df["amount"].fillna(0) > 0]
    name = df["name"].fillna("")
    st_or_delist = name.str.contains("ST|退", case=False, regex=True)
    fresh_listing = name.str.match(r"^[NCU]", case=False, na=False)
    df = df[~st_or_delist & ~fresh_listing]
    return df.reset_index(drop=True)


def select_liquid_by_bucket(
    spot: pd.DataFrame,
    per_bucket: int,
    min_amount: float,
) -> pd.DataFrame:
    df = filter_tradeable_universe(spot)
    if "amount" not in df.columns:
        df["amount"] = 0.0
    df = df[df["amount"].fillna(0) >= min_amount]
    selected = []
    for bucket, _, _ in PRICE_BUCKETS:
        bucket_df = df[df["price_bucket"] == bucket].sort_values("amount", ascending=False)
        selected.append(bucket_df.head(per_bucket))
    out = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    return out.sort_values(["price_bucket", "amount"], ascending=[True, False]).reset_index(drop=True)


def build_and_save_universe(
    processed_dir: Path,
    per_bucket: int = 20,
    min_amount: float = 50_000_000,
) -> UniverseBuildResult:
    processed_dir.mkdir(parents=True, exist_ok=True)
    try:
        spot = fetch_spot_universe()
    except Exception:
        spot = fetch_index_constituent_universe()
    selected = select_liquid_by_bucket(spot, per_bucket=per_bucket, min_amount=min_amount)
    index_snapshot = fetch_index_snapshot()

    spot_path = processed_dir / "latest_spot_universe.csv"
    selected_path = processed_dir / "selected_liquid_universe.csv"
    index_path = processed_dir / "latest_market_indices.csv"
    spot.to_csv(spot_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    if index_snapshot is not None:
        index_snapshot.to_csv(index_path, index=False, encoding="utf-8-sig")
    else:
        index_path = None
    return UniverseBuildResult(spot_path, selected_path, index_path, selected)
