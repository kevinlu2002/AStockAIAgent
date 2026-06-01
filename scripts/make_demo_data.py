from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_symbol(symbol: str, dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates)
    market = rng.normal(0.0002, 0.012, n)
    cycle = 0.004 * np.sin(np.linspace(0, 18, n))
    momentum = np.zeros(n)
    for i in range(1, n):
        momentum[i] = 0.88 * momentum[i - 1] + rng.normal(0, 0.004)
    returns = market + cycle + momentum * 0.35
    close = 20 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.025, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.025, n))
    volume = rng.lognormal(mean=14.5, sigma=0.35, size=n).astype(int)
    amount = volume * close
    turnover = rng.uniform(0.2, 3.0, n)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "open": open_.round(3),
            "high": high.round(3),
            "low": low.round(3),
            "close": close.round(3),
            "volume": volume,
            "amount": amount.round(2),
            "turnover": turnover.round(3),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--symbols", default="000001,600519,300750")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-05-26")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(args.start, args.end)
    for idx, symbol in enumerate([s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]):
        df = make_symbol(symbol, dates, seed=2026 + idx)
        path = out_dir / f"{symbol}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"wrote {path} rows={len(df)}")


if __name__ == "__main__":
    main()
