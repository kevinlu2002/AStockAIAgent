from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "volatility_5",
    "volatility_20",
    "range_pct",
    "close_to_sma_5",
    "close_to_sma_20",
    "close_to_sma_60",
    "sma_5_to_20",
    "sma_20_to_60",
    "volume_z_20",
    "amount_z_20",
    "turnover",
    "limit_up_touch",
    "limit_up_close",
    "failed_limit_up",
    "limit_up_strength",
    "failed_limit_up_turnover",
    "atr_pct_14",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ma_bull_stack",
    "ma_bear_stack",
    "golden_cross_5_20",
    "dead_cross_5_20",
    "close_to_high_20",
    "close_to_low_20",
    "breakout_20",
    "breakdown_20",
    "volume_price_confirm",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_pct",
    "hammer",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",
]


def coerce_model_features(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = list(columns or FEATURE_COLUMNS)
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")

    out = frame.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).astype("float64")
    return out


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
    return rsi


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return ((series - mean) / std.replace(0, np.nan)).fillna(0.0)


def _future_rolling_min(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def _limit_up_threshold(symbol: str) -> float:
    code = "".join(ch for ch in str(symbol) if ch.isdigit()).zfill(6)
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def _feature_one(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    close = g["close"]
    high = g["high"]
    low = g["low"]
    volume = g["volume"]
    amount = g["amount"]

    ret_1 = close.pct_change()
    g["ret_1"] = ret_1
    for window in [3, 5, 10, 20, 60]:
        g[f"ret_{window}"] = close.pct_change(window)

    previous_close = close.shift(1)
    price_base = previous_close.replace(0, np.nan)
    limit_threshold = _limit_up_threshold(str(g["symbol"].iloc[0]))
    intraday_high_ret = high / price_base - 1
    close_ret = close / price_base - 1
    limit_up_touch = intraday_high_ret >= limit_threshold - 0.006
    limit_up_close = limit_up_touch & (close_ret >= limit_threshold - 0.008)
    g["limit_up_touch"] = limit_up_touch.astype(float)
    g["limit_up_close"] = limit_up_close.astype(float)
    g["failed_limit_up"] = (limit_up_touch & ~limit_up_close).astype(float)
    g["limit_up_strength"] = (close_ret / limit_threshold).clip(lower=-3.0, upper=1.5)

    g["volatility_5"] = ret_1.rolling(5, min_periods=5).std()
    g["volatility_20"] = ret_1.rolling(20, min_periods=20).std()
    g["range_pct"] = (high - low) / close.replace(0, np.nan)

    sma_5 = close.rolling(5, min_periods=5).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()
    sma_60 = close.rolling(60, min_periods=60).mean()
    g["close_to_sma_5"] = close / sma_5 - 1
    g["close_to_sma_20"] = close / sma_20 - 1
    g["close_to_sma_60"] = close / sma_60 - 1
    g["sma_5_to_20"] = sma_5 / sma_20 - 1
    g["sma_20_to_60"] = sma_20 / sma_60 - 1
    g["ma_bull_stack"] = ((sma_5 > sma_20) & (sma_20 > sma_60) & (close > sma_20)).astype(float)
    g["ma_bear_stack"] = ((sma_5 < sma_20) & (sma_20 < sma_60) & (close < sma_20)).astype(float)
    g["golden_cross_5_20"] = ((sma_5 > sma_20) & (sma_5.shift(1) <= sma_20.shift(1))).astype(float)
    g["dead_cross_5_20"] = ((sma_5 < sma_20) & (sma_5.shift(1) >= sma_20.shift(1))).astype(float)
    high_20 = high.shift(1).rolling(20, min_periods=20).max()
    low_20 = low.shift(1).rolling(20, min_periods=20).min()
    g["close_to_high_20"] = close / high_20 - 1
    g["close_to_low_20"] = close / low_20 - 1
    g["breakout_20"] = ((close > high_20) & (volume > volume.rolling(20, min_periods=20).mean())).astype(float)
    g["breakdown_20"] = ((close < low_20) & (volume > volume.rolling(20, min_periods=20).mean())).astype(float)

    g["volume_z_20"] = _zscore(volume, 20)
    g["amount_z_20"] = _zscore(amount, 20)
    g["turnover"] = pd.to_numeric(g["turnover"], errors="coerce").fillna(0.0)
    g["failed_limit_up_turnover"] = g["failed_limit_up"] * g["turnover"]
    g["volume_price_confirm"] = ((ret_1 > 0) & (g["volume_z_20"] > 1.0)).astype(float)

    body = (close - g["open"]).abs()
    candle_range = (high - low).replace(0, np.nan)
    upper_shadow = high - pd.concat([close, g["open"]], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, g["open"]], axis=1).min(axis=1) - low
    g["upper_shadow_pct"] = upper_shadow / candle_range
    g["lower_shadow_pct"] = lower_shadow / candle_range
    g["body_pct"] = body / candle_range
    red = close > g["open"]
    green = close < g["open"]
    prev_red = red.shift(1).fillna(False)
    prev_green = green.shift(1).fillna(False)
    prev_open = g["open"].shift(1)
    prev_close = close.shift(1)
    g["hammer"] = ((g["lower_shadow_pct"] > 0.55) & (g["upper_shadow_pct"] < 0.20) & (g["body_pct"] < 0.35)).astype(float)
    g["shooting_star"] = ((g["upper_shadow_pct"] > 0.55) & (g["lower_shadow_pct"] < 0.20) & (g["body_pct"] < 0.35)).astype(float)
    g["bullish_engulfing"] = (red & prev_green & (close >= prev_open) & (g["open"] <= prev_close)).astype(float)
    g["bearish_engulfing"] = (green & prev_red & (g["open"] >= prev_close) & (close <= prev_open)).astype(float)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.rolling(14, min_periods=14).mean()
    g["atr_pct_14"] = atr_14 / close.replace(0, np.nan)

    g["rsi_14"] = _rsi(close)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    g["macd"] = macd / close.replace(0, np.nan)
    g["macd_signal"] = macd_signal / close.replace(0, np.nan)
    g["macd_hist"] = (macd - macd_signal) / close.replace(0, np.nan)

    g[f"target_return_{horizon}d"] = close.shift(-horizon) / close - 1
    future_returns = pd.concat([close.shift(-i) / close - 1 for i in range(1, horizon + 1)], axis=1)
    g[f"target_trend_return_{horizon}d"] = future_returns.mean(axis=1).where(future_returns.notna().all(axis=1))
    future_min_low = _future_rolling_min(low, horizon)
    g[f"target_drawdown_{horizon}d"] = future_min_low / close - 1
    return g


def build_feature_table(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    needed = {"date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover"}
    missing = needed - set(history.columns)
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")

    frames = [_feature_one(g, horizon) for _, g in history.groupby("symbol", sort=False)]
    features = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    return features


def training_rows(feature_table: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_cols = [
        f"target_return_{horizon}d",
        f"target_trend_return_{horizon}d",
        f"target_drawdown_{horizon}d",
    ]
    cols = FEATURE_COLUMNS + target_cols
    return coerce_model_features(feature_table, cols).dropna(subset=cols).reset_index(drop=True)
