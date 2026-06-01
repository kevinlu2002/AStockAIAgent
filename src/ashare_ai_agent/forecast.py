from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pandas.tseries.offsets import BDay

from .config import AppConfig
from .features import FEATURE_COLUMNS, build_feature_table, coerce_model_features
from .knowledge import load_kline_knowledge
from .models import load_model


def _select_model(bundle: dict[str, Any], days: int) -> tuple[int, Any, Any]:
    horizon_models = bundle.get("horizon_models")
    if isinstance(horizon_models, dict) and horizon_models:
        available = sorted(int(h) for h in horizon_models)
        selected = min(available, key=lambda h: abs(h - days))
        payload = horizon_models[str(selected)]
        return selected, payload["return_model"], payload["drawdown_model"]
    return int(bundle["horizon_days"]), bundle["return_model"], bundle["drawdown_model"]


def _kline_bias(row: pd.Series, weights: dict[str, float]) -> float:
    positive = (
        weights.get("trend_follow", 1.0) * 0.25 * float(row.get("ma_bull_stack", 0) or 0)
        + weights.get("breakout", 1.0) * 0.30 * float(row.get("breakout_20", 0) or 0)
        + weights.get("volume_confirm", 1.0) * 0.22 * float(row.get("volume_price_confirm", 0) or 0)
        + weights.get("macd_momentum", 1.0) * 3.0 * max(0.0, float(row.get("macd_hist", 0) or 0))
        + weights.get("reversal_candle", 1.0) * 0.12 * float(row.get("hammer", 0) or 0)
        + weights.get("reversal_candle", 1.0) * 0.15 * float(row.get("bullish_engulfing", 0) or 0)
    )
    negative = (
        weights.get("trend_follow", 1.0) * 0.25 * float(row.get("ma_bear_stack", 0) or 0)
        + weights.get("breakout", 1.0) * 0.30 * float(row.get("breakdown_20", 0) or 0)
        + weights.get("macd_momentum", 1.0) * 3.0 * max(0.0, -float(row.get("macd_hist", 0) or 0))
        + weights.get("reversal_candle", 1.0) * 0.12 * float(row.get("shooting_star", 0) or 0)
        + weights.get("reversal_candle", 1.0) * 0.15 * float(row.get("bearish_engulfing", 0) or 0)
    )
    ret_20 = float(row.get("ret_20", 0.0) or 0.0)
    ret_60 = float(row.get("ret_60", 0.0) or 0.0)
    volume_z = float(row.get("volume_z_20", 0.0) or 0.0)
    close_to_high = float(row.get("close_to_high_20", -1.0) or -1.0)
    if (ret_20 > 0.22 or ret_60 > 0.55) and (close_to_high > -0.04 or volume_z > 1.8):
        negative += weights.get("anti_chase", 1.0) * 0.20
        negative += weights.get("expectation_exhaustion", 1.0) * 0.14
    if float(row.get("volatility_20", 0.0) or 0.0) > 0.055:
        negative += weights.get("risk_discipline", 1.0) * 0.12
    return max(-1.0, min(1.0, positive - negative))


def _future_dates(latest_date: pd.Timestamp, days: int) -> list[pd.Timestamp]:
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    base = max(pd.Timestamp(latest_date).normalize(), today)
    return [(base + BDay(i)).normalize() for i in range(1, days + 1)]


def forecast_kline(history: pd.DataFrame, cfg: AppConfig, symbol: str, days: int = 5) -> dict[str, Any]:
    days = max(1, min(int(days), 20))
    symbol = str(symbol).zfill(6)
    hist = history[history["symbol"].astype(str).str.zfill(6) == symbol].sort_values("date").copy()
    if len(hist) < cfg.model.min_rows_per_symbol:
        raise ValueError(f"{symbol} 历史数据不足，无法预测未来K线")

    bundle = load_model(cfg.model.model_dir)
    model_features = list(bundle.get("feature_columns") or FEATURE_COLUMNS)
    horizon, return_model, drawdown_model = _select_model(bundle, days)
    table = coerce_model_features(build_feature_table(hist, horizon=horizon), model_features).dropna(subset=model_features)
    if table.empty:
        raise ValueError(f"{symbol} 特征不足，无法预测未来K线")

    latest = table.sort_values("date").iloc[-1]
    model_input = pd.DataFrame([latest[model_features].to_dict()]).astype("float64")
    model_return = float(return_model.predict(model_input)[0])
    model_drawdown = float(drawdown_model.predict(model_input)[0])
    knowledge = load_kline_knowledge(cfg)
    weights = knowledge.get("weights", {})
    bias = _kline_bias(latest, weights if isinstance(weights, dict) else {})

    scale = max(0.2, min(6.0, days / max(horizon, 1)))
    target_return = (1 + max(-0.95, min(2.0, model_return))) ** scale - 1
    target_return = max(-0.35, min(0.45, target_return + 0.018 * bias))
    drawdown = max(-0.45, min(-0.001, model_drawdown * math.sqrt(scale)))
    recent = hist.tail(40)
    close_series = recent["close"].astype(float)
    recent_vol = float(close_series.pct_change().tail(20).std() or 0.015)
    atr_pct = float(latest.get("atr_pct_14", recent_vol * 1.7) or recent_vol * 1.7)
    atr_pct = max(0.006, min(0.12, atr_pct))
    last_close = float(close_series.iloc[-1])
    log_target = math.log(max(0.05, 1 + target_return))
    dates = _future_dates(pd.Timestamp(hist["date"].max()), days)

    rows: list[dict[str, Any]] = []
    previous_close = last_close
    for index, date in enumerate(dates, start=1):
        progress = index / days
        wave = math.sin(index * 1.37) * recent_vol * 0.55 * (1 - abs(progress - 0.5))
        if bias < -0.25:
            wave -= recent_vol * 0.20 * (1 - progress)
        elif bias > 0.25:
            wave += recent_vol * 0.16 * (1 - progress)
        close = last_close * math.exp(log_target * progress + wave)
        open_price = previous_close * (1 + (close / previous_close - 1) * 0.28)
        range_pct = atr_pct * (0.65 + 0.10 * math.sin(index))
        high = max(open_price, close) * (1 + range_pct * 0.42)
        low = min(open_price, close) * (1 - range_pct * 0.42)
        if index <= max(2, days // 2):
            low = min(low, last_close * (1 + drawdown * min(1.0, index / max(1, days // 2))) * 1.01)
        volume = float(recent["volume"].tail(20).mean() or 0) * (1 + min(0.45, abs(bias) * 0.22))
        rows.append(
            {
                "date": date.date().isoformat(),
                "open": round(open_price, 3),
                "high": round(max(high, open_price, close), 3),
                "low": round(min(low, open_price, close), 3),
                "close": round(close, 3),
                "volume": round(volume, 0),
                "forecast": True,
            }
        )
        previous_close = close

    if target_return > 0.015 and bias >= -0.35:
        timing = f"{rows[0]['date']} 09:40 至 {rows[min(days - 1, 2)]['date']} 10:30，回踩预测区间低位且量能不转弱时分批"
    elif target_return > 0:
        timing = f"{rows[0]['date']} 至 {rows[min(days - 1, 3)]['date']}，只在回踩支撑且重新放量时观察"
    else:
        timing = f"{rows[0]['date']} 之后暂不主动买入，等待重新站上短期均线"

    return {
        "symbol": symbol,
        "days": days,
        "selected_model_horizon": horizon,
        "target_return": round(target_return, 4),
        "target_drawdown": round(drawdown, 4),
        "kline_bias": round(bias, 4),
        "confidence": round(max(30, min(88, 62 + bias * 12 - recent_vol * 260)), 1),
        "suggested_future_entry_window": timing,
        "knowledge_generated_at": knowledge.get("generated_at"),
        "rows": rows,
    }
