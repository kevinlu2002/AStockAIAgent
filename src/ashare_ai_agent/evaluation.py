from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.offsets import BDay

from .config import AppConfig
from .data import fetch_history_with_akshare, read_local_history, save_raw_history


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
FORECAST_SNAPSHOT = "forecast_snapshots.csv"
RECOMMENDATION_SNAPSHOT = "recommendation_snapshots.csv"
FORECAST_EVALUATION = "forecast_evaluations.csv"
RECOMMENDATION_EVALUATION = "recommendation_evaluations.csv"
SUMMARY_JSON = "forecast_evaluation_summary.json"


def _now_text() -> str:
    return datetime.now(SHANGHAI_TZ).isoformat(timespec="seconds")


def _stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _to_float(value: object, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() in {"", "-"}:
            return default
        if pd.isna(value):
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _safe_date(value: object) -> str:
    try:
        if value is None:
            return ""
        if isinstance(value, str) and value.strip() in {"", "-"}:
            return ""
        if pd.isna(value):
            return ""
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return ""


def _review_date(signal_date: object, holding_days: int) -> str:
    signal = _safe_date(signal_date)
    if not signal:
        return ""
    try:
        return (pd.Timestamp(signal) + BDay(max(1, int(holding_days)))).date().isoformat()
    except Exception:
        return ""


def _append_csv(path: Path, rows: list[dict[str, object]], dedupe_keys: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return path
    if path.exists():
        existing = pd.read_csv(path, dtype={"symbol": str})
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming
    keys = [key for key in (dedupe_keys or []) if key in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def append_kline_forecast_snapshot(
    cfg: AppConfig,
    forecast_payload: dict[str, object],
    *,
    base_date: object = "",
    base_close: object = None,
    source: str = "kline_api",
) -> Path:
    rows = forecast_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return cfg.output.reports_dir / FORECAST_SNAPSHOT

    generated_at = _now_text()
    symbol = str(forecast_payload.get("symbol") or "").zfill(6)
    days = int(forecast_payload.get("days") or len(rows))
    normalized_base_date = _safe_date(base_date)
    forecast_id = _stable_id("kline", source, symbol, normalized_base_date, generated_at[:10], days)
    out: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        target_date = _safe_date(row.get("date"))
        if not symbol.strip("0") or not target_date:
            continue
        out.append(
            {
                "forecast_id": forecast_id,
                "generated_at": generated_at,
                "generated_date": generated_at[:10],
                "source": source,
                "symbol": symbol,
                "base_date": normalized_base_date,
                "base_close": _to_float(base_close),
                "horizon_days": days,
                "days_ahead": index,
                "target_date": target_date,
                "predicted_open": _to_float(row.get("open")),
                "predicted_high": _to_float(row.get("high")),
                "predicted_low": _to_float(row.get("low")),
                "predicted_close": _to_float(row.get("close")),
                "predicted_volume": _to_float(row.get("volume")),
                "target_return": _to_float(forecast_payload.get("target_return")),
                "target_drawdown": _to_float(forecast_payload.get("target_drawdown")),
                "kline_bias": _to_float(forecast_payload.get("kline_bias")),
                "confidence": _to_float(forecast_payload.get("confidence")),
                "suggested_future_entry_window": forecast_payload.get("suggested_future_entry_window") or "",
            }
        )
    return _append_csv(cfg.output.reports_dir / FORECAST_SNAPSHOT, out, ["forecast_id", "target_date"])


def append_recommendation_snapshot(
    cfg: AppConfig,
    recommendations: pd.DataFrame,
    *,
    run_id: str,
    source: str,
    capital: float | None,
    holding_days: int | None,
    universe_mode: str,
) -> Path:
    if recommendations.empty:
        return cfg.output.reports_dir / RECOMMENDATION_SNAPSHOT

    generated_at = _now_text()
    out: list[dict[str, object]] = []
    for _, row in recommendations.iterrows():
        symbol = str(row.get("symbol") or "").zfill(6)
        if not symbol.strip("0"):
            continue
        requested_days = int(row.get("requested_holding_days") or holding_days or cfg.model.horizon_days)
        signal_date = _safe_date(row.get("signal_date"))
        out.append(
            {
                "snapshot_id": _stable_id("recommendation", run_id, symbol),
                "run_id": run_id,
                "generated_at": generated_at,
                "source": source,
                "universe_mode": universe_mode,
                "capital": _to_float(capital),
                "symbol": symbol,
                "name": row.get("name") or symbol,
                "signal_date": signal_date,
                "review_date": _review_date(signal_date, requested_days),
                "action": row.get("action") or "",
                "score": _to_float(row.get("score")),
                "pred_return": _to_float(row.get("pred_return")),
                "pred_drawdown": _to_float(row.get("pred_drawdown")),
                "requested_holding_days": requested_days,
                "reference_entry": _to_float(row.get("reference_entry")),
                "buy_price_low": _to_float(row.get("buy_price_low")),
                "buy_price_high": _to_float(row.get("buy_price_high")),
                "buy_price_range": row.get("buy_price_range") or "",
                "planned_buy_time": row.get("planned_buy_time") or "",
                "planned_review_time": row.get("planned_review_time") or "",
                "shares": int(row.get("shares") or 0),
                "cash": _to_float(row.get("cash"), 0.0),
                "stop_loss": _to_float(row.get("stop_loss")),
                "risk_amount": _to_float(row.get("risk_amount"), 0.0),
                "position_pct": _to_float(row.get("position_pct"), 0.0),
                "news_score": _to_float(row.get("news_score"), 0.0),
                "technical_signals": row.get("technical_signals") or "",
                "reason": row.get("reason") or "",
            }
        )
    return _append_csv(cfg.output.reports_dir / RECOMMENDATION_SNAPSHOT, out, ["snapshot_id"])


def _load_history(cfg: AppConfig, symbol: str, refresh_history: bool) -> pd.DataFrame | None:
    symbol = str(symbol).zfill(6)
    if refresh_history:
        try:
            hist = fetch_history_with_akshare(symbol, cfg)
            save_raw_history(hist, cfg.data.raw_dir, symbol)
            return hist
        except Exception:
            pass
    return read_local_history(cfg.data.raw_dir, symbol)


def _match_actual_row(hist: pd.DataFrame, target_date: object) -> pd.Series | None:
    target = pd.Timestamp(target_date).normalize()
    work = hist.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    exact = work[work["date"] == target]
    if not exact.empty:
        return exact.iloc[0]
    future = work[work["date"] > target].sort_values("date")
    if future.empty:
        return None
    row = future.iloc[0]
    if (pd.Timestamp(row["date"]) - target).days > 7:
        return None
    return row


def _direction_hit(predicted_return: float | None, actual_return: float | None) -> bool | None:
    if predicted_return is None or actual_return is None:
        return None
    if predicted_return == 0 or actual_return == 0:
        return abs(predicted_return - actual_return) < 1e-9
    return predicted_return > 0 and actual_return > 0 or predicted_return < 0 and actual_return < 0


def _pct_error(predicted: float | None, actual: float | None) -> float | None:
    if predicted is None or actual in (None, 0):
        return None
    return (predicted - actual) / actual


def _evaluate_kline_snapshots(raw: pd.DataFrame, cfg: AppConfig, refresh_history: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, group in raw.groupby(raw["symbol"].astype(str).str.zfill(6)):
        hist = _load_history(cfg, symbol, refresh_history)
        if hist is None or hist.empty:
            continue
        hist = hist.sort_values("date")
        for _, item in group.iterrows():
            actual = _match_actual_row(hist, item.get("target_date"))
            if actual is None:
                continue
            base_close = _to_float(item.get("base_close"))
            predicted_close = _to_float(item.get("predicted_close"))
            actual_close = _to_float(actual.get("close"))
            predicted_return = predicted_close / base_close - 1 if base_close and predicted_close else None
            actual_return = actual_close / base_close - 1 if base_close and actual_close else None
            rows.append(
                {
                    "forecast_id": item.get("forecast_id"),
                    "symbol": symbol,
                    "target_date": _safe_date(item.get("target_date")),
                    "actual_date": _safe_date(actual.get("date")),
                    "exact_date_match": _safe_date(item.get("target_date")) == _safe_date(actual.get("date")),
                    "days_ahead": item.get("days_ahead"),
                    "base_close": base_close,
                    "predicted_open": _to_float(item.get("predicted_open")),
                    "actual_open": _to_float(actual.get("open")),
                    "predicted_high": _to_float(item.get("predicted_high")),
                    "actual_high": _to_float(actual.get("high")),
                    "predicted_low": _to_float(item.get("predicted_low")),
                    "actual_low": _to_float(actual.get("low")),
                    "predicted_close": predicted_close,
                    "actual_close": actual_close,
                    "predicted_return": predicted_return,
                    "actual_return": actual_return,
                    "direction_hit": _direction_hit(predicted_return, actual_return),
                    "close_error_pct": _pct_error(predicted_close, actual_close),
                    "high_error_pct": _pct_error(_to_float(item.get("predicted_high")), _to_float(actual.get("high"))),
                    "low_error_pct": _pct_error(_to_float(item.get("predicted_low")), _to_float(actual.get("low"))),
                    "evaluated_at": _now_text(),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_recommendation_snapshots(raw: pd.DataFrame, cfg: AppConfig, refresh_history: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, group in raw.groupby(raw["symbol"].astype(str).str.zfill(6)):
        hist = _load_history(cfg, symbol, refresh_history)
        if hist is None or hist.empty:
            continue
        hist = hist.sort_values("date").copy()
        hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
        latest_date = hist["date"].max()
        for _, item in group.iterrows():
            signal_text = _safe_date(item.get("signal_date"))
            if not signal_text:
                continue
            signal_date = pd.Timestamp(signal_text).normalize()
            review_text = _safe_date(item.get("review_date"))
            review_date = pd.Timestamp(review_text).normalize() if review_text else latest_date
            window_end = min(review_date, latest_date)
            future = hist[(hist["date"] > signal_date) & (hist["date"] <= window_end)]
            if future.empty:
                continue
            reference = _to_float(item.get("reference_entry"))
            if not reference:
                continue
            end = future.iloc[-1]
            end_close = _to_float(end.get("close"))
            actual_return = end_close / reference - 1 if end_close else None
            actual_drawdown = future["low"].astype(float).min() / reference - 1
            pred_return = _to_float(item.get("pred_return"))
            buy_low = _to_float(item.get("buy_price_low"))
            buy_high = _to_float(item.get("buy_price_high"))
            buy_window_hit = None
            if buy_low is not None and buy_high is not None:
                buy_window_hit = bool(((future["low"] <= buy_high) & (future["high"] >= buy_low)).any())
            rows.append(
                {
                    "snapshot_id": item.get("snapshot_id"),
                    "run_id": item.get("run_id"),
                    "symbol": symbol,
                    "name": item.get("name") or symbol,
                    "action": item.get("action"),
                    "signal_date": _safe_date(signal_date),
                    "review_date": review_text,
                    "evaluated_until": _safe_date(end.get("date")),
                    "final_evaluation": latest_date >= review_date,
                    "reference_entry": reference,
                    "pred_return": pred_return,
                    "actual_return": actual_return,
                    "return_error": pred_return - actual_return if pred_return is not None and actual_return is not None else None,
                    "pred_drawdown": _to_float(item.get("pred_drawdown")),
                    "actual_drawdown": actual_drawdown,
                    "direction_hit": _direction_hit(pred_return, actual_return),
                    "buy_window_hit": buy_window_hit,
                    "buy_price_low": buy_low,
                    "buy_price_high": buy_high,
                    "evaluated_at": _now_text(),
                }
            )
    return pd.DataFrame(rows)


def _rate(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return round(float(clean.astype(bool).mean()), 4)


def _mean_abs(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return round(float(clean.abs().mean()), 6)


def _rmse(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return round(float((clean.pow(2).mean()) ** 0.5), 6)


def evaluate_saved_predictions(
    cfg: AppConfig,
    *,
    refresh_history: bool = False,
    symbols: list[str] | None = None,
) -> dict[str, object]:
    reports_dir = cfg.output.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    symbol_filter = {str(symbol).zfill(6) for symbol in symbols or []}

    forecast_eval = pd.DataFrame()
    forecast_path = reports_dir / FORECAST_SNAPSHOT
    if forecast_path.exists():
        raw_forecasts = pd.read_csv(forecast_path, dtype={"symbol": str})
        if symbol_filter:
            raw_forecasts = raw_forecasts[raw_forecasts["symbol"].astype(str).str.zfill(6).isin(symbol_filter)]
        if not raw_forecasts.empty:
            forecast_eval = _evaluate_kline_snapshots(raw_forecasts, cfg, refresh_history)
            if not forecast_eval.empty:
                forecast_eval.to_csv(reports_dir / FORECAST_EVALUATION, index=False, encoding="utf-8-sig")

    recommendation_eval = pd.DataFrame()
    recommendation_path = reports_dir / RECOMMENDATION_SNAPSHOT
    if recommendation_path.exists():
        raw_recommendations = pd.read_csv(recommendation_path, dtype={"symbol": str})
        if symbol_filter:
            raw_recommendations = raw_recommendations[
                raw_recommendations["symbol"].astype(str).str.zfill(6).isin(symbol_filter)
            ]
        if not raw_recommendations.empty:
            recommendation_eval = _evaluate_recommendation_snapshots(raw_recommendations, cfg, refresh_history)
            if not recommendation_eval.empty:
                recommendation_eval.to_csv(reports_dir / RECOMMENDATION_EVALUATION, index=False, encoding="utf-8-sig")

    summary = {
        "evaluated_at": _now_text(),
        "refresh_history": refresh_history,
        "kline": {
            "snapshots": int(pd.read_csv(forecast_path).shape[0]) if forecast_path.exists() else 0,
            "evaluated": int(len(forecast_eval)),
            "direction_hit_rate": _rate(forecast_eval["direction_hit"]) if "direction_hit" in forecast_eval else None,
            "mean_abs_close_error_pct": _mean_abs(forecast_eval["close_error_pct"]) if "close_error_pct" in forecast_eval else None,
            "rmse_close_error_pct": _rmse(forecast_eval["close_error_pct"]) if "close_error_pct" in forecast_eval else None,
        },
        "recommendations": {
            "snapshots": int(pd.read_csv(recommendation_path).shape[0]) if recommendation_path.exists() else 0,
            "evaluated": int(len(recommendation_eval)),
            "final_evaluated": int(recommendation_eval["final_evaluation"].sum()) if "final_evaluation" in recommendation_eval else 0,
            "direction_hit_rate": _rate(recommendation_eval["direction_hit"])
            if "direction_hit" in recommendation_eval
            else None,
            "buy_window_hit_rate": _rate(recommendation_eval["buy_window_hit"])
            if "buy_window_hit" in recommendation_eval
            else None,
            "mean_abs_return_error": _mean_abs(recommendation_eval["return_error"])
            if "return_error" in recommendation_eval
            else None,
        },
        "files": {
            "forecast_snapshots": str(forecast_path),
            "recommendation_snapshots": str(recommendation_path),
            "forecast_evaluations": str(reports_dir / FORECAST_EVALUATION),
            "recommendation_evaluations": str(reports_dir / RECOMMENDATION_EVALUATION),
            "summary": str(reports_dir / SUMMARY_JSON),
        },
    }
    (reports_dir / SUMMARY_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_evaluation_summary(cfg: AppConfig) -> dict[str, object]:
    path = cfg.output.reports_dir / SUMMARY_JSON
    if not path.exists():
        return {
            "evaluated_at": None,
            "kline": {"snapshots": 0, "evaluated": 0},
            "recommendations": {"snapshots": 0, "evaluated": 0},
            "files": {"summary": str(path)},
        }
    return json.loads(path.read_text(encoding="utf-8"))
