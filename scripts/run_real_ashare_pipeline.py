from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for _proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(_proxy_key, None)
os.environ.setdefault("NO_PROXY", "*")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ashare_ai_agent.config import AppConfig, DataConfig, load_config
from ashare_ai_agent.data import fetch_history_with_akshare, normalize_history, save_raw_history
from ashare_ai_agent.features import FEATURE_COLUMNS, build_feature_table, training_rows
from ashare_ai_agent.models import load_model, predict_positive_probability, train_proxy_model
from ashare_ai_agent.recommend import make_recommendations, write_recommendation_reports
from ashare_ai_agent.universe import add_price_bucket, build_and_save_universe


def _today_yyyymmdd() -> str:
    return pd.Timestamp.today(tz="Asia/Shanghai").strftime("%Y%m%d")


def _override_dates(cfg: AppConfig, start_date: str, end_date: str, symbols: list[str]) -> AppConfig:
    data = replace(
        cfg.data,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        allow_download=True,
    )
    return replace(cfg, data=data)


def _fetch_histories(cfg: AppConfig, selected: pd.DataFrame, sleep_seconds: float, max_failures: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for i, row in selected.reset_index(drop=True).iterrows():
        symbol = str(row["symbol"]).zfill(6)
        name = str(row.get("name", ""))
        print(f"[{i + 1}/{len(selected)}] fetching {symbol} {name}")
        try:
            hist = fetch_history_with_akshare(symbol, cfg)
            hist = normalize_history(hist, symbol=symbol)
            hist["name"] = name
            save_raw_history(hist.drop(columns=["name"]), cfg.data.raw_dir, symbol)
            frames.append(hist)
        except Exception as exc:
            failures.append({"symbol": symbol, "name": name, "error": repr(exc)})
            print(f"  failed: {exc!r}")
            if len(failures) > max_failures:
                break
        time.sleep(max(0.0, sleep_seconds))

    if failures:
        fail_path = cfg.project_root / "reports" / "history_fetch_failures.json"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    if not frames:
        raise RuntimeError("No history data was fetched successfully.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def _bucket_metrics(valid: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, group in valid.groupby("price_bucket"):
        if group.empty:
            continue
        y = group[f"target_return_{horizon}d"].to_numpy()
        pred = group["pred_return"].to_numpy()
        if "pred_direction_prob" in group:
            pred_direction = (group["pred_direction_prob"].to_numpy() >= 0.5).astype(bool)
        else:
            pred_direction = pred > 0
        trend_col = f"target_trend_return_{horizon}d"
        trend_accuracy = None
        if "pred_trend_prob" in group and trend_col in group:
            trend_accuracy = float(((group["pred_trend_prob"].to_numpy() >= 0.5) == (group[trend_col].to_numpy() > 0)).mean())
        top = (
            group.sort_values(["date", "score"], ascending=[True, False])
            .groupby("date")
            .head(3)
        )
        rows.append(
            {
                "price_bucket": bucket,
                "rows": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "return_mae": float(mean_absolute_error(y, pred)),
                "return_rmse": float(mean_squared_error(y, pred) ** 0.5),
                "direction_hit_rate": float((pred_direction == (y > 0)).mean()),
                "trend_hit_rate": trend_accuracy,
                "top3_daily_mean_forward_return": float(top[f"target_return_{horizon}d"].mean()),
                "top3_daily_win_rate": float((top[f"target_return_{horizon}d"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("price_bucket").reset_index(drop=True)


def _write_real_test_report(history: pd.DataFrame, cfg: AppConfig, spot: pd.DataFrame) -> tuple[Path, Path]:
    bundle = load_model(cfg.model.model_dir)
    horizon = int(bundle["horizon_days"])
    horizon_payload = {}
    if isinstance(bundle.get("horizon_models"), dict):
        horizon_payload = bundle["horizon_models"].get(str(horizon), {})
    return_model = horizon_payload.get("validation_return_model") or bundle["return_model"]
    drawdown_model = horizon_payload.get("validation_drawdown_model") or bundle["drawdown_model"]
    direction_model = horizon_payload.get("validation_direction_model") or horizon_payload.get("direction_model") or bundle.get("direction_model")
    trend_model = horizon_payload.get("validation_trend_model") or horizon_payload.get("trend_model") or bundle.get("trend_model")
    table = build_feature_table(history.drop(columns=["name"], errors="ignore"), horizon=horizon)
    rows = training_rows(table, horizon=horizon)
    dates = np.array(sorted(rows["date"].dropna().unique()))
    split_index = max(1, int(len(dates) * (1 - cfg.model.validation_fraction)))
    split_index = min(split_index, len(dates) - 1)
    cutoff = dates[split_index]
    valid = rows[rows["date"] >= cutoff].copy()
    valid["pred_return"] = return_model.predict(valid[FEATURE_COLUMNS])
    valid["pred_drawdown"] = drawdown_model.predict(valid[FEATURE_COLUMNS])
    if direction_model is not None:
        valid["pred_direction_prob"] = predict_positive_probability(direction_model, valid[FEATURE_COLUMNS])
    if trend_model is not None:
        valid["pred_trend_prob"] = predict_positive_probability(trend_model, valid[FEATURE_COLUMNS])
    valid["score"] = valid["pred_return"] / valid["pred_drawdown"].abs().clip(lower=0.01)
    if "pred_direction_prob" in valid:
        valid["score"] += (valid["pred_direction_prob"] - 0.5) * 1.50
    if "pred_trend_prob" in valid:
        valid["score"] += (valid["pred_trend_prob"] - 0.5) * 1.00
    valid = add_price_bucket(valid.rename(columns={"close": "price"}), price_col="price").rename(columns={"price": "close"})

    name_map = spot[["symbol", "name"]].drop_duplicates("symbol")
    valid = valid.merge(name_map, on="symbol", how="left")

    cfg.output.reports_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cfg.output.reports_dir / "real_validation_predictions.csv"
    metric_path = cfg.output.reports_dir / "real_data_test_by_price_bucket.csv"
    valid.to_csv(pred_path, index=False, encoding="utf-8-sig")
    _bucket_metrics(valid, horizon=horizon).to_csv(metric_path, index=False, encoding="utf-8-sig")
    return pred_path, metric_path


def _write_bucket_recommendations(rec: pd.DataFrame, spot: pd.DataFrame, reports_dir: Path) -> tuple[pd.DataFrame, Path]:
    meta_cols = ["symbol", "name", "price", "price_bucket", "amount", "turnover", "pct_chg"]
    meta = spot[[c for c in meta_cols if c in spot.columns]].drop_duplicates("symbol")
    merged = rec.merge(meta, on="symbol", how="left")
    if "price_bucket" not in merged.columns:
        merged = add_price_bucket(merged.rename(columns={"reference_entry": "price"}), "price").rename(columns={"price": "reference_entry"})

    action_rank = {"BUY": 0, "WATCH": 1, "AVOID": 2}
    merged["_rank"] = merged["action"].map(action_rank).fillna(9)
    best = (
        merged.sort_values(["price_bucket", "_rank", "score"], ascending=[True, True, False])
        .groupby("price_bucket")
        .head(3)
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    path = reports_dir / "best_recommendations_by_price_bucket.csv"
    best.to_csv(path, index=False, encoding="utf-8-sig")
    return best, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live A-share data, train, test, and recommend by price bucket.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default=_today_yyyymmdd())
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--min-amount", type=float, default=50_000_000)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--max-failures", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = cfg.project_root / "data" / "processed"
    print("fetching full A-share spot universe")
    universe = build_and_save_universe(processed_dir, per_bucket=args.per_bucket, min_amount=args.min_amount)
    selected = universe.selected
    if selected.empty:
        raise RuntimeError("Selected universe is empty. Lower --min-amount or check data source.")
    symbols = selected["symbol"].astype(str).str.zfill(6).tolist()
    cfg = _override_dates(cfg, args.start_date, args.end_date, symbols=symbols)

    print(f"selected {len(selected)} symbols from latest spot universe")
    history = _fetch_histories(cfg, selected, sleep_seconds=args.sleep_seconds, max_failures=args.max_failures)
    fetched_symbols = sorted(history["symbol"].unique().tolist())
    cfg = _override_dates(cfg, args.start_date, args.end_date, symbols=fetched_symbols)

    print(f"training proxy model on {len(fetched_symbols)} real A-share symbols")
    result = train_proxy_model(history.drop(columns=["name"], errors="ignore"), cfg)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    pred_path, bucket_metric_path = _write_real_test_report(history, cfg, universe.selected)
    rec = make_recommendations(history.drop(columns=["name"], errors="ignore"), cfg, symbols=fetched_symbols, capital=args.capital)
    csv_path, md_path = write_recommendation_reports(rec, cfg.output.reports_dir)
    best, best_path = _write_bucket_recommendations(rec, universe.selected, cfg.output.reports_dir)

    print(f"validation predictions: {pred_path}")
    print(f"bucket test metrics: {bucket_metric_path}")
    print(f"recommendations: {csv_path}")
    print(f"recommendations markdown: {md_path}")
    print(f"best by price bucket: {best_path}")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
