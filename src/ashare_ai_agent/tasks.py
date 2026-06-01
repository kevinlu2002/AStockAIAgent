from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from .config import AppConfig
from .data import fetch_history_with_akshare, normalize_history, read_local_history, save_raw_history
from .models import train_proxy_model


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
ProgressCallback = Callable[[dict[str, object]], None]


def today_yyyymmdd() -> str:
    return datetime.now(SHANGHAI_TZ).strftime("%Y%m%d")


def cached_symbols(cfg: AppConfig, limit: int | None = None) -> list[str]:
    symbols: list[str] = []
    selected_path = cfg.project_root / "data" / "processed" / "selected_liquid_universe.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path, dtype={"symbol": str})
        symbols.extend(selected["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).dropna().tolist())
    for file in sorted(cfg.data.raw_dir.glob("*.csv")):
        symbol = file.stem[:6]
        if symbol.isdigit():
            symbols.append(symbol)
    deduped = sorted(set(str(symbol).zfill(6) for symbol in symbols))
    if limit and limit > 0:
        return deduped[:limit]
    return deduped


def _cfg_with_symbols_and_today(cfg: AppConfig, symbols: list[str]) -> AppConfig:
    return replace(cfg, data=replace(cfg.data, symbols=symbols, end_date=today_yyyymmdd(), allow_download=True))


def update_histories(
    cfg: AppConfig,
    symbols: list[str],
    max_workers: int = 8,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    cfg = _cfg_with_symbols_and_today(cfg, symbols)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    total = len(symbols)
    if total == 0:
        raise RuntimeError("没有可更新的股票代码")

    def fetch_one(symbol: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            hist = fetch_history_with_akshare(symbol, cfg)
            hist = normalize_history(hist, symbol=symbol)
            save_raw_history(hist, cfg.data.raw_dir, symbol)
            return symbol, hist, None
        except Exception as exc:
            local = read_local_history(cfg.data.raw_dir, symbol)
            if local is not None and len(local) >= cfg.model.min_rows_per_symbol:
                return symbol, local, None
            return symbol, None, repr(exc)

    workers = max(1, min(max_workers, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, symbol) for symbol in symbols]
        for done, future in enumerate(as_completed(futures), start=1):
            symbol, hist, error = future.result()
            if hist is None:
                failures.append({"symbol": symbol, "error": str(error)})
            else:
                frames.append(hist)
            if progress:
                progress(
                    {
                        "stage": "更新历史行情",
                        "processed": done,
                        "total": total,
                        "message": symbol,
                        "progress": round(done / total * 70, 1),
                    }
                )

    if not frames:
        raise RuntimeError("没有成功更新任何历史行情")
    history = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    return history, failures


def run_daily_update_and_train(
    cfg: AppConfig,
    max_workers: int = 8,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    started_at = datetime.now(SHANGHAI_TZ)
    symbols = cached_symbols(cfg, limit=limit)
    if progress:
        progress({"stage": "准备每日更新", "processed": 0, "total": len(symbols), "message": f"{len(symbols)} 只股票"})

    history, failures = update_histories(cfg, symbols, max_workers=max_workers, progress=progress)
    trained_symbols = sorted(history["symbol"].unique().tolist())
    train_cfg = _cfg_with_symbols_and_today(cfg, trained_symbols)
    if progress:
        progress({"stage": "重新训练模型", "processed": len(trained_symbols), "total": len(trained_symbols), "progress": 85})
    result = train_proxy_model(history, train_cfg)
    finished_at = datetime.now(SHANGHAI_TZ)
    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "symbols_requested": len(symbols),
        "symbols_trained": len(trained_symbols),
        "failures": failures[:50],
        "metrics": result["metrics"],
        "model_path": str(result["model_path"]),
    }
    out_path = cfg.output.reports_dir / "daily_update_train_status.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress({"stage": "每日更新完成", "processed": len(trained_symbols), "total": len(symbols), "progress": 100})
    return payload


def last_daily_status(cfg: AppConfig) -> dict[str, object] | None:
    path = cfg.output.reports_dir / "daily_update_train_status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
