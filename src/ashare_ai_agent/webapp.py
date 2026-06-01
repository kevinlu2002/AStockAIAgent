from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path
import re
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for
import pandas as pd
import requests
from pandas.tseries.offsets import BDay

from .config import AppConfig, load_config
from .data import fetch_history_with_akshare, normalize_history, read_local_history, save_raw_history
from .evaluation import (
    append_kline_forecast_snapshot,
    append_recommendation_snapshot,
    evaluate_saved_predictions,
    load_evaluation_summary,
)
from .forecast import forecast_kline
from .knowledge import learn_kline_knowledge, load_kline_knowledge
from .news import analyze_news_impact
from .recommend import make_recommendations
from .tasks import last_daily_status, run_daily_update_and_train
from .universe import add_price_bucket


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
FETCH_WORKERS = max(1, min(int(os.environ.get("ASHARE_FETCH_WORKERS", "8")), 24))
AUTOMATION: dict[str, object] = {
    "started": False,
    "auto_news_enabled": False,
    "auto_knowledge_enabled": False,
    "auto_retrain_enabled": False,
    "news_running": False,
    "knowledge_running": False,
    "retrain_running": False,
    "last_news": None,
    "last_knowledge": None,
    "last_retrain": None,
    "last_error": None,
}
AUTOMATION_LOCK = threading.Lock()


def _market_code(symbol: str) -> str:
    symbol = str(symbol).zfill(6)
    if symbol.startswith(("4", "8")):
        return f"bj{symbol}"
    return f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"


def _tencent_code(symbol: str) -> str:
    return _market_code(symbol)


def _normalize_symbol(value: object) -> str:
    match = re.search(r"(\d{6})", str(value or ""))
    if not match:
        raise ValueError("请输入 6 位 A 股股票代码，例如 000001 或 600519")
    symbol = match.group(1)
    if not symbol.startswith(("0", "3", "4", "6", "8")):
        raise ValueError("仅支持 A 股代码，通常以 0、3、4、6、8 开头")
    return symbol


def _now_text() -> str:
    return datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _price_bucket(price: float) -> str:
    if price < 10:
        return "0-10"
    if price < 100:
        return "10-100"
    return "100+"


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    safe = df.replace([float("inf"), float("-inf")], pd.NA).astype(object)
    safe = safe.where(pd.notnull(safe), None)
    return safe.to_dict(orient="records")


def _num(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-"):
            return default
        return float(value)
    except Exception:
        return default


def _read_history_csv(cfg: AppConfig, symbol: str) -> pd.DataFrame:
    path = cfg.data.raw_dir / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return normalize_history(pd.read_csv(path), symbol=symbol)


def _load_metrics(cfg: AppConfig) -> dict[str, object]:
    path = cfg.model.model_dir / "metrics.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _load_selected_universe(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.project_root / "data" / "processed" / "selected_liquid_universe.csv"
    if path.exists():
        df = pd.read_csv(path, dtype={"symbol": str})
    else:
        rows = []
        for file in sorted(cfg.data.raw_dir.glob("*.csv")):
            symbol = file.stem[:6]
            if symbol.isdigit():
                rows.append({"symbol": symbol, "name": symbol})
        df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("没有可用股票池，请先运行 scripts/run_real_ashare_pipeline.py")
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    return df.drop_duplicates("symbol").reset_index(drop=True)


def _load_all_a_universe(cfg: AppConfig) -> pd.DataFrame:
    cache_path = cfg.project_root / "data" / "processed" / "all_a_code_name.csv"
    try:
        import akshare as ak  # type: ignore

        df = ak.stock_info_a_code_name().rename(columns={"code": "symbol"})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    except Exception:
        if not cache_path.exists():
            raise
        df = pd.read_csv(cache_path, dtype={"symbol": str})

    df["symbol"] = df["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    df["name"] = df.get("name", df["symbol"]).astype(str)
    df = df[df["symbol"].str.startswith(("0", "3", "4", "6", "8"))]
    name = df["name"].fillna("")
    df = df[~name.str.contains("ST|退", case=False, regex=True)]
    return df.drop_duplicates("symbol").reset_index(drop=True)


def _candidate_universe(cfg: AppConfig, mode: str) -> pd.DataFrame:
    selected = _load_selected_universe(cfg)[["symbol", "name"]].drop_duplicates("symbol")
    if mode == "cached":
        cached = []
        for file in sorted(cfg.data.raw_dir.glob("*.csv")):
            symbol = file.stem[:6]
            if symbol.isdigit():
                cached.append({"symbol": symbol, "name": symbol})
        cached_df = pd.DataFrame(cached)
        if cached_df.empty:
            return selected
        out = pd.concat([selected, cached_df], ignore_index=True)
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
        return out.drop_duplicates("symbol").reset_index(drop=True)

    all_a = _load_all_a_universe(cfg)
    limit_map = {"sample300": 300, "sample1000": 1000, "all": None}
    limit = limit_map.get(mode, 300)
    if limit is None or len(all_a) <= limit:
        sampled = all_a
    else:
        selected_symbols = set(selected["symbol"])
        rest = all_a[~all_a["symbol"].isin(selected_symbols)]
        need = max(0, limit - len(selected))
        sampled_rest = rest.sample(n=min(need, len(rest)), random_state=42)
        sampled = pd.concat([selected, sampled_rest], ignore_index=True)
    sampled["symbol"] = sampled["symbol"].astype(str).str.zfill(6)
    return sampled.drop_duplicates("symbol").reset_index(drop=True)


def _symbol_universe(cfg: AppConfig, symbol: str) -> pd.DataFrame:
    name = symbol
    for loader in (_load_selected_universe, _load_all_a_universe):
        try:
            universe = loader(cfg)
            matched = universe[universe["symbol"].astype(str).str.zfill(6) == symbol]
            if not matched.empty:
                name = str(matched.iloc[0].get("name") or symbol)
                break
        except Exception:
            continue
    return pd.DataFrame([{"symbol": symbol, "name": name}])


def _news_universe(news_payload: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for stock in news_payload.get("stocks", []) if isinstance(news_payload.get("stocks"), list) else []:
        if not isinstance(stock, dict):
            continue
        try:
            symbol = _normalize_symbol(stock.get("symbol"))
        except ValueError:
            continue
        rows.append({"symbol": symbol, "name": str(stock.get("name") or symbol)})
    if not rows:
        raise RuntimeError("近期新闻没有匹配到可分析的 A 股候选")
    return pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)


def _news_signal_map(news_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    signals: dict[str, dict[str, object]] = {}
    for stock in news_payload.get("stocks", []) if isinstance(news_payload.get("stocks"), list) else []:
        if not isinstance(stock, dict):
            continue
        try:
            symbol = _normalize_symbol(stock.get("symbol"))
        except ValueError:
            continue
        signals[symbol] = stock
    return signals


def _append_review_dates(df: pd.DataFrame, holding_days: int | None) -> pd.DataFrame:
    out = df.copy()
    if holding_days is None or holding_days <= 0:
        out["planned_review_time"] = ""
        return out
    signal_dates = pd.to_datetime(out["signal_date"])
    out["planned_review_time"] = (signal_dates + BDay(holding_days)).dt.date.astype(str)
    return out


def _best_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    action_rank = {"BUY": 0, "WATCH": 1, "AVOID": 2}
    ranked = df.copy()
    ranked["_rank"] = ranked["action"].map(action_rank).fillna(9)
    return (
        ranked.sort_values(["price_bucket", "_rank", "score"], ascending=[True, True, False])
        .groupby("price_bucket", dropna=False)
        .head(5)
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )


def _load_or_fetch_history(cfg: AppConfig, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    if force_refresh:
        try:
            hist = fetch_history_with_akshare(symbol, cfg)
            save_raw_history(hist, cfg.data.raw_dir, symbol)
            return hist
        except Exception:
            local = read_local_history(cfg.data.raw_dir, symbol)
            if local is not None and len(local) >= cfg.model.min_rows_per_symbol:
                return local
            raise

    local = read_local_history(cfg.data.raw_dir, symbol)
    if local is not None and len(local) >= cfg.model.min_rows_per_symbol:
        return local
    hist = fetch_history_with_akshare(symbol, cfg)
    save_raw_history(hist, cfg.data.raw_dir, symbol)
    return hist


def _fetch_history_task(cfg: AppConfig, row: dict[str, object], force_refresh: bool) -> tuple[str, str, pd.DataFrame | None, str | None]:
    symbol = str(row["symbol"]).zfill(6)
    name = str(row.get("name") or symbol)
    try:
        hist = _load_or_fetch_history(cfg, symbol, force_refresh=force_refresh)
        hist["name"] = name
        return symbol, name, hist, None
    except Exception as exc:
        return symbol, name, None, repr(exc)


def _set_job(job_id: str, **updates: object) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = _now_text()


def _get_job(job_id: str) -> dict[str, object] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job is not None else None


def _job_cancel_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        return bool(JOBS.get(job_id, {}).get("cancel_requested"))


def _cancel_job(job_id: str) -> dict[str, object] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        if job.get("status") in {"done", "error", "cancelled"}:
            return dict(job)
        job["cancel_requested"] = True
        job["stage"] = "正在停止"
        job["message"] = "已收到停止请求，正在结束当前批次"
        job["updated_at"] = _now_text()
        return dict(job)


def _automation_update(**updates: object) -> None:
    with AUTOMATION_LOCK:
        AUTOMATION.update(updates)
        AUTOMATION["updated_at"] = _now_text()


def _automation_snapshot(cfg: AppConfig | None = None) -> dict[str, object]:
    with AUTOMATION_LOCK:
        snapshot = dict(AUTOMATION)
    if cfg is not None and snapshot.get("last_retrain") is None:
        try:
            snapshot["last_retrain"] = last_daily_status(cfg)
        except Exception:
            pass
    return snapshot


def _run_news_once(cfg: AppConfig, refresh: bool = True) -> dict[str, object]:
    _automation_update(news_running=True, last_error=None)
    try:
        payload = analyze_news_impact(cfg, limit=12, refresh=refresh)
        _automation_update(news_running=False, last_news=payload)
        return payload
    except Exception as exc:
        _automation_update(news_running=False, last_error=f"news: {exc!r}")
        raise


def _run_knowledge_once(cfg: AppConfig, refresh: bool = True) -> dict[str, object]:
    _automation_update(knowledge_running=True, last_error=None)
    try:
        payload = learn_kline_knowledge(cfg, refresh=refresh)
        _automation_update(knowledge_running=False, last_knowledge=payload)
        return payload
    except Exception as exc:
        _automation_update(knowledge_running=False, last_error=f"knowledge: {exc!r}")
        raise


def _run_retrain_once(cfg: AppConfig) -> dict[str, object]:
    _automation_update(retrain_running=True, last_error=None)

    def progress(update: dict[str, object]) -> None:
        _automation_update(retrain_progress=update)

    try:
        limit_value = os.environ.get("ASHARE_RETRAIN_SYMBOL_LIMIT", "").strip()
        limit = int(limit_value) if limit_value else None
        payload = run_daily_update_and_train(cfg, max_workers=FETCH_WORKERS, limit=limit, progress=progress)
        try:
            payload["evaluation"] = evaluate_saved_predictions(cfg)
        except Exception as eval_exc:
            payload["evaluation_error"] = repr(eval_exc)
        _automation_update(retrain_running=False, last_retrain=payload, retrain_progress={"progress": 100, "stage": "完成"})
        return payload
    except Exception as exc:
        _automation_update(retrain_running=False, last_error=f"retrain: {exc!r}")
        raise


def _background_loop(cfg: AppConfig) -> None:
    news_interval = max(300, int(os.environ.get("ASHARE_NEWS_INTERVAL_SECONDS", "900")))
    knowledge_interval = max(21_600, int(os.environ.get("ASHARE_KNOWLEDGE_INTERVAL_SECONDS", "86400")))
    retrain_time = os.environ.get("ASHARE_RETRAIN_TIME", "16:30")
    last_news_ts = 0.0
    last_knowledge_ts = 0.0
    last_retrain_date = ""
    while True:
        now = datetime.now(SHANGHAI_TZ)
        snapshot = _automation_snapshot()
        if snapshot.get("auto_knowledge_enabled") and time.time() - last_knowledge_ts >= knowledge_interval:
            try:
                _run_knowledge_once(cfg, refresh=True)
            except Exception:
                pass
            last_knowledge_ts = time.time()

        if snapshot.get("auto_news_enabled") and time.time() - last_news_ts >= news_interval:
            try:
                _run_news_once(cfg, refresh=True)
            except Exception:
                pass
            last_news_ts = time.time()

        if snapshot.get("auto_retrain_enabled") and now.strftime("%H:%M") >= retrain_time and last_retrain_date != now.date().isoformat():
            try:
                _run_retrain_once(cfg)
                last_retrain_date = now.date().isoformat()
            except Exception:
                last_retrain_date = now.date().isoformat()
        time.sleep(30)


def _start_background_tasks(cfg: AppConfig) -> None:
    with AUTOMATION_LOCK:
        if AUTOMATION.get("started"):
            return
        AUTOMATION["started"] = True
        AUTOMATION["auto_news_enabled"] = os.environ.get("ASHARE_AUTO_NEWS", "0") == "1"
        AUTOMATION["auto_knowledge_enabled"] = os.environ.get("ASHARE_AUTO_KNOWLEDGE", "0") == "1"
        AUTOMATION["auto_retrain_enabled"] = os.environ.get("ASHARE_AUTO_RETRAIN", "0") == "1"
        AUTOMATION["news_interval_seconds"] = int(os.environ.get("ASHARE_NEWS_INTERVAL_SECONDS", "900"))
        AUTOMATION["knowledge_interval_seconds"] = int(os.environ.get("ASHARE_KNOWLEDGE_INTERVAL_SECONDS", "86400"))
        AUTOMATION["retrain_time"] = os.environ.get("ASHARE_RETRAIN_TIME", "16:30")
        AUTOMATION["fetch_workers"] = FETCH_WORKERS
    thread = threading.Thread(target=_background_loop, args=(cfg,), daemon=True)
    thread.start()


def _run_analysis_job(job_id: str, cfg: AppConfig, payload: dict[str, object]) -> None:
    started = time.time()
    try:
        if _job_cancel_requested(job_id):
            _set_job(job_id, status="cancelled", stage="已停止", progress=100, message="任务已停止")
            return
        capital = float(payload.get("capital") or cfg.risk.capital)
        raw_holding = payload.get("holding_days")
        holding_days = int(raw_holding) if raw_holding not in (None, "", 0, "0") else None
        mode = str(payload.get("universe_mode") or "sample300")
        target_symbol = payload.get("symbol")
        news_payload: dict[str, object] | None = None
        news_signals: dict[str, dict[str, object]] | None = None

        _set_job(job_id, status="running", stage="准备股票池", progress=2, message="正在读取候选股票池")
        if target_symbol:
            symbol = _normalize_symbol(target_symbol)
            universe = _symbol_universe(cfg, symbol)
            mode = "single"
            force_refresh = True
            _set_job(job_id, stage="准备指定股票", progress=4, message=f"准备在线抓取 {symbol} 历史行情")
        elif mode == "news":
            _set_job(job_id, stage="综合新闻", progress=4, message="正在抓取并聚合近期国际新闻主题")
            news_payload = analyze_news_impact(cfg, limit=30, refresh=bool(payload.get("refresh_news", True)))
            universe = _news_universe(news_payload)
            news_signals = _news_signal_map(news_payload)
            force_refresh = False
        else:
            universe = _candidate_universe(cfg, mode)
            force_refresh = False
        total = int(len(universe))
        if total == 0:
            raise RuntimeError("候选股票池为空")
        if _job_cancel_requested(job_id):
            _set_job(job_id, status="cancelled", stage="已停止", progress=100, message="任务已停止")
            return

        frames: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        rows = universe.to_dict(orient="records")
        worker_count = 1 if mode == "single" else min(FETCH_WORKERS, max(total, 1))
        _set_job(
            job_id,
            stage="读取历史行情",
            progress=5,
            processed=0,
            total=total,
            message=f"并发读取历史行情，线程数 {worker_count}",
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending_rows = iter(rows)
            futures = set()
            completed = 0
            for _ in range(worker_count):
                try:
                    futures.add(executor.submit(_fetch_history_task, cfg, next(pending_rows), force_refresh))
                except StopIteration:
                    break

            while futures:
                if _job_cancel_requested(job_id):
                    for future in futures:
                        future.cancel()
                    _set_job(
                        job_id,
                        status="cancelled",
                        stage="已停止",
                        progress=100,
                        processed=completed,
                        total=total,
                        message=f"任务已停止，已处理 {completed}/{total}",
                        elapsed_seconds=round(time.time() - started, 1),
                    )
                    return
                done_future = next(as_completed(futures))
                futures.remove(done_future)
                completed += 1
                future = done_future
                symbol, name, hist, error = future.result()
                if hist is None:
                    failures.append({"symbol": symbol, "name": name, "error": str(error)})
                else:
                    frames.append(hist)
                progress = 5 + int(completed / total * 70)
                _set_job(
                    job_id,
                    stage="读取历史行情",
                    progress=progress,
                    processed=completed,
                    total=total,
                    message=f"{symbol} {name}",
                )
                if not _job_cancel_requested(job_id):
                    try:
                        futures.add(executor.submit(_fetch_history_task, cfg, next(pending_rows), force_refresh))
                    except StopIteration:
                        pass

        if not frames:
            raise RuntimeError("没有成功读取任何历史行情")
        if _job_cancel_requested(job_id):
            _set_job(job_id, status="cancelled", stage="已停止", progress=100, message="任务已停止")
            return

        _set_job(job_id, stage="模型评分", progress=82, message=f"正在对 {len(frames)} 只股票生成建议")
        history = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
        symbols = sorted(history["symbol"].unique().tolist())
        rec = make_recommendations(
            history.drop(columns=["name"], errors="ignore"),
            cfg,
            symbols=symbols,
            capital=capital,
            holding_days=holding_days,
            news_signals=news_signals,
        )
        meta = universe[["symbol", "name"]].drop_duplicates("symbol")
        rec = rec.merge(meta, on="symbol", how="left")
        if "price_bucket" not in rec.columns:
            rec["price_bucket"] = rec["reference_entry"].astype(float).map(_price_bucket)
        rec = add_price_bucket(rec.rename(columns={"reference_entry": "price"}), price_col="price").rename(
            columns={"price": "reference_entry"}
        )
        rec = _append_review_dates(rec, holding_days)
        best = _best_by_bucket(rec)

        reports_dir = cfg.output.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        rec.to_csv(reports_dir / "web_dynamic_recommendations.csv", index=False, encoding="utf-8-sig")
        best.to_csv(reports_dir / "web_dynamic_best_by_bucket.csv", index=False, encoding="utf-8-sig")
        evaluation_summary: dict[str, object] | None = None
        try:
            append_recommendation_snapshot(
                cfg,
                rec,
                run_id=job_id,
                source="web_analysis",
                capital=capital,
                holding_days=holding_days,
                universe_mode=mode,
            )
            evaluation_summary = evaluate_saved_predictions(cfg)
        except Exception as eval_exc:
            evaluation_summary = {"error": repr(eval_exc)}

        _set_job(
            job_id,
            status="done",
            stage="完成",
            progress=100,
            message=f"完成 {len(symbols)} 只股票评分，失败 {len(failures)} 只",
            generated_at=_now_text(),
            elapsed_seconds=round(time.time() - started, 1),
            capital=capital,
            holding_days=holding_days,
            universe_mode=mode,
            total=total,
            processed=total,
            failures=failures[:30],
            result={"best": _records(best), "all": _records(rec), "news": news_payload, "evaluation": evaluation_summary},
        )
    except Exception as exc:
        _set_job(job_id, status="error", stage="失败", progress=100, message=str(exc), error=repr(exc))


def _parse_sina_quote(text: str, symbol: str) -> dict[str, object] | None:
    match = re.search(r'="(.*)"', text)
    if not match:
        return None
    parts = match.group(1).split(",")
    if len(parts) < 32 or not parts[0]:
        return None
    current = float(parts[3] or 0)
    prev_close = float(parts[2] or 0)
    change = current - prev_close if current and prev_close else 0.0
    pct = change / prev_close * 100 if prev_close else 0.0
    return {
        "symbol": symbol,
        "name": parts[0],
        "open": float(parts[1] or 0),
        "prev_close": prev_close,
        "price": current,
        "high": float(parts[4] or 0),
        "low": float(parts[5] or 0),
        "volume": float(parts[8] or 0),
        "amount": float(parts[9] or 0),
        "date": parts[30],
        "time": parts[31],
        "change": change,
        "pct_chg": pct,
        "source": "sina_realtime",
    }


def _sina_quote(symbol: str) -> dict[str, object] | None:
    code = _market_code(symbol)
    url = f"https://hq.sinajs.cn/list={code}"
    response = requests.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=8)
    response.raise_for_status()
    return _parse_sina_quote(response.text, symbol)


def _eastmoney_secid(symbol: str) -> str:
    symbol = str(symbol).zfill(6)
    market = "1" if symbol.startswith("6") else "0"
    return f"{market}.{symbol}"


def _em_scaled(data: dict[str, object], field: str, scale: float = 100.0) -> float:
    value = data.get(field)
    if value in (None, "", "-"):
        return 0.0
    return float(value) / scale


def _eastmoney_quote(symbol: str) -> dict[str, object] | None:
    secid = _eastmoney_secid(symbol)
    fields = ",".join(
        [
            "f43",
            "f44",
            "f45",
            "f46",
            "f47",
            "f48",
            "f57",
            "f58",
            "f60",
            "f86",
            "f116",
            "f117",
            "f168",
            "f170",
            "f171",
        ]
    )
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields={fields}"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
        timeout=8,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    price = _em_scaled(data, "f43")
    prev_close = _em_scaled(data, "f60")
    change = _em_scaled(data, "f171") if data.get("f171") not in (None, "", "-") else price - prev_close
    pct = _em_scaled(data, "f170") if data.get("f170") not in (None, "", "-") else ((price / prev_close - 1) * 100 if prev_close else 0.0)
    tick_time = str(data.get("f86") or "")
    date = tick_time[:8] if len(tick_time) >= 8 else ""
    time_text = tick_time[8:14] if len(tick_time) >= 14 else ""
    date_text = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else ""
    time_display = f"{time_text[:2]}:{time_text[2:4]}:{time_text[4:6]}" if len(time_text) == 6 else ""
    return {
        "symbol": symbol,
        "name": data.get("f58") or symbol,
        "open": _em_scaled(data, "f46"),
        "prev_close": prev_close,
        "price": price,
        "high": _em_scaled(data, "f44"),
        "low": _em_scaled(data, "f45"),
        "volume": float(data.get("f47") or 0),
        "amount": float(data.get("f48") or 0),
        "turnover": _em_scaled(data, "f168"),
        "total_market_cap": float(data.get("f116") or 0),
        "float_market_cap": float(data.get("f117") or 0),
        "date": date_text,
        "time": time_display,
        "change": change,
        "pct_chg": pct,
        "source": "eastmoney_realtime",
    }


def _parse_tencent_quote(text: str, symbol: str) -> dict[str, object] | None:
    match = re.search(r'="(.*)"', text)
    if not match:
        return None
    parts = match.group(1).split("~")
    if len(parts) < 40:
        return None
    price = _num(parts[3])
    prev_close = _num(parts[4])
    open_price = _num(parts[5])
    tick_time = parts[30] if len(parts) > 30 else ""
    date_text = f"{tick_time[:4]}-{tick_time[4:6]}-{tick_time[6:8]}" if len(tick_time) >= 8 else ""
    time_text = f"{tick_time[8:10]}:{tick_time[10:12]}:{tick_time[12:14]}" if len(tick_time) >= 14 else ""
    return {
        "symbol": symbol,
        "name": parts[1] or symbol,
        "open": open_price,
        "prev_close": prev_close,
        "price": price,
        "high": _num(parts[33]),
        "low": _num(parts[34]),
        "volume": _num(parts[36]) * 100,
        "amount": _num(parts[37]) * 10000,
        "turnover": _num(parts[38]),
        "total_market_cap": _num(parts[45]) * 100000000 if len(parts) > 45 else 0.0,
        "float_market_cap": _num(parts[44]) * 100000000 if len(parts) > 44 else 0.0,
        "date": date_text,
        "time": time_text,
        "change": _num(parts[31], price - prev_close),
        "pct_chg": _num(parts[32], (price / prev_close - 1) * 100 if prev_close else 0.0),
        "source": "tencent_realtime",
    }


def _tencent_quote(symbol: str) -> dict[str, object] | None:
    code = _tencent_code(symbol)
    response = requests.get(
        f"http://qt.gtimg.cn/q={code}",
        headers={"User-Agent": "Mozilla/5.0", "Referer": "http://finance.qq.com/"},
        timeout=8,
    )
    response.raise_for_status()
    response.encoding = "gbk"
    return _parse_tencent_quote(response.text, symbol)


def _fallback_quote(cfg: AppConfig, symbol: str) -> dict[str, object]:
    hist = _read_history_csv(cfg, symbol).sort_values("date")
    row = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else row
    price = float(row["close"])
    prev_close = float(prev["close"])
    return {
        "symbol": symbol,
        "name": symbol,
        "open": float(row["open"]),
        "prev_close": prev_close,
        "price": price,
        "high": float(row["high"]),
        "low": float(row["low"]),
        "volume": float(row["volume"]),
        "amount": float(row.get("amount", 0.0)),
        "date": pd.Timestamp(row["date"]).date().isoformat(),
        "time": "15:00:00",
        "change": price - prev_close,
        "pct_chg": (price / prev_close - 1) * 100 if prev_close else 0.0,
        "source": "cached_daily",
    }


def _parse_tencent_minute_rows(raw_rows: list[str], date_text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    last_volume = 0.0
    last_amount = 0.0
    for raw in raw_rows:
        parts = str(raw).split()
        if len(parts) < 4:
            continue
        hhmm = parts[0].zfill(4)
        price = _num(parts[1])
        cum_volume = _num(parts[2]) * 100
        cum_amount = _num(parts[3])
        volume = max(0.0, cum_volume - last_volume)
        amount = max(0.0, cum_amount - last_amount)
        last_volume = cum_volume
        last_amount = cum_amount
        avg_price = cum_amount / cum_volume if cum_volume > 0 else price
        rows.append(
            {
                "datetime": f"{date_text} {hhmm[:2]}:{hhmm[2:]}:00",
                "date": date_text,
                "time": f"{hhmm[:2]}:{hhmm[2:]}",
                "price": round(price, 4),
                "avg_price": round(avg_price, 4),
                "volume": round(volume, 0),
                "amount": round(amount, 2),
                "cum_volume": round(cum_volume, 0),
                "cum_amount": round(cum_amount, 2),
            }
        )
    return rows


def _tencent_timeline(symbol: str, range_mode: str) -> dict[str, object]:
    code = _tencent_code(symbol)
    if range_mode == "5d":
        url = f"http://web.ifzq.gtimg.cn/appstock/app/day/query?code={code}"
    else:
        url = f"http://web.ifzq.gtimg.cn/appstock/app/minute/query?code={code}"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "http://gu.qq.com/"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", {}).get(code, {}).get("data", {}) if isinstance(payload, dict) else {}
    rows: list[dict[str, object]] = []
    if range_mode == "5d":
        day_items = data if isinstance(data, list) else data.get("data", [])
        for day in day_items:
            if not isinstance(day, dict):
                continue
            raw_date = str(day.get("date") or "")
            if len(raw_date) != 8:
                continue
            date_text = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            rows.extend(_parse_tencent_minute_rows(day.get("data", []), date_text))
    else:
        raw_date = str(data.get("date") or datetime.now(SHANGHAI_TZ).strftime("%Y%m%d")) if isinstance(data, dict) else ""
        date_text = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else datetime.now(SHANGHAI_TZ).date().isoformat()
        raw_rows = data.get("data", []) if isinstance(data, dict) else []
        rows = _parse_tencent_minute_rows(raw_rows, date_text)
    if not rows:
        raise RuntimeError("没有可用分时数据")
    rows = sorted(rows, key=lambda item: str(item.get("datetime") or ""))
    return {"symbol": symbol, "range": range_mode, "source": "tencent_minute", "rows": rows}


def _fallback_timeline_from_daily(cfg: AppConfig, symbol: str, range_mode: str) -> dict[str, object]:
    hist = _read_history_csv(cfg, symbol).sort_values("date")
    take = 5 if range_mode == "5d" else 1
    rows = []
    for _, row in hist.tail(take).iterrows():
        date_text = pd.Timestamp(row["date"]).date().isoformat()
        rows.append(
            {
                "datetime": f"{date_text} 15:00:00",
                "date": date_text,
                "time": "15:00",
                "price": float(row["close"]),
                "avg_price": float(row["close"]),
                "volume": float(row["volume"]),
                "amount": float(row.get("amount", 0.0)),
                "cum_volume": float(row["volume"]),
                "cum_amount": float(row.get("amount", 0.0)),
            }
        )
    return {"symbol": symbol, "range": range_mode, "source": "cached_daily_fallback", "rows": rows}


def _resample_ohlcv(hist: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "daily":
        return hist
    rule = "W-FRI" if period == "weekly" else "ME"
    work = hist.copy()
    work["date"] = pd.to_datetime(work["date"])
    out = (
        work.set_index("date")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    out["symbol"] = hist["symbol"].iloc[0] if "symbol" in hist.columns and not hist.empty else ""
    out["turnover"] = 0.0
    return out


def create_app(config_path: str | Path | None = None) -> Flask:
    cfg = load_config(config_path or PROJECT_ROOT / "configs" / "default.toml")
    app = Flask(__name__, static_folder=None)
    app.secret_key = os.environ.get("ASHARE_SECRET_KEY", "dev-local-secret-change-me")
    _start_background_tasks(cfg)

    def _password_enabled() -> bool:
        return bool(os.environ.get("ASHARE_WEB_PASSWORD"))

    def _wants_json() -> bool:
        return request.path.startswith("/api/") or "application/json" in request.headers.get("Accept", "")

    @app.before_request
    def require_login():
        if not _password_enabled():
            return None
        if request.path in {"/login", "/healthz"} or request.path.startswith("/assets/"):
            return None
        if session.get("authenticated"):
            return None
        if _wants_json():
            return jsonify({"error": "authentication_required"}), 401
        return redirect(url_for("login"))

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "time": _now_text()})

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if not _password_enabled():
            return redirect("/")
        error = ""
        if request.method == "POST":
            password = request.form.get("password", "")
            if password == os.environ.get("ASHARE_WEB_PASSWORD"):
                session["authenticated"] = True
                return redirect("/")
            error = "密码不正确"
        return (
            """
            <!doctype html>
            <html lang="zh-CN">
              <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>A 股 AI 推荐登录</title>
                <link rel="stylesheet" href="/assets/styles.css" />
              </head>
              <body>
                <main class="login-shell">
                  <form class="login-box" method="post">
                    <h1>A 股 AI 推荐</h1>
                    <p>请输入访问密码</p>
                    <input name="password" type="password" autofocus />
                    <button type="submit">登录</button>
                    <span class="login-error">"""
            + error
            + """</span>
                  </form>
                </main>
              </body>
            </html>
            """
        )

    @app.get("/")
    def index():
        return send_from_directory(WEB_ROOT, "index.html")

    @app.get("/assets/<path:name>")
    def assets(name: str):
        return send_from_directory(WEB_ROOT / "assets", name)

    @app.get("/api/time")
    def api_time():
        now = datetime.now(SHANGHAI_TZ)
        return jsonify({"iso": now.isoformat(timespec="seconds"), "display": _now_text(), "timezone": "Asia/Shanghai"})

    @app.get("/api/status")
    def api_status():
        universe = _load_selected_universe(cfg)
        metrics = _load_metrics(cfg)
        latest_dates = []
        for symbol in universe["symbol"].head(10):
            try:
                hist = _read_history_csv(cfg, symbol)
                latest_dates.append(pd.Timestamp(hist["date"].max()).date().isoformat())
            except Exception:
                pass
        return jsonify(
            {
                "symbols": int(universe["symbol"].nunique()),
                "model_backend": metrics.get("backend"),
                "model_horizon_days": metrics.get("default_horizon_days", cfg.model.horizon_days),
                "available_horizons": metrics.get("available_horizons", [cfg.model.horizon_days]),
                "latest_cached_date": max(latest_dates) if latest_dates else None,
                "metrics": metrics,
            }
        )

    @app.get("/api/news/impact")
    def api_news_impact():
        limit = max(3, min(int(request.args.get("limit", "12")), 30))
        refresh = request.args.get("refresh", "0") in {"1", "true", "yes"}
        return jsonify(analyze_news_impact(cfg, limit=limit, refresh=refresh))

    @app.get("/api/automation/status")
    def api_automation_status():
        return jsonify(_automation_snapshot(cfg))

    @app.get("/api/knowledge/status")
    def api_knowledge_status():
        refresh = request.args.get("refresh", "0") in {"1", "true", "yes"}
        if refresh:
            return jsonify(learn_kline_knowledge(cfg, refresh=True))
        return jsonify(load_kline_knowledge(cfg))

    @app.post("/api/knowledge/learn")
    def api_knowledge_learn():
        thread = threading.Thread(target=_run_knowledge_once, args=(cfg, True), daemon=True)
        thread.start()
        return jsonify({"status": "started"})

    @app.post("/api/automation/news/run")
    def api_automation_news_run():
        thread = threading.Thread(target=_run_news_once, args=(cfg, True), daemon=True)
        thread.start()
        return jsonify({"status": "started"})

    @app.post("/api/automation/retrain/run")
    def api_automation_retrain_run():
        if _automation_snapshot().get("retrain_running"):
            return jsonify({"status": "already_running"})
        thread = threading.Thread(target=_run_retrain_once, args=(cfg,), daemon=True)
        thread.start()
        return jsonify({"status": "started"})

    @app.get("/api/evaluation/status")
    def api_evaluation_status():
        return jsonify(load_evaluation_summary(cfg))

    @app.post("/api/evaluation/run")
    def api_evaluation_run():
        payload = request.get_json(silent=True) or {}
        refresh = bool(payload.get("refresh_history")) or request.args.get("refresh", "0") in {"1", "true", "yes"}
        symbols: list[str] | None = None
        if payload.get("symbol"):
            try:
                symbols = [_normalize_symbol(payload.get("symbol"))]
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        return jsonify(evaluate_saved_predictions(cfg, refresh_history=refresh, symbols=symbols))

    @app.post("/api/analysis/start")
    def api_analysis_start():
        payload = request.get_json(silent=True) or {}
        job_id = uuid.uuid4().hex
        _set_job(job_id, status="queued", stage="排队", progress=0, message="等待开始", created_at=_now_text())
        thread = threading.Thread(target=_run_analysis_job, args=(job_id, cfg, payload), daemon=True)
        thread.start()
        return jsonify({"job_id": job_id})

    @app.post("/api/stock/analyze/start")
    def api_stock_analysis_start():
        payload = request.get_json(silent=True) or {}
        try:
            symbol = _normalize_symbol(payload.get("symbol"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        job_id = uuid.uuid4().hex
        _set_job(job_id, status="queued", stage="排队", progress=0, message="等待开始", created_at=_now_text())
        thread = threading.Thread(
            target=_run_analysis_job,
            args=(job_id, cfg, payload | {"symbol": symbol, "universe_mode": "single"}),
            daemon=True,
        )
        thread.start()
        return jsonify({"job_id": job_id, "symbol": symbol})

    @app.get("/api/analysis/<job_id>")
    def api_analysis_status(job_id: str):
        job = _get_job(job_id)
        if job is None:
            return jsonify({"status": "missing", "message": "任务不存在"}), 404
        return jsonify(job)

    @app.post("/api/analysis/<job_id>/cancel")
    def api_analysis_cancel(job_id: str):
        job = _cancel_job(job_id)
        if job is None:
            return jsonify({"status": "missing", "message": "任务不存在"}), 404
        return jsonify({"status": job.get("status"), "message": job.get("message")})

    @app.post("/api/recommendations")
    def api_recommendations():
        payload = request.get_json(silent=True) or {}
        job_id = uuid.uuid4().hex
        _set_job(job_id, status="queued", stage="排队", progress=0, message="等待开始", created_at=_now_text())
        _run_analysis_job(job_id, cfg, payload | {"universe_mode": payload.get("universe_mode") or "cached"})
        job = _get_job(job_id) or {}
        if job.get("status") == "error":
            return jsonify(job), 500
        result = job.get("result", {})
        return jsonify(
            {
                "generated_at": job.get("generated_at"),
                "capital": job.get("capital"),
                "holding_days": job.get("holding_days"),
                "model_horizon_days": cfg.model.horizon_days,
                "best": result.get("best", []) if isinstance(result, dict) else [],
                "all": result.get("all", []) if isinstance(result, dict) else [],
            }
        )

    @app.get("/api/quote/<symbol>")
    def api_quote(symbol: str):
        symbol = str(symbol).zfill(6)
        try:
            quote = _eastmoney_quote(symbol)
            if quote is None or not float(quote.get("price") or 0):
                quote = _tencent_quote(symbol)
            if quote is None or not float(quote.get("price") or 0):
                quote = _sina_quote(symbol)
            if quote is None or not float(quote.get("price") or 0):
                quote = _fallback_quote(cfg, symbol)
        except Exception:
            try:
                quote = _tencent_quote(symbol)
                if quote is None or not float(quote.get("price") or 0):
                    quote = _sina_quote(symbol)
                if quote is None or not float(quote.get("price") or 0):
                    quote = _fallback_quote(cfg, symbol)
            except Exception:
                quote = _fallback_quote(cfg, symbol)
        quote["server_time"] = _now_text()
        return jsonify(quote)

    @app.get("/api/kline/<symbol>")
    def api_kline(symbol: str):
        symbol = str(symbol).zfill(6)
        limit = int(request.args.get("limit", "160"))
        forecast_days = max(0, min(int(request.args.get("forecast_days", "8")), 20))
        period = request.args.get("period", "daily")
        if period not in {"daily", "weekly", "monthly"}:
            period = "daily"
        full_hist = _read_history_csv(cfg, symbol).sort_values("date")
        hist = _resample_ohlcv(full_hist, period).tail(max(20, min(limit, 420)))
        rows = []
        for _, row in hist.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
        forecast_payload = None
        if period == "daily" and forecast_days > 0:
            try:
                forecast_payload = forecast_kline(full_hist, cfg, symbol=symbol, days=forecast_days)
                latest = full_hist.sort_values("date").iloc[-1]
                append_kline_forecast_snapshot(
                    cfg,
                    forecast_payload,
                    base_date=latest["date"],
                    base_close=latest["close"],
                    source="kline_api",
                )
            except Exception as exc:
                forecast_payload = {"error": str(exc), "rows": []}
        return jsonify(
            {
                "symbol": symbol,
                "period": period,
                "rows": rows,
                "forecast": forecast_payload.get("rows", []) if isinstance(forecast_payload, dict) else [],
                "forecast_summary": forecast_payload,
            }
        )

    @app.get("/api/timeline/<symbol>")
    def api_timeline(symbol: str):
        symbol = str(symbol).zfill(6)
        range_mode = request.args.get("range", "1d")
        if range_mode not in {"1d", "5d"}:
            range_mode = "1d"
        try:
            payload = _tencent_timeline(symbol, range_mode)
        except Exception:
            payload = _fallback_timeline_from_daily(cfg, symbol, range_mode)
        try:
            payload["quote"] = _tencent_quote(symbol) or _fallback_quote(cfg, symbol)
        except Exception:
            payload["quote"] = _fallback_quote(cfg, symbol)
        payload["server_time"] = _now_text()
        return jsonify(payload)

    return app


def main() -> None:
    app = create_app()
    host = os.environ.get("ASHARE_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("ASHARE_PORT", "7860")))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
