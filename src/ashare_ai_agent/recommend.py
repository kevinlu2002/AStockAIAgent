from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

from .config import AppConfig
from .features import FEATURE_COLUMNS, build_feature_table, coerce_model_features
from .knowledge import load_kline_knowledge
from .models import load_model, predict_positive_probability
from .portfolio import plan_position


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("|", "/") for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _holding_context(horizon: int, holding_days: int | None) -> tuple[int, float]:
    days = int(holding_days) if holding_days and holding_days > 0 else horizon
    scale = max(0.2, min(6.0, days / max(horizon, 1)))
    return days, scale


def _action(row: pd.Series, cfg: AppConfig, horizon: int, holding_days: int | None) -> str:
    _, scale = _holding_context(horizon, holding_days)
    min_return = cfg.risk.min_expected_return * max(0.35, min(3.0, scale))
    max_drawdown = max(-0.35, cfg.risk.max_expected_drawdown * math.sqrt(scale))
    direction_prob = float(row.get("direction_probability", 0.5) or 0.5)
    trend_prob = float(row.get("trend_probability", 0.5) or 0.5)
    min_direction_prob = float(row.get("min_direction_probability", 0.58) or 0.58)
    min_trend_prob = float(row.get("min_trend_probability", 0.58) or 0.58)
    if (
        row["pred_return"] >= min_return
        and row["pred_drawdown"] >= max_drawdown
        and row["score"] > 0.35
        and direction_prob >= min_direction_prob
        and trend_prob >= min_trend_prob
    ):
        return "BUY"
    if row["pred_return"] > 0 and row["score"] > 0 and direction_prob >= 0.50:
        return "WATCH"
    return "AVOID"


def _reason(row: pd.Series) -> str:
    parts: list[str] = []
    parts.append("预测收益为正" if row["pred_return"] > 0 else "预测收益不占优")
    parts.append(f"10日涨跌概率{float(row.get('direction_probability', 0.5) or 0.5) * 100:.1f}%")
    parts.append(f"趋势概率{float(row.get('trend_probability', 0.5) or 0.5) * 100:.1f}%")
    parts.append("价格高于20日均线" if row["close_to_sma_20"] > 0 else "价格低于20日均线")
    parts.append("预测回撤适中" if row["pred_drawdown"] > -0.08 else "预测回撤风险偏大")
    signals = _technical_signal_labels(row)
    if signals:
        parts.append("K线信号：" + "、".join(signals[:3]))
    if row.get("knowledge_themes"):
        parts.append("活知识：" + str(row.get("knowledge_themes")))
    return "；".join(parts)


def _adjust_prediction_columns(latest: pd.DataFrame, horizon: int, holding_days: int | None) -> pd.DataFrame:
    _, scale = _holding_context(horizon, holding_days)
    out = latest.copy()
    raw_return = out["model_pred_return"].clip(lower=-0.95, upper=2.0)
    out["pred_return"] = (1 + raw_return).pow(scale) - 1
    raw_drawdown = out["model_pred_drawdown"].clip(upper=-0.001, lower=-0.5)
    out["pred_drawdown"] = (raw_drawdown * math.sqrt(scale)).clip(lower=-0.5, upper=-0.001)
    out["base_score"] = out["pred_return"] / out["pred_drawdown"].abs().clip(lower=0.01)
    out["direction_confidence"] = (out["direction_probability"] - 0.5).abs() * 2.0
    out["trend_confidence"] = (out["trend_probability"] - 0.5).abs() * 2.0
    return out


def _technical_tilt(row: pd.Series, horizon: int, holding_days: int | None) -> float:
    days, _ = _holding_context(horizon, holding_days)
    volume_z = max(-3.0, min(3.0, float(row.get("volume_z_20", 0.0) or 0.0)))
    short_signal = (
        4.0 * float(row.get("ret_3", 0.0) or 0.0)
        + 3.0 * float(row.get("ret_5", 0.0) or 0.0)
        + 8.0 * float(row.get("macd_hist", 0.0) or 0.0)
        + 0.05 * volume_z
    )
    swing_signal = (
        2.2 * float(row.get("ret_20", 0.0) or 0.0)
        + 1.4 * float(row.get("close_to_sma_20", 0.0) or 0.0)
        + 1.2 * float(row.get("sma_5_to_20", 0.0) or 0.0)
        - 3.0 * float(row.get("volatility_20", 0.0) or 0.0)
    )
    long_signal = (
        1.2 * float(row.get("ret_60", 0.0) or 0.0)
        + 2.0 * float(row.get("sma_20_to_60", 0.0) or 0.0)
        + 0.8 * float(row.get("close_to_sma_60", 0.0) or 0.0)
        - 4.0 * float(row.get("volatility_20", 0.0) or 0.0)
    )
    kline_signal = _kline_signal_score(row)
    if days <= 3:
        tilt = short_signal
    elif days <= 10:
        tilt = 0.55 * short_signal + 0.45 * swing_signal
    elif days <= 25:
        tilt = 0.25 * short_signal + 0.60 * swing_signal + 0.15 * long_signal
    else:
        tilt = 0.15 * short_signal + 0.35 * swing_signal + 0.50 * long_signal
    tilt += 0.18 * kline_signal
    return max(-0.8, min(0.8, tilt))


def _kline_signal_score(row: pd.Series) -> float:
    positive = (
        0.35 * float(row.get("ma_bull_stack", 0.0) or 0.0)
        + 0.30 * float(row.get("golden_cross_5_20", 0.0) or 0.0)
        + 0.35 * float(row.get("breakout_20", 0.0) or 0.0)
        + 0.25 * float(row.get("volume_price_confirm", 0.0) or 0.0)
        + 0.20 * float(row.get("hammer", 0.0) or 0.0)
        + 0.25 * float(row.get("bullish_engulfing", 0.0) or 0.0)
    )
    negative = (
        0.35 * float(row.get("ma_bear_stack", 0.0) or 0.0)
        + 0.30 * float(row.get("dead_cross_5_20", 0.0) or 0.0)
        + 0.35 * float(row.get("breakdown_20", 0.0) or 0.0)
        + 0.20 * float(row.get("shooting_star", 0.0) or 0.0)
        + 0.25 * float(row.get("bearish_engulfing", 0.0) or 0.0)
    )
    return max(-1.0, min(1.0, positive - negative))


def _technical_signal_labels(row: pd.Series) -> list[str]:
    pairs = [
        ("ma_bull_stack", "均线多头排列"),
        ("golden_cross_5_20", "5日线上穿20日线"),
        ("breakout_20", "放量突破20日高点"),
        ("volume_price_confirm", "价涨量增"),
        ("hammer", "锤头线"),
        ("bullish_engulfing", "阳包阴"),
        ("limit_up_close", "涨停封住"),
        ("failed_limit_up", "烂板/炸板"),
        ("ma_bear_stack", "均线空头排列"),
        ("dead_cross_5_20", "5日线下穿20日线"),
        ("breakdown_20", "放量跌破20日低点"),
        ("shooting_star", "长上影试压"),
        ("bearish_engulfing", "阴包阳"),
    ]
    return [label for key, label in pairs if float(row.get(key, 0.0) or 0.0) > 0.5]


def _limit_up_state(row: pd.Series) -> str:
    if float(row.get("limit_up_close", 0.0) or 0.0) > 0.5:
        return "涨停封住"
    if float(row.get("failed_limit_up", 0.0) or 0.0) > 0.5:
        return "烂板/炸板"
    if float(row.get("limit_up_touch", 0.0) or 0.0) > 0.5:
        return "触板未封"
    return "--"


def _knowledge_signal(row: pd.Series, weights: dict[str, float]) -> tuple[float, list[str]]:
    labels: list[str] = []
    score = 0.0
    volume_z = max(-3.0, min(3.0, float(row.get("volume_z_20", 0.0) or 0.0)))
    ret_20 = float(row.get("ret_20", 0.0) or 0.0)
    ret_60 = float(row.get("ret_60", 0.0) or 0.0)
    close_to_high = float(row.get("close_to_high_20", -1.0) or -1.0)
    volatility = float(row.get("volatility_20", 0.0) or 0.0)
    news_score = max(0.0, float(row.get("news_score", 0.0) or 0.0))
    themes_text = str(row.get("news_themes_text", "") or "").lower()
    close_to_sma_20 = float(row.get("close_to_sma_20", 0.0) or 0.0)
    failed_limit_up = float(row.get("failed_limit_up", 0.0) or 0.0) > 0.5
    limit_up_touch = float(row.get("limit_up_touch", 0.0) or 0.0) > 0.5
    limit_up_close = float(row.get("limit_up_close", 0.0) or 0.0) > 0.5
    limit_up_strength = float(row.get("limit_up_strength", 0.0) or 0.0)
    failed_turnover = float(row.get("failed_limit_up_turnover", 0.0) or 0.0)

    if float(row.get("ma_bull_stack", 0.0) or 0.0) > 0.5 and float(row.get("sma_20_to_60", 0.0) or 0.0) > 0:
        score += 0.20 * weights.get("trend_follow", 1.0)
        labels.append("趋势主线")
    if float(row.get("breakout_20", 0.0) or 0.0) > 0.5 and volume_z > 0.5:
        score += 0.18 * weights.get("breakout", 1.0) * weights.get("volume_confirm", 1.0)
        labels.append("放量突破")
    if -0.035 <= float(row.get("close_to_sma_20", 0.0) or 0.0) <= 0.035 and ret_20 > 0:
        score += 0.16 * weights.get("pullback_entry", 1.0)
        labels.append("回踩低吸")
    if news_score > 0:
        score += min(0.26, math.log1p(news_score) * 0.06) * weights.get("mainline_theme", 1.0)
        labels.append("新闻主线")
    if any(token in themes_text for token in ["ai", "半导体", "算力", "芯片", "光模块", "能源", "存储"]):
        score += 0.14 * weights.get("ai_infrastructure", 1.0)
        labels.append("AI基础设施")
    if volume_z > 1.0 and float(row.get("amount_z_20", 0.0) or 0.0) > 0.6 and ret_20 > 0:
        score += 0.12 * weights.get("institutional_flow", 1.0)
        labels.append("资金确认")
    if float(row.get("macd_hist", 0.0) or 0.0) > 0:
        score += min(0.10, float(row.get("macd_hist", 0.0) or 0.0) * 8.0) * weights.get("macd_momentum", 1.0)

    if limit_up_close and volume_z > 0.3 and ret_20 < 0.35:
        score += 0.12 * weights.get("limit_up_strength", 1.0)
        labels.append("涨停强势")
    if failed_limit_up:
        repair_setup = (
            limit_up_strength >= 0.70
            and volume_z >= 0.8
            and 0.0 < ret_20 < 0.35
            and close_to_sma_20 > -0.03
        )
        distribution_risk = (
            limit_up_strength < 0.55
            or ret_20 > 0.35
            or ret_60 > 0.80
            or failed_turnover > 8.0
            or volatility > 0.065
        )
        if repair_setup and not distribution_risk:
            score += 0.16 * weights.get("bad_board_repair", 1.0)
            labels.append("烂板弱转强观察")
        else:
            score -= 0.26 * weights.get("bad_board_risk", 1.0)
            labels.append("烂板风险")
    elif limit_up_touch and not limit_up_close:
        score -= 0.16 * weights.get("bad_board_risk", 1.0)
        labels.append("触板未封")

    overheated = (ret_20 > 0.22 or ret_60 > 0.55) and (close_to_high > -0.04 or volume_z > 1.8)
    if overheated:
        score -= 0.30 * weights.get("anti_chase", 1.0)
        score -= 0.18 * weights.get("expectation_exhaustion", 1.0)
        labels.append("高位兑现风险")
    if volatility > 0.055 or float(row.get("pred_drawdown", -0.1) or -0.1) < -0.12:
        score -= 0.18 * weights.get("risk_discipline", 1.0)
        labels.append("风控压制")
    if float(row.get("ma_bear_stack", 0.0) or 0.0) > 0.5:
        score -= 0.16 * weights.get("macro_cycle", 1.0)
        labels.append("宏观/趋势逆风")

    return max(-1.0, min(1.0, score)), labels[:5]


def _buy_price_bounds(row: pd.Series, action: str) -> tuple[float, float, str]:
    entry = float(row["close"])
    atr_pct = max(0.004, min(0.18, float(row.get("atr_pct_14", 0.0) or 0.0)))
    pullback = max(0.006, min(0.035, atr_pct * 0.45))
    chase = max(0.003, min(0.020, atr_pct * 0.25))
    if action == "BUY":
        low = entry * (1 - pullback)
        high = entry * (1 + chase)
        rule = "优先在区间内分批限价买入，突破区间上沿不追高"
    elif action == "WATCH":
        low = entry * (1 - pullback * 1.25)
        high = entry * (1 + chase * 0.50)
        rule = "观察为主，只在回落到区间内且量价未转弱时考虑"
    else:
        low = entry * (1 - pullback * 1.50)
        high = entry * (1 - pullback * 0.25)
        rule = "当前不建议主动买入，仅作为更低价格的观察区间"
    return round(min(low, high), 3), round(max(low, high), 3), rule


def _future_buy_window(signal_date: pd.Timestamp, action: str) -> str:
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    base = max(pd.Timestamp(signal_date).normalize(), today)
    first_day = (base + BDay(1)).date().isoformat()
    third_day = (base + BDay(3)).date().isoformat()
    if action == "BUY":
        return f"{first_day} 09:40 至 {third_day} 10:30，回踩买入区间且量能不转弱时分批"
    if action == "WATCH":
        return f"{first_day} 至 {third_day}，等待站回短期均线或放量突破后再考虑"
    return f"{first_day} 之后暂不主动买入，等待新信号"


def _dynamic_score(
    row: pd.Series,
    plan_shares: int,
    plan_position_pct: float,
    cfg: AppConfig,
    horizon: int,
    holding_days: int | None,
) -> float:
    rr_score = float(row["base_score"])
    technical = _technical_tilt(row, horizon, holding_days)
    affordability = 0.20 if plan_shares > 0 else -0.70
    position_use = min(max(plan_position_pct / max(cfg.risk.max_position_pct, 0.001), 0.0), 1.0) * 0.12
    direction_boost = (float(row.get("direction_probability", 0.5) or 0.5) - 0.5) * 1.60
    trend_boost = (float(row.get("trend_probability", 0.5) or 0.5) - 0.5) * 1.20
    knowledge_boost = max(-0.60, min(0.60, float(row.get("knowledge_score", 0.0) or 0.0) * 0.45))
    news_score = max(0.0, float(row.get("news_score", 0.0) or 0.0))
    news_boost = min(1.25, math.log1p(news_score) * 0.38)
    return rr_score + technical + knowledge_boost + direction_boost + trend_boost + affordability + position_use + news_boost


def _news_signal_maps(news_signals: dict[str, dict[str, object]] | None) -> dict[str, dict[str, object]]:
    if not news_signals:
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for symbol, payload in news_signals.items():
        normalized[str(symbol).zfill(6)] = payload
    return normalized


def _probability_threshold(threshold_payload: dict[str, object], key: str, fallback: float = 0.58) -> float:
    item = threshold_payload.get(key)
    if not isinstance(item, dict):
        return fallback
    confidence = float(item.get("selected_confidence_threshold", 0.16) or 0.16)
    return max(fallback, min(0.88, 0.5 + confidence / 2.0))


def _select_prediction_model(bundle: dict[str, object], requested_days: int | None) -> tuple[int, object, object, object | None, object | None, dict[str, object]]:
    horizon_models = bundle.get("horizon_models")
    if isinstance(horizon_models, dict) and horizon_models:
        available = sorted(int(h) for h in horizon_models)
        target = int(requested_days) if requested_days and requested_days > 0 else int(bundle["horizon_days"])
        selected = min(available, key=lambda h: abs(h - target))
        model_payload = horizon_models[str(selected)]
        return (
            selected,
            model_payload["return_model"],
            model_payload["drawdown_model"],
            model_payload.get("direction_model"),
            model_payload.get("trend_model"),
            dict(model_payload.get("confidence_thresholds") or {}),
        )
    return (
        int(bundle["horizon_days"]),
        bundle["return_model"],
        bundle["drawdown_model"],
        bundle.get("direction_model"),
        bundle.get("trend_model"),
        dict(bundle.get("confidence_thresholds") or {}),
    )


def make_recommendations(
    history: pd.DataFrame,
    cfg: AppConfig,
    symbols: list[str] | None = None,
    capital: float | None = None,
    holding_days: int | None = None,
    news_signals: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    bundle = load_model(cfg.model.model_dir)
    model_features = list(bundle.get("feature_columns") or FEATURE_COLUMNS)
    horizon, return_model, drawdown_model, direction_model, trend_model, confidence_thresholds = _select_prediction_model(bundle, holding_days)
    feature_table = coerce_model_features(build_feature_table(history, horizon=horizon), model_features)
    selected = set(symbols or cfg.data.symbols)
    latest = (
        feature_table[feature_table["symbol"].isin(selected)]
        .dropna(subset=model_features)
        .sort_values(["symbol", "date"])
        .groupby("symbol")
        .tail(1)
        .copy()
    )
    if latest.empty:
        raise ValueError("No latest feature rows are available for recommendation.")

    model_input = latest[model_features].astype("float64")
    latest["model_pred_return"] = return_model.predict(model_input)
    latest["model_pred_drawdown"] = drawdown_model.predict(model_input)
    if direction_model is not None:
        latest["direction_probability"] = predict_positive_probability(direction_model, model_input)
    else:
        latest["direction_probability"] = latest["model_pred_return"].apply(lambda value: 1.0 / (1.0 + math.exp(-float(value) * 18.0)))
    if trend_model is not None:
        latest["trend_probability"] = predict_positive_probability(trend_model, model_input)
    else:
        latest["trend_probability"] = latest["direction_probability"]
    latest["min_direction_probability"] = _probability_threshold(confidence_thresholds, "direction", fallback=0.58)
    latest["min_trend_probability"] = _probability_threshold(confidence_thresholds, "trend", fallback=0.58)
    latest = _adjust_prediction_columns(latest, horizon=horizon, holding_days=holding_days)
    news_by_symbol = _news_signal_maps(news_signals)
    knowledge = load_kline_knowledge(cfg)
    knowledge_weights = knowledge.get("weights", {})
    if not isinstance(knowledge_weights, dict):
        knowledge_weights = {}

    rows: list[dict[str, object]] = []
    for _, row in latest.iterrows():
        plan = plan_position(
            entry_price=float(row["close"]),
            atr_pct=float(row["atr_pct_14"]),
            predicted_return=float(row["pred_return"]),
            risk=cfg.risk,
            capital=capital,
        )
        row = row.copy()
        news = news_by_symbol.get(str(row["symbol"]).zfill(6), {})
        row["news_score"] = float(news.get("news_score", 0.0) or 0.0)
        themes = news.get("themes", [])
        themes_text = "、".join(str(item) for item in themes[:4]) if isinstance(themes, list) else str(themes or "")
        row["news_themes_text"] = themes_text
        typed_weights = {str(k): float(v) for k, v in knowledge_weights.items()}
        knowledge_score, knowledge_labels = _knowledge_signal(row, typed_weights)
        row["knowledge_score"] = knowledge_score
        row["knowledge_themes"] = "、".join(knowledge_labels)
        row["score"] = _dynamic_score(row, plan.shares, plan.position_pct, cfg, horizon, holding_days)
        action = _action(row, cfg, horizon=horizon, holding_days=holding_days)
        size_limited = action == "BUY" and plan.shares <= 0
        if size_limited:
            action = "WATCH"
        if action != "BUY":
            shares = 0
            cash = 0.0
        else:
            shares = plan.shares
            cash = plan.cash

        buy_low, buy_high, entry_rule = _buy_price_bounds(row, action)
        buy_window = _future_buy_window(pd.Timestamp(row["date"]), action)
        limit_up_state = _limit_up_state(row)
        news_reason = str(news.get("reason") or "")
        evidence = news.get("evidence", [])
        evidence_titles = []
        if isinstance(evidence, list):
            for item in evidence[:3]:
                if isinstance(item, dict) and item.get("title"):
                    evidence_titles.append(str(item["title"]))
        reason = _reason(row)
        if row["news_score"] > 0:
            reason += f"；近期新闻主题：{themes_text or '相关板块'}，新闻分 {row['news_score']:.2f}"
            if news_reason:
                reason += f"；{news_reason}"
        rows.append(
            {
                "symbol": row["symbol"],
                "signal_date": pd.Timestamp(row["date"]).date().isoformat(),
                "action": action,
                "score": round(float(row["score"]), 4),
                "pred_return": round(float(row["pred_return"]), 4),
                "pred_drawdown": round(float(row["pred_drawdown"]), 4),
                "direction_prediction": "UP" if float(row["direction_probability"]) >= 0.5 else "DOWN",
                "direction_probability": round(float(row["direction_probability"]), 4),
                "trend_prediction": "UP" if float(row["trend_probability"]) >= 0.5 else "DOWN",
                "trend_probability": round(float(row["trend_probability"]), 4),
                "direction_confidence": round(float(row["direction_confidence"]), 4),
                "trend_confidence": round(float(row["trend_confidence"]), 4),
                "model_pred_return": round(float(row["model_pred_return"]), 4),
                "model_pred_drawdown": round(float(row["model_pred_drawdown"]), 4),
                "requested_holding_days": int(holding_days) if holding_days and holding_days > 0 else int(horizon),
                "reference_entry": round(float(row["close"]), 3),
                "planned_buy_time": buy_window,
                "buy_price_low": buy_low,
                "buy_price_high": buy_high,
                "buy_price_range": f"{buy_low:.2f}-{buy_high:.2f}",
                "entry_rule": entry_rule,
                "shares": int(shares),
                "cash": round(float(cash), 2),
                "stop_loss": round(float(plan.stop_loss), 3),
                "risk_amount": round(float(plan.risk_amount if action == "BUY" else 0.0), 2),
                "position_pct": round(float(plan.position_pct if action == "BUY" else 0.0), 4),
                "effective_risk_per_trade_pct": round(float(plan.effective_risk_per_trade_pct), 4),
                "effective_max_position_pct": round(float(plan.effective_max_position_pct), 4),
                "sizing_mode": plan.sizing_mode,
                "knowledge_score": round(float(row["knowledge_score"]), 4),
                "knowledge_themes": row["knowledge_themes"],
                "knowledge_generated_at": knowledge.get("generated_at", ""),
                "news_score": round(float(row["news_score"]), 4),
                "news_themes": themes_text,
                "news_evidence": "；".join(evidence_titles),
                "technical_signals": "、".join(_technical_signal_labels(row)),
                "limit_up_state": limit_up_state,
                "reason": reason + ("；当前本金和动态风控约束下不足一手" if size_limited else ""),
            }
        )

    out = pd.DataFrame(rows)
    action_rank = {"BUY": 0, "WATCH": 1, "AVOID": 2}
    out["_action_rank"] = out["action"].map(action_rank).fillna(9)
    out = out.sort_values(["_action_rank", "score"], ascending=[True, False]).drop(columns=["_action_rank"])
    return out.reset_index(drop=True)


def write_recommendation_reports(df: pd.DataFrame, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "recommendations.csv"
    md_path = reports_dir / "recommendations.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "# A-share research recommendations",
        "",
        "These are model-generated research signals, not guaranteed investment returns.",
        "",
        _markdown_table(df),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
