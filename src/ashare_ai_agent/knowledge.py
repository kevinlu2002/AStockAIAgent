from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import requests

from .config import AppConfig


DEFAULT_KNOWLEDGE_URLS = [
    "https://www.investopedia.com/trading/candlestick-charting-what-is-it/",
    "https://www.investopedia.com/terms/m/movingaverage.asp",
    "https://www.investopedia.com/terms/m/macd.asp",
    "https://finance.sina.com.cn/roll/2025-02-09/doc-ineivyac7909215.shtml",
]

BASE_RULES = {
    "trend_follow": 1.00,
    "volume_confirm": 1.00,
    "breakout": 1.00,
    "support_resistance": 0.95,
    "reversal_candle": 0.75,
    "macd_momentum": 0.85,
    "rsi_extreme": 0.65,
    "pullback_entry": 0.90,
    "risk_discipline": 1.00,
    "anti_chase": 1.00,
    "expectation_exhaustion": 1.00,
    "mainline_theme": 1.00,
    "ai_infrastructure": 1.00,
    "institutional_flow": 1.00,
    "macro_cycle": 1.00,
    "limit_up_strength": 1.00,
    "bad_board_repair": 1.00,
    "bad_board_risk": 1.00,
}

KEYWORD_BUCKETS = {
    "trend_follow": ("moving average", "trend", "uptrend", "downtrend", "ma", "均线", "趋势"),
    "volume_confirm": ("volume", "turnover", "成交量", "放量", "缩量", "资金"),
    "breakout": ("breakout", "resistance", "突破", "压力位"),
    "support_resistance": ("support", "resistance", "支撑", "压力"),
    "reversal_candle": ("candlestick", "engulfing", "hammer", "shadow", "k-line", "k线", "阳包阴", "锤头", "上影线"),
    "macd_momentum": ("macd", "momentum", "dif", "dea", "动量"),
    "rsi_extreme": ("rsi", "overbought", "oversold", "超买", "超卖"),
    "pullback_entry": ("pullback", "buy the dip", "回踩", "低吸", "分批"),
    "risk_discipline": ("risk", "stop loss", "仓位", "止损", "杠杆", "少赌"),
    "anti_chase": ("do not chase", "追高", "不追", "高位"),
    "expectation_exhaustion": ("expectation", "利好落地", "预期", "兑现"),
    "mainline_theme": ("mainline", "sector", "theme", "赛道", "主线", "行业"),
    "ai_infrastructure": ("ai", "chip", "energy", "storage", "算力", "芯片", "能源", "存储", "光模块"),
    "institutional_flow": ("institutional", "fund", "holding", "机构", "基金", "重仓", "赎回"),
    "macro_cycle": ("macro", "economy", "liquidity", "policy", "宏观", "经济", "流动性", "政策"),
    "limit_up_strength": ("limit-up", "涨停", "封板", "打板", "封单"),
    "bad_board_repair": ("烂板", "炸板", "回封", "弱转强", "分歧转一致", "weak to strong"),
    "bad_board_risk": ("烂板风险", "炸板出货", "高位烂板", "封不住", "诱多", "failed breakout"),
}

PUBLIC_CREATOR_PROFILES = [
    {
        "name": "yuboluo",
        "display_name": "宇菠萝",
        "status": "public_summary",
        "sources": [
            "https://www.douyin.com/video/7533566511844248891",
            "https://www.sina.cn/news/detail/5198578429199451.html",
        ],
        "weights": {
            "risk_discipline": 0.18,
            "anti_chase": 0.18,
            "expectation_exhaustion": 0.16,
            "pullback_entry": 0.08,
        },
        "summary": "Novice-pitfall rules: avoid chasing crowded expectation trades, avoid buying only because a known event is coming, and require risk controls.",
    },
    {
        "name": "justin_sun",
        "display_name": "孙宇晨",
        "status": "public_summary",
        "sources": [
            "https://www.panewslab.com/zh/articles/019e1a41-5243-708f-bbbb-d5c74dbbc65f",
            "https://emcreative.eastmoney.com/app_fortune/article/index.html?artCode=20260510110403508450670&postId=1705093219",
        ],
        "weights": {
            "mainline_theme": 0.10,
            "ai_infrastructure": 0.18,
            "macro_cycle": 0.08,
            "risk_discipline": 0.04,
        },
        "summary": "AI-infrastructure narrative: chips, energy and storage can become sequential bottlenecks; treat it as theme evidence, not direct stock advice.",
    },
    {
        "name": "haoyunxiake",
        "display_name": "好运侠客",
        "status": "unverified_no_public_source",
        "sources": [],
        "weights": {},
        "summary": "No reliable public stock-analysis source was found in the current search, so this profile is recorded but not activated.",
    },
    {
        "name": "liyien",
        "display_name": "李一恩",
        "status": "public_summary",
        "sources": [
            "https://www.douyin.com/shipin/7598014203756169222",
            "https://zjt.aniu.tv/experts_elist_qc_0_px_2_p_22.shtml",
        ],
        "weights": {
            "mainline_theme": 0.18,
            "ai_infrastructure": 0.14,
            "institutional_flow": 0.14,
            "risk_discipline": 0.10,
            "trend_follow": 0.06,
        },
        "summary": "Industry-research and mainline-sector rules: prefer strong industry logic, watch fund redemption pressure, and avoid gambling-style short term trades.",
    },
    {
        "name": "hanxiuyun",
        "display_name": "韩秀云",
        "status": "public_summary",
        "sources": [
            "https://finance.sina.com.cn/other/relink/sinadaxue/2020-11-25/doc-iiznctke3214400.shtml",
            "https://news.sina.cn/sa/2010-06-18/detail-ikftssap3318977.d.html?vt=4",
        ],
        "weights": {
            "macro_cycle": 0.20,
            "risk_discipline": 0.14,
            "mainline_theme": 0.08,
        },
        "summary": "Macro-first rules: connect stock choices to macro cycles, policy and liquidity, and avoid leverage or asset-mismatch speculation.",
    },
]


def _cache_path(cfg: AppConfig) -> Path:
    return cfg.project_root / "data" / "processed" / "learned_kline_knowledge.json"


def _clean_text(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _configured_urls() -> list[str]:
    raw = os.environ.get("ASHARE_KNOWLEDGE_URLS", "").strip()
    if not raw:
        return DEFAULT_KNOWLEDGE_URLS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _fetch_source_text(url: str) -> str:
    response = requests.get(url, timeout=12, headers={"User-Agent": "AStockAIAgent/1.0"})
    response.raise_for_status()
    return _clean_text(response.text)


def _apply_creator_profiles(weights: dict[str, float]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for profile in PUBLIC_CREATOR_PROFILES:
        active = profile["status"] == "public_summary"
        if active:
            for bucket, boost in profile["weights"].items():
                weights[bucket] = min(1.55, weights.get(bucket, 1.0) + float(boost))
        profiles.append(
            {
                "name": profile["name"],
                "display_name": profile["display_name"],
                "status": profile["status"],
                "active": active,
                "sources": profile["sources"],
                "summary": profile["summary"],
                "weights": profile["weights"],
            }
        )
    return profiles


def _build_payload(weights: dict[str, float], sources: list[dict[str, Any]]) -> dict[str, Any]:
    creator_profiles = _apply_creator_profiles(weights)
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "weights": {key: round(float(value), 3) for key, value in weights.items()},
        "sources": sources,
        "creator_profiles": creator_profiles,
        "note": "Only public summaries and topic weights are stored. Raw creator/video text is not copied. Unverified creators are recorded but not activated.",
    }


def base_knowledge_payload() -> dict[str, Any]:
    return _build_payload(dict(BASE_RULES), [])


def learn_kline_knowledge(cfg: AppConfig, refresh: bool = False, max_cache_age_seconds: int = 86_400) -> dict[str, Any]:
    path = _cache_path(cfg)
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < max_cache_age_seconds:
        return json.loads(path.read_text(encoding="utf-8"))

    weights = dict(BASE_RULES)
    sources: list[dict[str, Any]] = []
    for url in _configured_urls():
        try:
            text = _fetch_source_text(url)
            bucket_hits: dict[str, int] = {}
            for bucket, keywords in KEYWORD_BUCKETS.items():
                count = sum(text.count(keyword.lower()) for keyword in keywords)
                if count:
                    bucket_hits[bucket] = count
                    weights[bucket] = min(1.55, weights.get(bucket, 1.0) + min(0.25, count / 120))
            sources.append({"url": url, "ok": True, "hits": bucket_hits})
        except Exception as exc:
            sources.append({"url": url, "ok": False, "error": repr(exc)})

    payload = _build_payload(weights, sources)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_kline_knowledge(cfg: AppConfig) -> dict[str, Any]:
    path = _cache_path(cfg)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return base_knowledge_payload()
