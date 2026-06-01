from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import email.utils
import html
import json
import math
from pathlib import Path
import re
import time
from typing import Any
from xml.etree import ElementTree as ET

import requests

from .config import AppConfig


NEWS_FEEDS = [
    {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "BBC Business", "url": "https://feeds.bbci.co.uk/news/business/rss.xml"},
    {"name": "BBC Technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml"},
    {"name": "CNBC World", "url": "https://www.cnbc.com/id/100727362/device/rss/rss.html"},
]

CATALYST_WORDS = [
    "surge",
    "jump",
    "rally",
    "shortage",
    "sanction",
    "conflict",
    "war",
    "missile",
    "tariff",
    "ban",
    "export control",
    "stimulus",
    "rate cut",
    "supply cut",
    "demand",
    "investment",
    "record high",
    "breakthrough",
]

RISK_WORDS = [
    "slump",
    "plunge",
    "oversupply",
    "glut",
    "peace deal",
    "weak demand",
    "recall",
    "probe",
    "lawsuit",
]


@dataclass(frozen=True)
class Theme:
    key: str
    name: str
    keywords: tuple[str, ...]
    affected_logic: str
    stocks: tuple[tuple[str, str], ...]


THEMES = [
    Theme(
        key="ai_chips",
        name="AI/半导体算力",
        keywords=(
            "ai",
            "artificial intelligence",
            "semiconductor",
            "chip",
            "nvidia",
            "tsmc",
            "gpu",
            "data centre",
            "data center",
            "export control",
            "huawei",
            "advanced chip",
        ),
        affected_logic="全球 AI、芯片、算力投资或出口管制变化，可能提高国产半导体和算力链关注度。",
        stocks=(
            ("688981", "中芯国际"),
            ("002371", "北方华创"),
            ("603986", "兆易创新"),
            ("688041", "海光信息"),
            ("300308", "中际旭创"),
        ),
    ),
    Theme(
        key="energy_oil",
        name="油气能源",
        keywords=("oil", "crude", "brent", "opec", "gas", "lng", "middle east", "iran", "russia", "supply cut"),
        affected_logic="油价、天然气供给、地缘冲突或 OPEC 变化，可能影响油气开采和能源央企。",
        stocks=(("600938", "中国海油"), ("601857", "中国石油"), ("600028", "中国石化"), ("600256", "广汇能源")),
    ),
    Theme(
        key="gold",
        name="黄金避险",
        keywords=("gold", "safe haven", "central bank", "inflation", "dollar", "rate cut", "geopolitical"),
        affected_logic="避险、通胀或降息预期增强时，黄金资产和贵金属股可能获得资金关注。",
        stocks=(("600489", "中金黄金"), ("600547", "山东黄金"), ("601899", "紫金矿业"), ("000975", "银泰黄金")),
    ),
    Theme(
        key="defense",
        name="军工防务",
        keywords=("war", "conflict", "missile", "drone", "nato", "military", "defence", "defense", "ukraine", "taiwan"),
        affected_logic="国际冲突、军费和防务订单预期上升，可能提升军工板块风险偏好。",
        stocks=(("600760", "中航沈飞"), ("000768", "中航西飞"), ("600893", "航发动力"), ("600118", "中国卫星")),
    ),
    Theme(
        key="shipping",
        name="航运港口",
        keywords=("red sea", "suez", "shipping", "freight", "container", "port", "maritime", "canal"),
        affected_logic="航线扰动或运价上升，可能利好集运、油运和港口链条。",
        stocks=(("601919", "中远海控"), ("600026", "中远海能"), ("601872", "招商轮船"), ("001872", "招商港口")),
    ),
    Theme(
        key="battery_ev",
        name="新能源车/电池",
        keywords=("ev", "electric vehicle", "battery", "lithium", "nickel", "cobalt", "tesla", "byd", "charging"),
        affected_logic="全球电动车需求、电池材料价格和政策变化，可能影响电池及整车链。",
        stocks=(("300750", "宁德时代"), ("002594", "比亚迪"), ("002466", "天齐锂业"), ("002460", "赣锋锂业"), ("603799", "华友钴业")),
    ),
    Theme(
        key="solar_storage",
        name="光伏/储能",
        keywords=("solar", "renewable", "photovoltaic", "inverter", "energy storage", "grid", "polysilicon"),
        affected_logic="海外可再生能源政策、储能需求和电网投资变化，可能影响光伏储能产业链。",
        stocks=(("601012", "隆基绿能"), ("300274", "阳光电源"), ("688599", "天合光能"), ("002129", "TCL中环")),
    ),
    Theme(
        key="agriculture",
        name="农业粮食",
        keywords=("grain", "wheat", "corn", "soybean", "food security", "drought", "flood", "fertilizer", "el nino"),
        affected_logic="极端天气、粮价和化肥供需变化，可能影响种业、农垦和化肥链条。",
        stocks=(("000998", "隆平高科"), ("600598", "北大荒"), ("600313", "农发种业"), ("600096", "云天化")),
    ),
    Theme(
        key="healthcare",
        name="医药医疗",
        keywords=("virus", "pandemic", "vaccine", "drug", "health", "hospital", "disease", "who", "outbreak"),
        affected_logic="公共卫生事件、新药进展或医疗需求变化，可能影响创新药和医疗器械。",
        stocks=(("600276", "恒瑞医药"), ("300760", "迈瑞医疗"), ("600196", "复星医药"), ("300122", "智飞生物")),
    ),
    Theme(
        key="critical_minerals",
        name="稀土/关键矿产",
        keywords=("rare earth", "critical minerals", "magnet", "gallium", "germanium", "tungsten", "export controls"),
        affected_logic="关键矿产出口管制、供应链重构和海外资源风险，可能影响稀土及小金属板块。",
        stocks=(("600111", "北方稀土"), ("000831", "中国稀土"), ("600549", "厦门钨业"), ("002428", "云南锗业")),
    ),
]


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _recency_weight(published_at: datetime | None) -> float:
    if published_at is None:
        return 0.45
    hours = max(0.0, (datetime.now(timezone.utc) - published_at).total_seconds() / 3600)
    return max(0.25, math.exp(-hours / 96))


def _fetch_feed(feed: dict[str, str], timeout: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response = requests.get(feed["url"], timeout=timeout, headers={"User-Agent": "AStockAIAgent/1.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    items: list[dict[str, Any]] = []
    for item in root.findall(".//item")[:30]:
        title = _strip_html(item.findtext("title") or "")
        summary = _strip_html(item.findtext("description") or "")
        link = (item.findtext("link") or "").strip()
        published_at = _parse_datetime(item.findtext("pubDate"))
        if not title:
            continue
        items.append(
            {
                "source": feed["name"],
                "title": title,
                "summary": summary,
                "link": link,
                "published_at": published_at.isoformat() if published_at else None,
            }
        )
    return items, {"source": feed["name"], "ok": True, "count": len(items)}


def fetch_latest_news() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    articles: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []
    for feed in NEWS_FEEDS:
        try:
            items, item_status = _fetch_feed(feed)
            articles.extend(items)
            status.append(item_status)
        except Exception as exc:
            status.append({"source": feed["name"], "ok": False, "error": repr(exc)})
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for article in articles:
        key = str(article.get("link") or article.get("title"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped, status


def score_articles_by_theme(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    theme_hits: dict[str, dict[str, Any]] = {}
    for theme in THEMES:
        theme_hits[theme.key] = {"theme": theme, "score": 0.0, "evidence": [], "keywords": set()}

    for article in articles:
        title = str(article.get("title") or "")
        summary = str(article.get("summary") or "")
        text = f"{title} {summary}".lower()
        published_at = _parse_datetime(str(article.get("published_at") or ""))
        recency = _recency_weight(published_at)
        for theme in THEMES:
            matched = [kw for kw in theme.keywords if kw in text]
            if not matched:
                continue
            catalysts = [kw for kw in CATALYST_WORDS if kw in text]
            risks = [kw for kw in RISK_WORDS if kw in text]
            raw = len(set(matched)) + 0.4 * len(set(catalysts)) - 0.35 * len(set(risks))
            if raw <= 0:
                continue
            score = recency * raw
            bucket = theme_hits[theme.key]
            bucket["score"] += score
            bucket["keywords"].update(matched)
            if len(bucket["evidence"]) < 4:
                bucket["evidence"].append(
                    {
                        "source": article.get("source"),
                        "title": title,
                        "link": article.get("link"),
                        "published_at": article.get("published_at"),
                        "matched_keywords": matched[:5],
                    }
                )

    ranked: list[dict[str, Any]] = []
    for bucket in theme_hits.values():
        if bucket["score"] <= 0:
            continue
        theme: Theme = bucket["theme"]
        ranked.append(
            {
                "key": theme.key,
                "theme": theme.name,
                "score": round(float(bucket["score"]), 3),
                "keywords": sorted(bucket["keywords"]),
                "affected_logic": theme.affected_logic,
                "evidence": bucket["evidence"],
                "stocks": [{"symbol": symbol, "name": name} for symbol, name in theme.stocks],
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def rank_stocks_from_themes(themes: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    stocks: dict[str, dict[str, Any]] = {}
    for theme in themes:
        theme_score = float(theme["score"])
        for index, stock in enumerate(theme["stocks"]):
            symbol = str(stock["symbol"])
            weight = max(0.55, 1.0 - index * 0.08)
            score = theme_score * weight
            item = stocks.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "name": stock["name"],
                    "news_score": 0.0,
                    "themes": [],
                    "evidence": [],
                    "reason": "",
                    "signal": "NEWS_WATCH",
                },
            )
            item["news_score"] += score
            item["themes"].append(theme["theme"])
            item["evidence"].extend(theme["evidence"][:2])

    ranked = sorted(stocks.values(), key=lambda item: item["news_score"], reverse=True)[:limit]
    for item in ranked:
        item["news_score"] = round(float(item["news_score"]), 3)
        item["confidence"] = min(92, round(45 + item["news_score"] * 9))
        item["signal"] = "NEWS_BUY_CANDIDATE" if item["news_score"] >= 3.0 else "NEWS_WATCH"
        themes_text = "、".join(dict.fromkeys(item["themes"]))
        item["reason"] = f"近期国际新闻触发 {themes_text} 主题，建议再结合模型评分、实时价格和成交量确认。"
        unique_evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
        for evidence in item["evidence"]:
            key = str(evidence.get("link") or evidence.get("title"))
            if key in seen:
                continue
            seen.add(key)
            unique_evidence.append(evidence)
            if len(unique_evidence) >= 3:
                break
        item["evidence"] = unique_evidence
    return ranked


def _cache_path(cfg: AppConfig) -> Path:
    return cfg.project_root / "data" / "processed" / "news_impact_cache.json"


def analyze_news_impact(cfg: AppConfig, limit: int = 12, refresh: bool = False, max_cache_age_seconds: int = 1200) -> dict[str, Any]:
    path = _cache_path(cfg)
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < max_cache_age_seconds:
        return json.loads(path.read_text(encoding="utf-8"))

    articles, source_status = fetch_latest_news()
    themes = score_articles_by_theme(articles)
    stocks = rank_stocks_from_themes(themes, limit=limit)
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "article_count": len(articles),
        "source_status": source_status,
        "themes": themes[:8],
        "stocks": stocks,
        "disclaimer": "新闻信号只识别潜在催化剂，不构成确定收益预测；应结合模型、价格、仓位和止损共同判断。",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
