from __future__ import annotations

from ashare_ai_agent.news import rank_stocks_from_themes, score_articles_by_theme


def test_news_theme_scoring_finds_oil_and_gold() -> None:
    articles = [
        {
            "source": "test",
            "title": "Oil prices surge as Middle East conflict raises supply concerns",
            "summary": "Investors also buy gold as a safe haven.",
            "link": "https://example.test/news",
            "published_at": "2026-05-26T00:00:00+00:00",
        }
    ]
    themes = score_articles_by_theme(articles)
    names = {row["theme"] for row in themes}
    assert "油气能源" in names
    assert "黄金避险" in names


def test_news_stock_ranking_returns_a_share_candidates() -> None:
    themes = [
        {
            "theme": "AI/半导体算力",
            "score": 3.2,
            "stocks": [{"symbol": "688981", "name": "中芯国际"}],
            "evidence": [{"source": "test", "title": "AI chip investment rises", "link": "https://example.test"}],
        }
    ]
    ranked = rank_stocks_from_themes(themes)
    assert ranked[0]["symbol"] == "688981"
    assert ranked[0]["signal"] == "NEWS_BUY_CANDIDATE"
