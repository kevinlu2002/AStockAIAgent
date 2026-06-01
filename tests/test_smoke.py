from __future__ import annotations

import pandas as pd

from ashare_ai_agent import recommend as recommend_module
from ashare_ai_agent.features import FEATURE_COLUMNS, build_feature_table, training_rows
from ashare_ai_agent.portfolio import dynamic_risk_profile, plan_position
from ashare_ai_agent.config import RiskConfig
from ashare_ai_agent.universe import add_price_bucket
from ashare_ai_agent.recommend import make_recommendations
from ashare_ai_agent.config import load_config


class _DtypeCheckingModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows):  # noqa: ANN001
        bad = [col for col in rows.columns if not pd.api.types.is_numeric_dtype(rows[col])]
        assert not bad
        return [self.value] * len(rows)


def test_feature_table_has_training_rows() -> None:
    dates = pd.bdate_range("2024-01-01", periods=120)
    close = pd.Series(range(100, 220), dtype=float)
    history = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1000000,
            "amount": close * 1000000,
            "turnover": 1.0,
        }
    )
    table = build_feature_table(history, horizon=5)
    rows = training_rows(table, horizon=5)
    assert not rows.empty
    assert set(FEATURE_COLUMNS).issubset(rows.columns)


def test_failed_limit_up_feature_detects_bad_board() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    close = pd.Series([10.0] * 79 + [10.5], dtype=float)
    high = pd.Series([10.2] * 79 + [11.0], dtype=float)
    history = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close,
            "high": high,
            "low": close * 0.98,
            "close": close,
            "volume": 1000000,
            "amount": close * 1000000,
            "turnover": 2.0,
        }
    )
    latest = build_feature_table(history, horizon=5).iloc[-1]
    assert latest["limit_up_touch"] == 1.0
    assert latest["limit_up_close"] == 0.0
    assert latest["failed_limit_up"] == 1.0
    assert latest["failed_limit_up_turnover"] == 2.0


def test_bad_board_knowledge_signal_separates_repair_and_risk() -> None:
    weights = {
        "bad_board_repair": 1.0,
        "bad_board_risk": 1.0,
        "pullback_entry": 1.0,
        "risk_discipline": 1.0,
        "anti_chase": 1.0,
        "expectation_exhaustion": 1.0,
    }
    repair_row = pd.Series(
        {
            "failed_limit_up": 1.0,
            "limit_up_strength": 0.75,
            "failed_limit_up_turnover": 3.0,
            "volume_z_20": 1.2,
            "amount_z_20": 0.4,
            "ret_20": 0.12,
            "ret_60": 0.22,
            "close_to_sma_20": 0.02,
            "close_to_high_20": -0.08,
            "volatility_20": 0.03,
            "pred_drawdown": -0.05,
        }
    )
    score, labels = recommend_module._knowledge_signal(repair_row, weights)
    assert score > 0
    assert "烂板弱转强观察" in labels

    risk_row = repair_row.copy()
    risk_row["ret_20"] = 0.48
    risk_row["failed_limit_up_turnover"] = 10.0
    risk_score, risk_labels = recommend_module._knowledge_signal(risk_row, weights)
    assert risk_score < 0
    assert "烂板风险" in risk_labels


def test_position_plan_uses_lots() -> None:
    risk = RiskConfig(
        capital=100000,
        risk_per_trade_pct=0.01,
        max_position_pct=0.2,
        max_portfolio_positions=5,
        stop_loss_pct=0.07,
        atr_stop_multiple=2.0,
        min_expected_return=0.025,
        max_expected_drawdown=-0.10,
        lot_size=100,
    )
    plan = plan_position(10.0, 0.02, 0.05, risk)
    assert plan.shares % 100 == 0
    assert plan.cash <= 100000 * plan.effective_max_position_pct


def test_small_capital_dynamic_risk_profile_allows_concentrated_position() -> None:
    risk = RiskConfig(
        capital=1000,
        risk_per_trade_pct=0.01,
        max_position_pct=0.2,
        max_portfolio_positions=5,
        stop_loss_pct=0.07,
        atr_stop_multiple=2.0,
        min_expected_return=0.025,
        max_expected_drawdown=-0.10,
        lot_size=100,
    )
    risk_pct, pos_pct, mode = dynamic_risk_profile(1000, risk)
    assert risk_pct >= 0.05
    assert pos_pct >= 0.95
    assert "小资金" in mode
    plan = plan_position(5.0, 0.02, 0.05, risk, capital=1000)
    assert plan.shares >= 100


def test_price_bucket_assignment() -> None:
    df = pd.DataFrame({"price": [5.5, 20.0, 150.0]})
    assert add_price_bucket(df)["price_bucket"].tolist() == ["0-10", "10-100", "100+"]


def test_recommendations_coerce_feature_dtypes_before_model_predict(monkeypatch) -> None:
    cfg = load_config("configs/default.toml")
    dates = pd.bdate_range("2024-01-01", periods=180)
    close = pd.Series(range(100, 280), dtype=float)
    close.iloc[90] = 0.0
    history = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close.where(close > 0, 100.0) * 0.995,
            "high": close.where(close > 0, 100.0) * 1.015,
            "low": close.where(close > 0, 100.0) * 0.985,
            "close": close,
            "volume": 1000000,
            "amount": close.where(close > 0, 100.0) * 1000000,
            "turnover": 1.0,
        }
    )

    monkeypatch.setattr(
        recommend_module,
        "load_model",
        lambda _model_dir: {
            "return_model": _DtypeCheckingModel(0.03),
            "drawdown_model": _DtypeCheckingModel(-0.04),
            "horizon_days": 5,
            "feature_columns": FEATURE_COLUMNS,
        },
    )
    monkeypatch.setattr(
        recommend_module,
        "load_kline_knowledge",
        lambda _cfg: {
            "generated_at": "2026-01-01T00:00:00+08:00",
            "weights": {
                "trend_follow": 1.0,
                "mainline_theme": 1.2,
                "risk_discipline": 1.0,
                "anti_chase": 1.0,
            },
        },
    )

    recommendations = make_recommendations(
        history,
        cfg,
        symbols=["000001"],
        capital=100000,
        news_signals={
            "000001": {
                "news_score": 3.5,
                "themes": ["AI/半导体算力"],
                "reason": "测试新闻主题",
                "evidence": [{"title": "AI chip investment rises"}],
            }
        },
    )
    assert not recommendations.empty
    assert recommendations.iloc[0]["news_score"] == 3.5
    assert "knowledge_score" in recommendations.columns
