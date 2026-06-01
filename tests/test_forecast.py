from __future__ import annotations

import pandas as pd

from ashare_ai_agent import forecast as forecast_module
from ashare_ai_agent.config import load_config
from ashare_ai_agent.features import FEATURE_COLUMNS
from ashare_ai_agent.forecast import forecast_kline


class _FakeModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows):  # noqa: ANN001
        return [self.value] * len(rows)


def test_forecast_kline_returns_future_rows(monkeypatch) -> None:
    cfg = load_config("configs/default.toml")
    dates = pd.bdate_range("2025-01-01", periods=220)
    close = pd.Series(range(100, 320), dtype=float)
    history = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close * 0.995,
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": 1000000,
            "amount": close * 1000000,
            "turnover": 1.0,
        }
    )

    monkeypatch.setattr(
        forecast_module,
        "load_model",
        lambda _model_dir: {
            "return_model": _FakeModel(0.03),
            "drawdown_model": _FakeModel(-0.04),
            "horizon_days": 5,
            "feature_columns": FEATURE_COLUMNS,
        },
    )
    monkeypatch.setattr(
        forecast_module,
        "load_kline_knowledge",
        lambda _cfg: {"weights": {"trend_follow": 1.0, "breakout": 1.0, "volume_confirm": 1.0}},
    )

    payload = forecast_kline(history, cfg, symbol="000001", days=5)
    assert len(payload["rows"]) == 5
    assert all(row["forecast"] for row in payload["rows"])
    assert "suggested_future_entry_window" in payload
