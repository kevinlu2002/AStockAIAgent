from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import pandas as pd

from ashare_ai_agent.config import load_config
from ashare_ai_agent.data import save_raw_history
from ashare_ai_agent.evaluation import (
    append_kline_forecast_snapshot,
    append_recommendation_snapshot,
    evaluate_saved_predictions,
)


def _tmp_cfg(root: Path):
    cfg = load_config("configs/default.toml")
    return replace(
        cfg,
        project_root=root,
        data=replace(cfg.data, raw_dir=root / "data" / "raw"),
        output=replace(cfg.output, reports_dir=root / "reports"),
    )


def test_prediction_snapshots_can_be_evaluated() -> None:
    root = Path(".test-output") / "evaluation"
    shutil.rmtree(root, ignore_errors=True)
    cfg = _tmp_cfg(root)
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "symbol": "000001",
            "open": [10.0, 10.2, 10.8],
            "high": [10.3, 11.0, 11.2],
            "low": [9.9, 10.1, 10.7],
            "close": [10.0, 10.8, 11.0],
            "volume": [1000, 1100, 1200],
            "amount": [10000, 11880, 13200],
            "turnover": [1.0, 1.1, 1.2],
        }
    )
    save_raw_history(history, cfg.data.raw_dir, "000001")

    append_kline_forecast_snapshot(
        cfg,
        {
            "symbol": "000001",
            "days": 1,
            "target_return": 0.05,
            "target_drawdown": -0.03,
            "rows": [
                {
                    "date": "2026-01-05",
                    "open": 10.1,
                    "high": 10.9,
                    "low": 10.0,
                    "close": 10.7,
                    "volume": 1000,
                }
            ],
        },
        base_date="2026-01-02",
        base_close=10.0,
    )
    append_recommendation_snapshot(
        cfg,
        pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "name": "平安银行",
                    "signal_date": "2026-01-02",
                    "action": "BUY",
                    "score": 1.0,
                    "pred_return": 0.06,
                    "pred_drawdown": -0.03,
                    "requested_holding_days": 1,
                    "reference_entry": 10.0,
                    "buy_price_low": 9.8,
                    "buy_price_high": 10.2,
                    "shares": 100,
                    "cash": 1000,
                }
            ]
        ),
        run_id="test-run",
        source="test",
        capital=100000,
        holding_days=1,
        universe_mode="single",
    )

    summary = evaluate_saved_predictions(cfg)

    assert summary["kline"]["evaluated"] == 1
    assert summary["kline"]["direction_hit_rate"] == 1.0
    assert summary["recommendations"]["evaluated"] == 1
    assert summary["recommendations"]["direction_hit_rate"] == 1.0
    assert summary["recommendations"]["buy_window_hit_rate"] == 1.0
    shutil.rmtree(root, ignore_errors=True)
