from __future__ import annotations

from dataclasses import dataclass
import math

from .config import RiskConfig


@dataclass(frozen=True)
class PositionPlan:
    shares: int
    cash: float
    entry_price: float
    stop_loss: float
    risk_amount: float
    position_pct: float
    effective_risk_per_trade_pct: float
    effective_max_position_pct: float
    sizing_mode: str


def _floor_lot(shares: float, lot_size: int) -> int:
    if shares <= 0:
        return 0
    return int(math.floor(shares / lot_size) * lot_size)


def dynamic_risk_profile(capital: float, risk: RiskConfig) -> tuple[float, float, str]:
    """Return risk-per-trade pct, max position pct, and a human-readable sizing mode."""
    if capital <= 5_000:
        return max(risk.risk_per_trade_pct, 0.05), max(risk.max_position_pct, 0.95), "小资金集中持仓"
    if capital <= 20_000:
        return max(risk.risk_per_trade_pct, 0.035), max(risk.max_position_pct, 0.70), "小资金偏集中"
    if capital <= 100_000:
        return max(risk.risk_per_trade_pct, 0.02), max(risk.max_position_pct, 0.35), "中等资金均衡"
    if capital <= 500_000:
        return max(risk.risk_per_trade_pct, 0.015), max(risk.max_position_pct, 0.25), "较大资金分散"
    return risk.risk_per_trade_pct, risk.max_position_pct, "大资金严格风控"


def plan_position(
    entry_price: float,
    atr_pct: float,
    predicted_return: float,
    risk: RiskConfig,
    capital: float | None = None,
) -> PositionPlan:
    capital = float(capital if capital is not None else risk.capital)
    entry_price = float(entry_price)
    if entry_price <= 0:
        return PositionPlan(0, 0.0, entry_price, 0.0, 0.0, 0.0, risk.risk_per_trade_pct, risk.max_position_pct, "无效价格")

    risk_per_trade_pct, max_position_pct, sizing_mode = dynamic_risk_profile(capital, risk)

    atr_pct = max(float(atr_pct or 0), 0)
    stop_gap_pct = max(risk.stop_loss_pct, atr_pct * risk.atr_stop_multiple)
    stop_loss = entry_price * (1 - stop_gap_pct)
    per_share_risk = max(entry_price - stop_loss, entry_price * 0.01)

    max_risk_cash = capital * risk_per_trade_pct
    risk_limited_shares = max_risk_cash / per_share_risk

    signal_scale = min(1.0, max(0.25, predicted_return / max(risk.min_expected_return * 2, 0.001)))
    position_cap_cash = capital * max_position_pct * signal_scale
    cap_limited_shares = position_cap_cash / entry_price

    shares = _floor_lot(min(risk_limited_shares, cap_limited_shares), risk.lot_size)
    cash = shares * entry_price
    risk_amount = shares * per_share_risk
    position_pct = cash / capital if capital > 0 else 0.0
    return PositionPlan(
        shares=shares,
        cash=float(cash),
        entry_price=float(entry_price),
        stop_loss=float(stop_loss),
        risk_amount=float(risk_amount),
        position_pct=float(position_pct),
        effective_risk_per_trade_pct=float(risk_per_trade_pct),
        effective_max_position_pct=float(max_position_pct),
        sizing_mode=sizing_mode,
    )
