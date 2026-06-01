from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DataConfig:
    symbols: list[str]
    start_date: str
    end_date: str
    adjust: str
    raw_dir: Path
    allow_download: bool


@dataclass(frozen=True)
class ModelConfig:
    horizon_days: int
    validation_fraction: float
    min_rows_per_symbol: int
    random_state: int
    model_dir: Path


@dataclass(frozen=True)
class RiskConfig:
    capital: float
    risk_per_trade_pct: float
    max_position_pct: float
    max_portfolio_positions: int
    stop_loss_pct: float
    atr_stop_multiple: float
    min_expected_return: float
    max_expected_drawdown: float
    lot_size: int


@dataclass(frozen=True)
class OutputConfig:
    reports_dir: Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    data: DataConfig
    model: ModelConfig
    risk: RiskConfig
    output: OutputConfig


def _project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name.lower() == "configs":
        return config_path.parent.parent
    return Path.cwd().resolve()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    root = _project_root(config_path)
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    data = raw.get("data", {})
    model = raw.get("model", {})
    risk = raw.get("risk", {})
    output = raw.get("output", {})

    return AppConfig(
        project_root=root,
        data=DataConfig(
            symbols=[str(s).zfill(6) for s in data.get("symbols", [])],
            start_date=str(data.get("start_date", "20180101")),
            end_date=str(data.get("end_date", "20260526")),
            adjust=str(data.get("adjust", "qfq")),
            raw_dir=_resolve(root, data.get("raw_dir", "data/raw")),
            allow_download=bool(data.get("allow_download", True)),
        ),
        model=ModelConfig(
            horizon_days=int(model.get("horizon_days", 5)),
            validation_fraction=float(model.get("validation_fraction", 0.2)),
            min_rows_per_symbol=int(model.get("min_rows_per_symbol", 180)),
            random_state=int(model.get("random_state", 42)),
            model_dir=_resolve(root, model.get("model_dir", "models")),
        ),
        risk=RiskConfig(
            capital=float(risk.get("capital", 100000.0)),
            risk_per_trade_pct=float(risk.get("risk_per_trade_pct", 0.01)),
            max_position_pct=float(risk.get("max_position_pct", 0.2)),
            max_portfolio_positions=int(risk.get("max_portfolio_positions", 5)),
            stop_loss_pct=float(risk.get("stop_loss_pct", 0.07)),
            atr_stop_multiple=float(risk.get("atr_stop_multiple", 2.0)),
            min_expected_return=float(risk.get("min_expected_return", 0.025)),
            max_expected_drawdown=float(risk.get("max_expected_drawdown", -0.10)),
            lot_size=int(risk.get("lot_size", 100)),
        ),
        output=OutputConfig(
            reports_dir=_resolve(root, output.get("reports_dir", "reports")),
        ),
    )
