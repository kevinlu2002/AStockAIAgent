from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import fetch_history_with_akshare, load_history, parse_symbols, save_raw_history
from .models import train_proxy_model
from .recommend import make_recommendations, write_recommendation_reports


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run an A-share research signal agent.")
    parser.add_argument("--config", default="configs/default.toml", help="Path to TOML config.")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch A-share history into data/raw.")
    fetch.add_argument("--symbols", default=None, help="Comma separated stock codes, e.g. 000001,600519.")

    train = sub.add_parser("train", help="Train the proxy model.")
    train.add_argument("--symbols", default=None, help="Optional comma separated stock codes.")

    rec = sub.add_parser("recommend", help="Generate buy/watch/avoid recommendations.")
    rec.add_argument("--symbols", default=None, help="Optional comma separated stock codes.")
    rec.add_argument("--capital", type=float, default=None, help="Override configured capital.")

    return parser


def cmd_fetch(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbols = parse_symbols(args.symbols, cfg.data.symbols)
    for symbol in symbols:
        df = fetch_history_with_akshare(symbol, cfg)
        path = save_raw_history(df, cfg.data.raw_dir, symbol)
        print(f"saved {symbol}: {path} rows={len(df)}")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbols = parse_symbols(args.symbols, cfg.data.symbols)
    history = load_history(cfg, symbols=symbols)
    result = train_proxy_model(history, cfg)
    print(f"model: {result['model_path']}")
    print(f"metrics: {result['metrics_path']}")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


def cmd_recommend(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbols = parse_symbols(args.symbols, cfg.data.symbols)
    history = load_history(cfg, symbols=symbols)
    df = make_recommendations(history, cfg, symbols=symbols, capital=args.capital)
    csv_path, md_path = write_recommendation_reports(df, cfg.output.reports_dir)
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print(df.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "recommend":
        cmd_recommend(args)
    else:
        parser.error(f"Unknown command: {args.command}")
