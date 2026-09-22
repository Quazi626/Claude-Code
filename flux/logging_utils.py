"""CSV + console logging for every signal, order, fill, and P&L event."""
from __future__ import annotations

import csv
import logging
from pathlib import Path


def setup_console_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def write_row(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


class FluxLoggers:
    """One CSV per event category, plus everything also goes to console via
    the standard `logging` module (see setup_console_logging)."""

    def __init__(self, cfg):
        log_dir = Path(cfg.get("logging", "log_dir", default="logs"))

        self.signals = CsvLogger(
            log_dir / cfg.get("logging", "signals_csv", default="signals.csv"),
            [
                "timestamp", "symbol", "direction", "price", "vwap", "ema_fast", "ema_slow",
                "volume", "volume_avg", "pivot_level_name", "pivot_level_value",
                "above_vwap", "below_vwap", "ema_bull", "ema_bear", "volume_confirm",
                "pivot_reclaim_up", "pivot_reclaim_down",
            ],
        )
        self.orders = CsvLogger(
            log_dir / cfg.get("logging", "orders_csv", default="orders.csv"),
            ["timestamp", "symbol", "option_symbol", "side", "qty", "limit_price",
             "order_id", "filled", "attempts", "reason"],
        )
        self.fills = CsvLogger(
            log_dir / cfg.get("logging", "fills_csv", default="fills.csv"),
            ["timestamp", "symbol", "option_symbol", "side", "qty", "avg_fill_price", "order_id"],
        )
        self.pnl = CsvLogger(
            log_dir / cfg.get("logging", "pnl_csv", default="pnl.csv"),
            [
                "timestamp", "symbol", "option_symbol", "direction", "entry_price", "exit_price",
                "contracts", "gross_pnl", "net_pnl", "exit_reason",
                "session_realized_pnl", "trades_today",
            ],
        )
