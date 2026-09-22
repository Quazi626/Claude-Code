"""File-flag kill switch: create the flag file (e.g. `touch KILL_SWITCH`) to
immediately flatten all positions and halt the bot for the rest of the
session. Checked every loop iteration; no keypress/stdin dependency, so it
works the same whether the bot is running interactively or headless/
backgrounded.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flux.data.alpaca_client import FluxAlpacaClient

log = logging.getLogger("flux.execution.kill_switch")


def is_triggered(flag_file: str) -> bool:
    return Path(flag_file).exists()


def flatten_all(client: FluxAlpacaClient) -> None:
    log.critical("KILL SWITCH TRIGGERED — flattening all positions and cancelling open orders")
    client.close_all_positions(cancel_orders=True)
