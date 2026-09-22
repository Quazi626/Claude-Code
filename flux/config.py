"""Configuration loading: config.yaml + .env, merged into a typed-ish namespace."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AlpacaCreds:
    api_key: str
    secret_key: str
    base_url: str
    data_url: str
    is_live: bool


@dataclass
class FluxConfig:
    raw: dict[str, Any]
    creds: AlpacaCreds
    live_trading_confirm_env: bool

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(config_path: str | None = None, env_path: str | None = None) -> FluxConfig:
    load_dotenv(dotenv_path=env_path or (REPO_ROOT / ".env"))

    path = Path(config_path or os.environ.get("FLUX_CONFIG_PATH", "config.yaml"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    environment = raw.get("mode", {}).get("environment", "paper")
    is_live_requested = environment == "live"

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    data_url = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")

    live_confirm_env = os.environ.get("LIVE_TRADING_CONFIRM", "").strip() == "YES"

    creds = AlpacaCreds(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url,
        data_url=data_url,
        is_live=is_live_requested,
    )

    if is_live_requested and "paper" in base_url:
        raise RuntimeError(
            "config.yaml requests mode.environment=live but ALPACA_BASE_URL still "
            "points at the paper endpoint. Set ALPACA_BASE_URL to the live endpoint "
            "explicitly to avoid accidental misconfiguration."
        )

    return FluxConfig(raw=raw, creds=creds, live_trading_confirm_env=live_confirm_env)


def require_live_confirmation(cfg: FluxConfig, input_fn=input) -> bool:
    """Enforce the two-factor live-trading gate: config flag AND typed confirmation.

    Returns True only if live trading is fully authorized. Paper mode always
    returns False (meaning: stay in paper / simulated order routing).
    """
    if not cfg.creds.is_live:
        return False

    if not cfg.live_trading_confirm_env:
        raise RuntimeError(
            "mode.environment=live but LIVE_TRADING_CONFIRM=YES is not set in the "
            "environment. Refusing to start in live mode."
        )

    print("=" * 70)
    print("  LIVE TRADING MODE REQUESTED — REAL MONEY WILL BE AT RISK")
    print("=" * 70)
    typed = input_fn(
        "Type exactly 'I UNDERSTAND THE RISK' to proceed in LIVE mode, "
        "anything else aborts: "
    )
    if typed.strip() != "I UNDERSTAND THE RISK":
        raise RuntimeError("Live trading confirmation phrase did not match. Aborting.")
    return True
