"""Thin wrappers around alpaca-py for equities/options/crypto data and trading.

Centralizing client construction and the handful of calls FLUX needs keeps the
rest of the codebase free of SDK-specific details, and gives us one place to
mock in tests/backtests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd
from alpaca.data.enums import OptionsFeed
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    OptionChainRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetStatus,
    ContractType,
    OrderSide,
    OrderType,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.models import Order, OptionContract
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

from flux.config import AlpacaCreds

log = logging.getLogger("flux.data.alpaca_client")


def parse_timeframe(tf_str: str) -> TimeFrame:
    """'5Min' / '15Min' / '1Day' -> alpaca TimeFrame."""
    amount = int("".join(c for c in tf_str if c.isdigit()) or 1)
    unit_str = "".join(c for c in tf_str if c.isalpha()).lower()
    unit_map = {
        "min": TimeFrameUnit.Minute,
        "minute": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
    }
    unit = unit_map.get(unit_str, TimeFrameUnit.Minute)
    return TimeFrame(amount, unit)


@dataclass
class FluxAlpacaClient:
    trading: TradingClient
    stock_data: StockHistoricalDataClient
    option_data: OptionHistoricalDataClient
    crypto_data: CryptoHistoricalDataClient

    @classmethod
    def from_creds(cls, creds: AlpacaCreds) -> "FluxAlpacaClient":
        is_paper = "paper" in creds.base_url
        trading = TradingClient(
            api_key=creds.api_key,
            secret_key=creds.secret_key,
            paper=is_paper,
        )
        stock_data = StockHistoricalDataClient(
            api_key=creds.api_key, secret_key=creds.secret_key
        )
        option_data = OptionHistoricalDataClient(
            api_key=creds.api_key, secret_key=creds.secret_key
        )
        crypto_data = CryptoHistoricalDataClient(
            api_key=creds.api_key, secret_key=creds.secret_key
        )
        return cls(
            trading=trading,
            stock_data=stock_data,
            option_data=option_data,
            crypto_data=crypto_data,
        )

    # ---- account / clock ----

    def get_account_equity(self) -> float:
        account = self.trading.get_account()
        return float(account.equity)

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    # ---- equities bars ----

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=parse_timeframe(timeframe),
            start=start,
            end=end,
            limit=limit,
        )
        bars = self.stock_data.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df

    def get_prior_session_ohlc(self, symbol: str, as_of: Optional[date] = None) -> dict:
        """Prior completed trading day's OHLC, for floor pivot calculation.

        Fetches the last several daily bars and drops one still dated "today"
        if present, rather than relying on exact end-of-range semantics (which
        vary depending on whether the request lands before/after the close).
        """
        today = as_of or date.today()
        start = datetime.combine(today, datetime.min.time()).replace(year=today.year - 1)
        df = self.get_stock_bars(symbol, "1Day", start=start, limit=10)
        if df.empty:
            raise ValueError(f"No daily bars available for {symbol} to compute pivots")
        df = df[df.index.date < today]
        if df.empty:
            raise ValueError(f"No prior-session daily bar available for {symbol} before {today}")
        prior = df.iloc[-1]
        return {
            "open": float(prior["open"]),
            "high": float(prior["high"]),
            "low": float(prior["low"]),
            "close": float(prior["close"]),
        }

    # ---- crypto (BTC gate) ----

    def get_latest_crypto_price(self, symbol: str = "BTC/USD") -> float:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            limit=1,
        )
        bars = self.crypto_data.get_crypto_bars(req)
        df = bars.df
        if df.empty:
            raise ValueError(f"No crypto bars returned for {symbol}")
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return float(df.iloc[-1]["close"])

    # ---- options chain / contracts ----

    def get_option_contracts(
        self,
        underlying_symbol: str,
        expiration_date: Optional[date] = None,
        expiration_date_gte: Optional[date] = None,
        expiration_date_lte: Optional[date] = None,
        contract_type: Optional[ContractType] = None,
    ) -> list[OptionContract]:
        """Contract metadata including open interest, via the trading API."""
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying_symbol],
            status=AssetStatus.ACTIVE,
            expiration_date=expiration_date,
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
            type=contract_type,
        )
        resp = self.trading.get_option_contracts(req)
        return list(resp.option_contracts)

    def get_option_chain_snapshot(self, underlying_symbol: str, expiration_date: Optional[date] = None):
        """Latest quote/trade/IV/greeks per contract symbol, via the market data API."""
        req = OptionChainRequest(
            underlying_symbol=underlying_symbol,
            feed=OptionsFeed.INDICATIVE,
            expiration_date=expiration_date,
        )
        return self.option_data.get_option_chain(req)

    # ---- orders ----

    def submit_limit_order(
        self,
        option_symbol: str,
        qty: int,
        side: OrderSide,
        limit_price: float,
        position_intent: PositionIntent,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
    ) -> Order:
        req = LimitOrderRequest(
            symbol=option_symbol,
            qty=qty,
            side=side,
            type=OrderType.LIMIT,
            time_in_force=time_in_force,
            limit_price=round(limit_price, 2),
            position_intent=position_intent,
            client_order_id=client_order_id,
        )
        return self.trading.submit_order(req)

    def cancel_order(self, order_id: str) -> None:
        self.trading.cancel_order_by_id(order_id)

    def get_order(self, order_id: str) -> Order:
        return self.trading.get_order_by_id(order_id)

    def get_positions(self):
        return self.trading.get_all_positions()

    def close_all_positions(self, cancel_orders: bool = True):
        return self.trading.close_all_positions(cancel_orders=cancel_orders)

    def close_position(self, symbol: str):
        return self.trading.close_position(symbol)
