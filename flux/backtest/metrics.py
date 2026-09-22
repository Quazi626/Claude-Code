"""Backtest performance reporting: win rate, average win/loss, max drawdown."""
from __future__ import annotations

from dataclasses import dataclass

from flux.backtest.engine import Trade


@dataclass
class BacktestMetrics:
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_win_usd: float
    avg_loss_usd: float
    total_pnl_usd: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    profit_factor: float

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades}  Win rate: {self.win_rate_pct:.1f}%  "
            f"Avg win: ${self.avg_win_usd:,.2f}  Avg loss: ${self.avg_loss_usd:,.2f}\n"
            f"Total P&L: ${self.total_pnl_usd:,.2f}  "
            f"Max drawdown: ${self.max_drawdown_usd:,.2f} ({self.max_drawdown_pct:.1f}%)  "
            f"Profit factor: {self.profit_factor:.2f}"
        )


def compute_metrics(trades: list[Trade], starting_equity: float = 25000.0) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ordered = sorted(trades, key=lambda t: t.exit_time)
    pnls = [t.net_pnl for t in ordered]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    equity_curve = [starting_equity]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)

    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = peak - eq
        dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    return BacktestMetrics(
        total_trades=len(pnls),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=(len(wins) / len(pnls) * 100.0) if pnls else 0.0,
        avg_win_usd=(sum(wins) / len(wins)) if wins else 0.0,
        avg_loss_usd=(sum(losses) / len(losses)) if losses else 0.0,
        total_pnl_usd=sum(pnls),
        max_drawdown_usd=max_dd,
        max_drawdown_pct=max_dd_pct,
        profit_factor=profit_factor,
    )
