from datetime import datetime, timedelta

import pytest

from flux.risk.rules import (
    ET,
    SessionRiskState,
    VixTier,
    btc_gate_allows,
    calc_contracts,
    classify_vix,
    is_entry_allowed,
    is_force_close_time,
    parse_hhmm,
    should_exit_position,
)


def et(hour, minute, day=15):
    return ET.localize(datetime(2025, 9, day, hour, minute))


# ---------------------------------------------------------------------------
# Time gates
# ---------------------------------------------------------------------------

class TestTimeGates:
    def test_entry_blocked_before_cutoff(self):
        cutoff = parse_hhmm("09:45")
        assert is_entry_allowed(et(9, 44), cutoff) is False
        assert is_entry_allowed(et(9, 30), cutoff) is False

    def test_entry_allowed_at_and_after_cutoff(self):
        cutoff = parse_hhmm("09:45")
        assert is_entry_allowed(et(9, 45), cutoff) is True
        assert is_entry_allowed(et(11, 0), cutoff) is True

    def test_force_close_not_triggered_before_cutoff(self):
        cutoff = parse_hhmm("15:15")
        assert is_force_close_time(et(15, 14), cutoff) is False

    def test_force_close_triggered_at_and_after_cutoff(self):
        cutoff = parse_hhmm("15:15")
        assert is_force_close_time(et(15, 15), cutoff) is True
        assert is_force_close_time(et(15, 59), cutoff) is True


# ---------------------------------------------------------------------------
# VIX sizing
# ---------------------------------------------------------------------------

class TestVixSizing:
    def test_full_size_below_25(self):
        sizing = classify_vix(24.9)
        assert sizing.tier == VixTier.FULL
        assert sizing.size_multiplier == 1.0
        assert sizing.allow_0dte is True
        assert sizing.min_dte == 0

    def test_half_size_between_25_and_30(self):
        sizing = classify_vix(27.0, half_size_multiplier=0.5)
        assert sizing.tier == VixTier.HALF
        assert sizing.size_multiplier == 0.5
        assert sizing.allow_0dte is False
        assert sizing.min_dte == 1
        assert sizing.max_dte == 3

    def test_boundary_25_is_half_size_not_full(self):
        sizing = classify_vix(25.0)
        assert sizing.tier == VixTier.HALF

    def test_boundary_30_is_extreme_not_half(self):
        sizing = classify_vix(30.0)
        assert sizing.tier == VixTier.EXTREME

    def test_no_0dte_above_30(self):
        sizing = classify_vix(31.0)
        assert sizing.tier == VixTier.EXTREME
        assert sizing.allow_0dte is False
        assert sizing.min_dte >= 1

    def test_extreme_still_sized_down(self):
        sizing = classify_vix(35.0, half_size_multiplier=0.5)
        assert sizing.size_multiplier == 0.5


# ---------------------------------------------------------------------------
# BTC gate
# ---------------------------------------------------------------------------

class TestBtcGate:
    def test_mstr_blocked_below_threshold(self):
        assert btc_gate_allows("MSTR", 70999.0, min_price_usd=71000) is False

    def test_coin_allowed_at_threshold(self):
        assert btc_gate_allows("COIN", 71000.0, min_price_usd=71000) is True

    def test_coin_allowed_above_threshold(self):
        assert btc_gate_allows("COIN", 80000.0, min_price_usd=71000) is True

    def test_ungated_symbol_always_allowed(self):
        assert btc_gate_allows("AAPL", 10.0, min_price_usd=71000) is True

    def test_case_insensitive_gated_symbol_match(self):
        assert btc_gate_allows("mstr", 1000.0, min_price_usd=71000) is False


# ---------------------------------------------------------------------------
# Session trade limits & daily stop
# ---------------------------------------------------------------------------

class TestSessionRiskState:
    def test_allows_trades_up_to_max(self):
        state = SessionRiskState(max_trades_per_session=3)
        for _ in range(3):
            allowed, _ = state.can_open_new_trade()
            assert allowed is True
            state.record_trade_opened()
        allowed, reason = state.can_open_new_trade()
        assert allowed is False
        assert reason == "max_trades_per_session_reached"

    def test_daily_profit_target_stops_trading(self):
        state = SessionRiskState(daily_profit_target=500.0, daily_max_loss=500.0)
        state.record_realized_pnl(500.0)
        allowed, reason = state.can_open_new_trade()
        assert allowed is False
        assert reason == "daily_profit_target_hit"

    def test_daily_max_loss_stops_trading(self):
        state = SessionRiskState(daily_profit_target=500.0, daily_max_loss=500.0)
        state.record_realized_pnl(-500.0)
        allowed, reason = state.can_open_new_trade()
        assert allowed is False
        assert reason == "daily_max_loss_hit"

    def test_stop_is_a_one_way_latch_for_the_day(self):
        state = SessionRiskState(daily_profit_target=500.0, daily_max_loss=500.0)
        state.record_realized_pnl(500.0)
        # A subsequent loss brings realized P&L back under the target, but
        # the bot must stay stopped for the rest of the day.
        state.record_realized_pnl(-100.0)
        allowed, reason = state.can_open_new_trade()
        assert allowed is False
        assert reason == "daily_profit_target_hit"

    def test_reset_for_new_session_clears_state(self):
        state = SessionRiskState(daily_profit_target=500.0, daily_max_loss=500.0, max_trades_per_session=1)
        state.record_trade_opened()
        state.record_realized_pnl(500.0)
        state.reset_for_new_session()
        allowed, reason = state.can_open_new_trade()
        assert allowed is True
        assert reason is None
        assert state.trades_taken == 0
        assert state.realized_pnl == 0.0

    def test_under_target_and_under_max_loss_allows_trading(self):
        state = SessionRiskState(daily_profit_target=500.0, daily_max_loss=500.0)
        state.record_realized_pnl(100.0)
        allowed, reason = state.can_open_new_trade()
        assert allowed is True
        assert reason is None


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_basic_sizing(self):
        # $25,000 equity, 1% risk = $250 risk budget, contract premium $2.00
        # -> contract cost = $200 -> 1 contract fits.
        contracts = calc_contracts(
            account_equity=25000, risk_pct_per_trade=1.0, vix_size_multiplier=1.0,
            option_price_per_share=2.0, max_contracts_per_trade=10,
        )
        assert contracts == 1

    def test_half_size_multiplier_reduces_contracts(self):
        full = calc_contracts(25000, 2.0, 1.0, 1.0, 10)
        half = calc_contracts(25000, 2.0, 0.5, 1.0, 10)
        assert half < full

    def test_capped_at_max_contracts(self):
        contracts = calc_contracts(
            account_equity=1_000_000, risk_pct_per_trade=50.0, vix_size_multiplier=1.0,
            option_price_per_share=0.10, max_contracts_per_trade=5,
        )
        assert contracts == 5

    def test_zero_when_premium_too_expensive_for_budget(self):
        contracts = calc_contracts(
            account_equity=1000, risk_pct_per_trade=1.0, vix_size_multiplier=1.0,
            option_price_per_share=50.0, max_contracts_per_trade=10,
        )
        assert contracts == 0

    def test_zero_for_non_positive_premium(self):
        assert calc_contracts(25000, 1.0, 1.0, 0.0, 10) == 0
        assert calc_contracts(25000, 1.0, 1.0, -1.0, 10) == 0


# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------

class TestExitRules:
    def test_stop_loss_triggers(self):
        entry_time = et(10, 0)
        reason = should_exit_position(
            entry_price=1.00, current_price=0.55, entry_time=entry_time, now=et(10, 5),
            stop_loss_pct=40.0, profit_target_pct=60.0, time_stop_minutes=120,
        )
        assert reason == "stop_loss"

    def test_profit_target_triggers(self):
        entry_time = et(10, 0)
        reason = should_exit_position(
            entry_price=1.00, current_price=1.65, entry_time=entry_time, now=et(10, 5),
            stop_loss_pct=40.0, profit_target_pct=60.0, time_stop_minutes=120,
        )
        assert reason == "profit_target"

    def test_time_stop_triggers_after_duration(self):
        entry_time = et(10, 0)
        reason = should_exit_position(
            entry_price=1.00, current_price=1.05, entry_time=entry_time, now=et(12, 1),
            stop_loss_pct=40.0, profit_target_pct=60.0, time_stop_minutes=120,
        )
        assert reason == "time_stop"

    def test_no_exit_when_nothing_triggered(self):
        entry_time = et(10, 0)
        reason = should_exit_position(
            entry_price=1.00, current_price=1.05, entry_time=entry_time, now=et(10, 30),
            stop_loss_pct=40.0, profit_target_pct=60.0, time_stop_minutes=120,
        )
        assert reason is None

    def test_stop_loss_takes_priority_over_time_stop(self):
        entry_time = et(10, 0)
        reason = should_exit_position(
            entry_price=1.00, current_price=0.50, entry_time=entry_time, now=et(12, 30),
            stop_loss_pct=40.0, profit_target_pct=60.0, time_stop_minutes=120,
        )
        assert reason == "stop_loss"
