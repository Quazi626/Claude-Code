from datetime import date, datetime, time as dtime

import pytz

from flux.backtest.engine import _t_years_remaining

ET = pytz.timezone("America/New_York")


def et(hour, minute, day=15):
    return ET.localize(datetime(2025, 9, day, hour, minute))


class TestTimeToExpiryRemaining:
    def test_0dte_at_midday_has_nonzero_time_value(self):
        # Regression: using whole calendar days (expiration - day).days gives
        # exactly 0 for a same-day contract, collapsing Black-Scholes to pure
        # intrinsic value. There are real hours left until the close.
        t_years = _t_years_remaining(et(12, 0), date(2025, 9, 15), dtime(16, 0))
        assert t_years > 0

    def test_0dte_shrinks_toward_zero_near_the_close(self):
        midday = _t_years_remaining(et(10, 0), date(2025, 9, 15), dtime(16, 0))
        near_close = _t_years_remaining(et(15, 55), date(2025, 9, 15), dtime(16, 0))
        assert 0 <= near_close < midday

    def test_at_or_after_close_is_zero(self):
        assert _t_years_remaining(et(16, 0), date(2025, 9, 15), dtime(16, 0)) == 0.0
        assert _t_years_remaining(et(16, 30), date(2025, 9, 15), dtime(16, 0)) == 0.0

    def test_multi_day_expiry_includes_full_intervening_days(self):
        one_day_out = _t_years_remaining(et(12, 0), date(2025, 9, 16), dtime(16, 0))
        same_day = _t_years_remaining(et(12, 0), date(2025, 9, 15), dtime(16, 0))
        assert one_day_out > same_day
