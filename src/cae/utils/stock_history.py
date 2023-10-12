from dataclasses import dataclass
from collections import deque
import datetime
import math


class MovingAverage:
    def __init__(self, duration):
        self._duration = duration
        self._sum = 0
        self._nans = 0
        self.values = deque()

    def add(self, value):
        if math.isnan:
            self._nans += 1
        else:
            self._sum += value
        self.values.append(value)
        if len(self.values) > self._duration:
            popped_value = self.values.popleft()
            if math.isnan(popped_value):
                self._nans -= 1
            else:
                self._sum -= popped_value

    def get(self):
        return (
            self._sum / (len(self.values) - self._nans)
            if len(self.values) - self._nans
            else math.nan
        )


def _coerce_to_float(value):
    try:
        val = float(value)
        return val
    except (ValueError, TypeError):
        return math.nan


def _coerce_to_float_with_factor(value, factor):
    return _coerce_to_float(value) * factor


def _coerce_time(row):
    """Supports iso and YY MM DD format"""
    date_string = row.get("Date", row.get("date"))
    try:
        date = datetime.datetime.strptime((date_string), "%Y-%m-%d")
    except ValueError:
        date = datetime.datetime.fromisoformat(date_string)
    return date


def _calculate_adjustment_factor(row):
    close_value = row.get("Close", row.get("close"))
    adj_close_value = row.get("Adj Close", row.get("adjclose"))
    if close_value in [None, 0] or adj_close_value is None or close_value:
        # We can't calculate the adjustment factor if we don't have the close
        # values to compare.
        # If it's 0, we can't calculate the adjustment factor either (it would wipe
        # everything)
        return 1

    assert not (
        close_value == 0 and adj_close_value != 0
    ), "Close value is 0 but adj close is not"
    # If the close value goes to 0, we can't calculate the adjustment factor either
    return adj_close_value / close_value


@dataclass
class StockRow:
    date: datetime.datetime
    high: float
    low: float
    open: float
    close: float
    volume: float
    moving_averages: ...

    def moving_average_for(self, duration):
        return self.moving_averages.get(duration)

    @property
    def is_infoless(self):
        return all(
            math.isnan(val)
            for val in [
                self.high,
                self.low,
                self.open,
                self.close,
                self.volume,
            ]
        )


def preprocess_rows(rows, moving_average_durations=None):
    if moving_average_durations is None:
        moving_average_durations = []

    moving_averages = {
        duration: MovingAverage(duration) for duration in moving_average_durations
    }

    new_rows = []
    for row in rows:
        date = _coerce_time(row)
        adjustment_factor = _calculate_adjustment_factor(row)
        recorded_high_value = _coerce_to_float_with_factor(
            row.get("High", row.get("high")), adjustment_factor
        )
        recorded_low_value = _coerce_to_float_with_factor(
            row.get("Low", row.get("low")), adjustment_factor
        )
        open_value = _coerce_to_float_with_factor(
            row.get("Open", row.get("open")), adjustment_factor
        )
        close_value = _coerce_to_float_with_factor(
            row.get("Close", row.get("close")), adjustment_factor
        )
        volume_value = _coerce_to_float(row.get("Volume", row.get("volume")))

        # Update moving averages
        for duration in moving_average_durations:
            moving_averages[duration].add(close_value)

        candle_vals = (recorded_low_value, recorded_high_value, close_value, open_value)
        # Stock data might be messy, sometimes highs really aren't highs for the
        # so we recalculate the high and low values ourselves
        high_value = max(
            (val for val in candle_vals if not math.isnan(val)), default=math.nan
        )
        low_value = min(
            (val for val in candle_vals if not math.isnan(val)), default=math.nan
        )

        new_rows.append(
            StockRow(
                date=date,
                high=high_value,
                low=low_value,
                close=close_value,
                open=open_value,
                volume=volume_value,
                moving_averages={
                    duration: moving_averages[duration].get()
                    for duration in moving_average_durations
                },
            )
        )
    return new_rows
