"""Calendar features (--time-features) and the seasonal-naive reference forecast."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_handling as dh


def write(tmp_path, freq="h", n=600, signal=None):
	dates = pd.date_range("2021-03-01", periods=n, freq=freq)          # a Monday
	rng = np.random.default_rng(0)
	y = signal if signal is not None else 10 + np.sin(np.arange(n) / 8) + rng.normal(0, 0.1, n)
	df = pd.DataFrame({"Date": dates, "rain": rng.gamma(1.0, size=n), "x_Label": y})
	path = tmp_path / "d.csv"
	df.to_csv(path, index=False)
	return str(path)


def read(path, **kw):
	return dh.readData("t", path, label_tag="_Label", input_seq_length=24, output_seq_length=6,
		batch_size=8, nan_imputation_method="min_std", transform_method="minmax", **kw)


def time_cols(w):
	return [c for c in w.input_columns if c in dh.TIME_FEATURE_COLUMNS]


def test_auto_is_the_previous_feature_set():
	assert dh.resolve_time_features("auto", True, 20) == ["time_of_day", "day_of_year", "trend"]
	assert dh.resolve_time_features("auto", False, 20) == ["day_of_year", "trend"]
	assert dh.resolve_time_features("auto", True, 0) == ["time_of_day", "day_of_year"]
	assert dh.resolve_time_features("none", True, 20) == []


def test_selected_features_and_day_of_week_values(tmp_path):
	w = read(write(tmp_path), time_features="day_of_week,time_of_day")
	assert time_cols(w) == ["Time sin", "Time cos", "Week sin", "Week cos"]
	df = pd.read_csv(write(tmp_path), parse_dates=["Date"])
	pos = (df["Date"].dt.day_of_week + df["Date"].dt.hour / 24) / 7
	assert pos.iloc[0] == 0.0 and pos.iloc[24] == pytest.approx(1 / 7)    # Monday 00:00, Tuesday 00:00


def test_daily_data_skip_time_of_day_and_trend_needs_a_period(tmp_path):
	w = read(write(tmp_path, freq="D", n=400), time_features="time_of_day,day_of_week")
	assert time_cols(w) == ["Week sin", "Week cos"]
	with pytest.raises(ValueError, match="trend-period-years"):
		dh.resolve_time_features("trend", True, 0)
	with pytest.raises(ValueError, match="unknown calendar feature"):
		dh.resolve_time_features("weekday", True, 20)


def test_seasonal_naive_is_exact_for_a_periodic_series(tmp_path):
	n, period = 600, 24
	signal = 10 + np.sin(2 * np.pi * np.arange(n) / period)                  # repeats every 24 steps
	w = read(write(tmp_path, n=n, signal=signal))
	observed, _ = w.observed_test_windows()
	seasonal = w.seasonal_naive_windows(period)
	ok = ~np.isnan(seasonal)
	assert ok.mean() > 0.9
	np.testing.assert_allclose(seasonal[ok], observed[ok], atol=1e-9)


def test_seasonal_naive_only_uses_past_observations(tmp_path):
	w = read(write(tmp_path))
	period = 4                                                                # shorter than the horizon (6)
	seasonal = w.seasonal_naive_windows(period)
	raw = w.raw_test_labels[:, 0]
	i, h = 10, 5                                                               # window 10, lead time 6
	target = i + w.label_start + h
	last_input = i + w.input_steps - 1
	k = int(np.ceil((target - last_input) / period))
	assert target - k * period <= last_input
	assert seasonal[i, h, 0] == raw[target - k * period]
