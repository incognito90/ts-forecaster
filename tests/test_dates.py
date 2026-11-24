"""readData Date handling: unsorted rows are sorted, the time step is inferred, duplicates stop the run."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_handling as dh


def write(tmp_path, dates, name="d.parquet"):
	n = len(dates)
	rng = np.random.default_rng(0)
	df = pd.DataFrame({"Date": dates, "rain": rng.gamma(1.0, size=n), "x_Label": 10 + np.sin(np.arange(n) / 10) + rng.normal(0, 0.1, n)})
	path = tmp_path / name
	df.to_parquet(path, engine="pyarrow")
	return str(path), df


def read(path, **kw):
	return dh.readData("t", path, label_tag="_Label", input_seq_length=6, output_seq_length=2,
		batch_size=8, nan_imputation_method="ffill", transform_method="minmax", **kw)


def test_unsorted_hourly_rows_are_sorted(tmp_path):
	dates = pd.date_range("2020-01-01", periods=400, freq="h")
	path, df = write(tmp_path, dates[::-1])        # reversed order
	w = read(path)
	assert w.timestep_seconds == 3600
	assert "Time sin" in w.input_columns            # sub-daily data get time-of-day features
	observed, _ = w.observed_test_windows()
	expected = df.sort_values("Date")["x_Label"].to_numpy()
	n_test = len(expected) - int(400 * 0.7) - int(400 * 0.15)
	assert np.allclose(observed[0, :, 0], expected[-n_test:][6:8])


def test_daily_step_inferred(tmp_path):
	path, _ = write(tmp_path, pd.date_range("2000-01-01", periods=400, freq="D").strftime("%Y-%m-%d"))
	w = read(path)
	assert w.timestep_seconds == 86400
	assert "Time sin" not in w.input_columns


def test_duplicate_dates_stop_the_run(tmp_path):
	dates = list(pd.date_range("2020-01-01", periods=399, freq="h"))
	path, _ = write(tmp_path, dates + [dates[10]])
	with pytest.raises(SystemExit, match="duplicate"):
		read(path)


def test_unreadable_dates_give_a_clear_error(tmp_path):
	path, _ = write(tmp_path, ["01/01/2000", "13/01/2000"] + ["not a date"] * 398)
	with pytest.raises(SystemExit, match="ISO dates"):
		read(path)
