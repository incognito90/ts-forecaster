"""Signed-log change input features (--change-lags, --change-columns)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_handling as dh


def frame(n=40, seed=0):
	rng = np.random.default_rng(seed)
	return pd.DataFrame({
		"Date": pd.date_range("2021-01-01", periods=n, freq="h"),
		"stage": rng.normal(2, 0.1, n),
		"x_Label": 10 + np.cumsum(rng.normal(0, 1, n)),
	})


def expected(series, lag, num_train):
	s = (series - series.shift(1)).iloc[:num_train].abs().median()
	d = series - series.shift(lag)
	return np.sign(d) * np.log1p(d.abs() / s)


def test_values_names_and_training_scale():
	df = frame()
	out, names = dh.add_change_features(df.copy(), ["x_Label"], "_Label", [1, 3], [], num_train=28)
	assert names == ["x_d1_change", "x_d3_change"]
	for k in (1, 3):
		np.testing.assert_allclose(out[f"x_d{k}_change"], expected(df["x_Label"], k, 28), equal_nan=True)
	# the scale comes from the training rows only
	later = df.copy()
	later.loc[30:, "x_Label"] *= 50
	out2, _ = dh.add_change_features(later, ["x_Label"], "_Label", [1], [], num_train=28)
	np.testing.assert_allclose(out2["x_d1_change"].iloc[:28], out["x_d1_change"].iloc[:28], equal_nan=True)


def test_causal_and_missing_values():
	df = frame()
	df.loc[10, "x_Label"] = np.nan
	out, _ = dh.add_change_features(df.copy(), ["x_Label"], "_Label", [1, 3], [], num_train=28)
	assert out["x_d1_change"].iloc[[10, 11]].isna().all()       # x_10 missing: changes at 10 and 11
	assert out["x_d3_change"].iloc[[10, 13]].isna().all()
	assert out["x_d1_change"].iloc[12:].notna().all()
	future = df.copy()
	future.loc[30:, "x_Label"] += 100                          # a later change (after training) does not alter earlier features
	out2, _ = dh.add_change_features(future, ["x_Label"], "_Label", [1, 3], [], num_train=28)
	np.testing.assert_allclose(out2["x_d3_change"].iloc[:30], out["x_d3_change"].iloc[:30], equal_nan=True)


def test_extra_columns_get_one_step_change():
	df = frame()
	out, names = dh.add_change_features(df.copy(), ["x_Label"], "_Label", [1], ["^sta"], num_train=28)
	assert names == ["x_d1_change", "stage_d1_change"]
	np.testing.assert_allclose(out["stage_d1_change"], expected(df["stage"], 1, 28), equal_nan=True)


def test_no_lags_means_no_change_features():
	df = frame()
	out, names = dh.add_change_features(df.copy(), ["x_Label"], "_Label", [], ["^sta"], num_train=28)
	assert names == [] and list(out.columns) == list(df.columns)


def test_constant_series_does_not_divide_by_zero():
	df = frame()
	df["x_Label"] = 5.0
	df.loc[30, "x_Label"] = 6.0
	out, _ = dh.add_change_features(df, ["x_Label"], "_Label", [1], [], num_train=28)
	assert np.isfinite(out["x_d1_change"].iloc[1:]).all()


@pytest.mark.parametrize("spec, lags", [("1,3,6", [1, 3, 6]), ([6, 1], [1, 6]), ("none", []), ("", []), (None, []), ("2", [2])])
def test_parse_change_lags(spec, lags):
	assert dh.parse_change_lags(spec) == lags


@pytest.mark.parametrize("spec", ["0", "a", "1,-3"])
def test_parse_change_lags_rejects(spec):
	with pytest.raises(ValueError):
		dh.parse_change_lags(spec)


def read(path, **kw):
	return dh.readData("t", path, label_tag="_Label", input_seq_length=6, output_seq_length=2,
		batch_size=8, nan_imputation_method="min_std", transform_method="minmax", **kw)


def test_read_data_default_adds_inputs_not_targets(tmp_path):
	path = tmp_path / "d.csv"
	frame(200).to_csv(path, index=False)
	w = read(str(path))
	assert w.label_columns == ["x_Label"]
	assert {"x_d1_change", "x_d3_change", "x_d6_change"} <= set(w.input_columns)
	off = read(str(path), change_lags="none")
	assert not [c for c in off.input_columns if c.endswith("_change")]


def test_grws_flow_and_stage_changes():
	"""Real GRWS data: flow changes over 1/3/6 h and the 1 h stage change, as used in testing."""
	df = pd.read_parquet(os.path.join(ROOT, "data", "grws", "Parquet", "Leggatt_closest.parquet"))
	q = [c for c in df if c.endswith("_Flow")][0]
	st = [c for c in df if "-HG-" in c][0]
	ntr = int(len(df) * 0.7)
	out, _ = dh.add_change_features(df.copy(), [q], "_Flow", [1, 3, 6], ["-HG-"], ntr)
	base = q.replace("_Flow", "")
	for k in (1, 3, 6):
		np.testing.assert_allclose(out[f"{base}_d{k}_change"], expected(df[q], k, ntr), equal_nan=True)
	np.testing.assert_allclose(out[f"{st}_d1_change"], expected(df[st], 1, ntr), equal_nan=True)
