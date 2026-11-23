"""Regression tests for user-facing bugs found in the pre-release review (B1-B17)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as run_config
import data_handling as dh
import search_space
from nan_imputer import NaNImputer
from output_paths import append_lines_locked, check_csv_header
from predictor_main import build_parser, clean_args, results_header, tuning_settings


def gappy_file(tmp_path, n=300):
	rng = np.random.default_rng(0)
	df = pd.DataFrame({
		"Date": pd.date_range("2021-01-01", periods=n, freq="h"),
		"rain": rng.gamma(1.0, size=n),
		"x_Label": 10 + np.sin(np.arange(n) / 8) + rng.normal(0, 0.1, n),
	})
	df.loc[[20, 21, 150, 260], "x_Label"] = np.nan
	path = tmp_path / "d.csv"
	df.to_csv(path, index=False)
	return str(path)


def read(path, **kw):
	return dh.readData("t", path, label_tag="_Label", input_seq_length=6, output_seq_length=2,
		batch_size=8, transform_method="minmax", **kw)


def parse(argv):
	parser = build_parser()
	run_config.apply_config(parser, argv)
	args = parser.parse_args(argv)
	clean_args(parser, args)
	return args


def test_b2_oversample_low_thresholds_alone(tmp_path):
	w = read(gappy_file(tmp_path), nan_imputation_method="min_std", oversample_low_thresholds=[0.1])
	assert w.num_train_samples > 210 - 8 + 1      # windows were added


def test_b3_interpolate_time(tmp_path):
	w = read(gappy_file(tmp_path), nan_imputation_method="interpolate_time")
	assert np.isfinite(w.train_df.to_numpy()).all()
	df = pd.DataFrame({"Date": pd.date_range("2021-01-01", periods=5, freq="h"), "a": [1.0, np.nan, 3.0, np.nan, 5.0]})
	out = NaNImputer(method="interpolate_time", time_index_column="Date").fit_transform(df)
	assert list(out.columns) == ["Date", "a"] and out["a"].tolist() == [1, 2, 3, 4, 5]


def test_b4_revert_label_predictions_without_scaling():
	frames = [pd.DataFrame({"rain": np.ones(20), "x_Flow": np.arange(20.0)}) for _ in range(3)]
	w = dh.WindowGenerator("t", 3, 2, 0, *frames, label_columns=["x_Flow"], train_nan_indices={},
		val_nan_indices={}, test_nan_indices={}, transform_method="none")
	p = np.arange(8.0).reshape(4, 2, 1)
	np.testing.assert_array_equal(w.revert_label_predictions(p), p)


def test_b1_tuning_settings_change_with_the_data(tmp_path):
	base = ["data.parquet", "exp", "--input-length", "24", "--output-length", "6"]
	a = tuning_settings(parse(base), [{"name": "num_layers", "type": "int", "low": 1, "high": 2}])
	b = tuning_settings(parse(["other.parquet"] + base[1:]), [{"name": "num_layers", "type": "int", "low": 1, "high": 2}])
	assert a != b and a["filename"] == "data.parquet"
	assert a == tuning_settings(parse(base), [{"name": "num_layers", "type": "int", "low": 1, "high": 2}])


def test_b6_results_header_checked_before_training(tmp_path):
	base = ["data.parquet", "exp", "--input-length", "24", "--output-length", "6"]
	path = str(tmp_path / "results.csv")
	header = results_header(parse(base))[2]
	append_lines_locked(path, ["x\n"], header=header)
	check_csv_header(path, header)                                        # same columns: fine
	with pytest.raises(ValueError, match="other columns"):
		check_csv_header(path, results_header(parse(base + ["--metrics", "all"]))[2])


def test_b7_ratios_that_sum_to_one_in_floating_point():
	args = parse(["data.parquet", "exp", "--input-length", "24", "--output-length", "6",
		"--train-ratio", "0.7", "--val-ratio", "0.1", "--test-ratio", "0.2"])
	assert args.train_ratio == 0.7
	with pytest.raises(SystemExit):
		parse(["data.parquet", "exp", "--input-length", "24", "--output-length", "6",
			"--train-ratio", "0.7", "--val-ratio", "0.2", "--test-ratio", "0.2"])


def test_b8_only_if_only_on_optional_settings():
	ok = [{"name": "use_noise", "type": "categorical", "choices": [True, False]},
		{"name": "noise", "type": "float", "low": 0.001, "high": 0.01, "only_if": "use_noise"}]
	search_space.load_search_space(ok)
	assert search_space.best_settings({"use_noise": False}, ok)["noise"] == 0.0
	bad = [{"name": "use_layers", "type": "categorical", "choices": [True, False]},
		{"name": "num_layers", "type": "int", "low": 1, "high": 3, "only_if": "use_layers"}]
	with pytest.raises(ValueError, match="only_if can only switch off"):
		search_space.load_search_space(bad)


def test_b9_yaml_scientific_notation_bounds():
	entries = yaml.safe_load("[{name: l2_regularization, type: float, low: 1e-10, high: 1e-5, log: true}]")
	assert isinstance(entries[0]["low"], str)                            # what PyYAML gives
	loaded = search_space.load_search_space(entries)
	assert loaded[0]["low"] == 1e-10 and loaded[0]["high"] == 1e-5
	with pytest.raises(ValueError, match="must be numbers"):
		search_space.load_search_space([{"name": "l2_regularization", "type": "float", "low": "abc", "high": 1}])


def test_kge_loss_runs_on_batches():
	import tensorflow as tf
	from metrics import kge_loss
	y = tf.constant(np.linspace(1, 2, 12).reshape(2, 6, 1), tf.float32)
	assert float(kge_loss(y, y)) == pytest.approx(0.0, abs=1e-5)


def test_peak_flow_error_averages_over_observed_peaks():
	from metrics import peak_flow_error
	obs = np.arange(1.0, 101.0)
	sim = obs + 1.0
	sim[-1] = 90.0                                  # 10 too low at the largest peak
	peaks = obs >= np.percentile(obs, 95, method="nearest")
	assert peak_flow_error(obs, sim, 95) == pytest.approx(np.mean(np.abs(sim - obs)[peaks]))


def test_change_is_the_default_target():
	args = parse(["data.parquet", "exp", "--input-length", "24", "--output-length", "6"])
	assert args.predict == "change"
