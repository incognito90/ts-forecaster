"""Explanation samples require complete raw target windows, before imputation."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handling import WindowGenerator
import shap_explain as se


@pytest.fixture
def window():
	# Window s: history s:s+3, gap s+3, prediction s+4:s+7.
	w = WindowGenerator.__new__(WindowGenerator)
	w.input_steps, w.shift, w.label_steps, w.label_start, w.total_window_size = 3, 1, 3, 4, 7
	w.num_labels, w.num_features, w.batch_size, w.timestep_seconds = 2, 1, 2, 3600
	w.label_columns = ["a_Label", "b_Label"]
	w.column_indices = {"input": 0}
	w.raw_test_labels = np.arange(30.0).reshape(15, 2)
	w.raw_test_labels[2, 1] = np.nan   # historical missingness: window 0 stays eligible
	w.raw_test_labels[3, 0] = np.nan   # gap missingness: window 0 stays eligible
	w.raw_test_labels[8, 1] = np.nan   # invalidates starts 2, 3, 4, on the OTHER target
	w.raw_test_labels[12, 0] = np.nan  # invalidates starts 6, 7, 8
	observed, _ = w.observed_test_windows()
	x = np.broadcast_to(np.arange(len(observed))[:, None, None], (len(observed), 3, 1)).astype("float32")
	# Labels look fully populated after imputation, so filtering must use raw data.
	y = np.nan_to_num(observed, nan=-1000.0).astype("float32")
	w.test_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
	w.train_ds = w.test_ds
	return w


def ids(x):
	return sorted(x[:, 0, 0].astype(int).tolist())


def test_complete_prediction_window_checks_every_step_and_target(window):
	assert np.flatnonzero(window.complete_test_window_mask()).tolist() == [0, 1, 5]


def test_random_sampling_keeps_mask_aligned_across_rejected_batches(window):
	x, y = se.sample_dataset_windows(window.test, None, 100, 123,
		valid_windows=window.complete_test_window_mask())
	assert ids(x) == [0, 1, 5]
	assert not (y == -1000.0).any()
	# Batch limits still count original batches, even if some are rejected.
	x, _ = se.sample_dataset_windows(window.test, 2, 100, 123,
		valid_windows=window.complete_test_window_mask())
	assert ids(x) == [0, 1]


@pytest.mark.parametrize("sampler, expected", [
	(se.sample_low_flow_test_windows, [0, 1]),
	(se.sample_high_flow_test_windows, [1, 5]),
])
def test_flow_quantiles_exclude_incomplete_windows(window, sampler, expected):
	x, y, rest_x, rest_y, threshold = sampler(window, 100, 0, 0, quantile=0.5)
	assert ids(x) == expected
	assert sorted(ids(x) + ids(rest_x)) == [0, 1, 5]
	assert threshold == 10.0  # median of eligible first-target observations 8, 10, 18
	assert not (y == -1000.0).any() and not (rest_y == -1000.0).any()


@pytest.mark.parametrize("sampler", [se.sample_low_flow_test_windows, se.sample_high_flow_test_windows])
def test_no_complete_forecasts_reports_clear_error(window, sampler):
	window.raw_test_labels[:] = np.nan
	with pytest.raises(ValueError, match="observed targets throughout the prediction window"):
		sampler(window, 10, 0, 0)


def test_random_shap_mode_passes_only_complete_forecasts_to_explainer(window, monkeypatch, tmp_path):
	explained = []
	class Explainer:
		def __init__(self, *args, **kwargs):
			pass
		def shap_values(self, x):
			explained.extend(ids(x))
			return np.zeros_like(x)
	monkeypatch.setattr(se.shap, "GradientExplainer", Explainer)
	monkeypatch.setattr(se, "make_scalar_output_model", lambda *args: None)
	monkeypatch.setattr(se, "create_recent_shap_summary_plot", lambda **kwargs: None)
	args = SimpleNamespace(test_name="test", shap_sample_mode="random", shap_horizon=1, shap_label_index=0,
		shap_background_batches=1, shap_background_size=2, shap_explain_batches=None, shap_explain_size=100, shap_seed=123)
	se.run_shap_analysis(None, window, args, tmp_path)
	assert explained == [0, 1, 5]
