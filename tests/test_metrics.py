"""Evaluation metrics: correct argument order, observations from raw data, baselines."""
import os
import sys

import HydroErr as he
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handling import WindowGenerator
import evaluation
from evaluation import compute_metrics, resolve_metric_groups
from predictor_main import csv_line

ALL = resolve_metric_groups("all")

rng = np.random.default_rng(1)
OBS = rng.gamma(2.0, 20.0, size=3000)
SIM = 0.8 * OBS + rng.normal(0, 3, size=OBS.size)      # under-predicts volume and variability


def scores_of(o, s, naive_mae=np.nan):
	return compute_metrics(o, s, ALL, naive_mae)[0]


def col(scores, name):
	return scores[name]


def test_kge_and_components_match_hydroerr():
	scores = scores_of(OBS, SIM)
	r, alpha, beta, kge = he.kge_2009(SIM, OBS, return_all=True)
	assert col(scores, "KGE") == pytest.approx(kge, rel=1e-6)
	assert col(scores, "Pearson Correlation") == pytest.approx(r, rel=1e-6)
	assert col(scores, "alpha") == pytest.approx(alpha, rel=1e-6)      # std(sim) / std(obs)
	assert col(scores, "beta") == pytest.approx(beta, rel=1e-6)        # mean(sim) / mean(obs)
	assert col(scores, "alpha") < 1 and col(scores, "beta") < 1        # under-prediction reads as < 1


def test_volume_and_relative_errors_use_observations():
	scores = scores_of(OBS, SIM)
	assert col(scores, "NSE") == pytest.approx(he.nse(SIM, OBS), rel=1e-9)
	assert col(scores, "model efficiency") == pytest.approx(he.nse(SIM, OBS), rel=1e-6)
	assert col(scores, "pbias") == pytest.approx(100 * (OBS.sum() - SIM.sum()) / OBS.sum(), rel=1e-6)
	assert col(scores, "rve") == pytest.approx((SIM.sum() - OBS.sum()) / OBS.sum(), rel=1e-6)
	assert col(scores, "mape") == pytest.approx(100 * np.mean(np.abs((OBS - SIM) / OBS)), rel=1e-5)


def test_nan_positions_are_skipped():
	obs = OBS.copy()
	obs[::10] = np.nan
	scores = scores_of(obs, SIM)
	keep = ~np.isnan(obs)
	assert col(scores, "NSE") == pytest.approx(he.nse(SIM[keep], obs[keep]), rel=1e-9)


def test_climatology_baseline_scores():
	clim = scores_of(OBS, np.full_like(OBS, OBS.mean()))
	assert clim["NSE"] == pytest.approx(0.0, abs=1e-12)
	assert clim["pbias"] == pytest.approx(0.0, abs=1e-6)


def make_window(transform):
	n = 80
	t = np.arange(n, dtype=float)
	flow = 50 + 10 * np.sin(t / 5)
	flow_raw = flow.copy()
	flow_raw[[30, 61]] = np.nan                         # missing test observations
	frames = [pd.DataFrame({"rain": rng.gamma(1.0, size=k), "x_Flow": flow[a:a + k]}) for a, k in ((0, 40), (40, 10), (50, 30))]
	return WindowGenerator("t", 6, 3, 0, *frames, batch_size=4, label_columns=["x_Flow"],
		train_nan_indices={}, val_nan_indices={}, test_nan_indices={}, transform_method=transform,
		raw_test_labels=flow_raw[50:80, None], train_label_means=np.array([flow[:40].mean()])), flow_raw[50:80]


def test_observed_windows_come_from_raw_data():
	w, raw = make_window("minmax")
	observed, last_observed = w.observed_test_windows()
	n_windows = 30 - (6 + 0 + 3) + 1
	assert observed.shape == (n_windows, 3, 1) and last_observed.shape == (n_windows, 1)
	for i in range(n_windows):
		np.testing.assert_array_equal(observed[i, :, 0], raw[i + 6:i + 9])
		# persistence: the last observed value at or before the last input row
		seen = raw[:i + 6][~np.isnan(raw[:i + 6])]
		expected = seen[-1] if len(seen) else np.nan
		assert np.array_equal(last_observed[i, 0], expected, equal_nan=True)


def test_core_metrics():
	naive = np.mean(np.abs(np.diff(OBS)))
	sc = scores_of(OBS, SIM, naive_mae=naive)
	assert sc["r2"] == pytest.approx(sc["NSE"], rel=1e-9)
	assert sc["bias"] == pytest.approx(np.mean(SIM - OBS))
	assert sc["mase"] == pytest.approx(np.mean(np.abs(SIM - OBS)) / naive, rel=1e-6)
	assert sc["smape"] == pytest.approx(100 * np.mean(2 * np.abs(SIM - OBS) / (np.abs(OBS) + np.abs(SIM))), rel=1e-9)
	assert compute_metrics([0.0, 1.0], [0.0, 1.0], ["smape"])[0]["smape"] == 0.0     # 0/0 counts as no error


def test_metric_group_selection():
	assert resolve_metric_groups("core") == evaluation.METRIC_GROUPS["core"]
	both = resolve_metric_groups("core,hydrology")
	assert both[:len(evaluation.METRIC_GROUPS["core"])] == evaluation.METRIC_GROUPS["core"] and "KGE" in both
	assert resolve_metric_groups(["all"]) == ALL and len(ALL) == len(set(ALL))
	with pytest.raises(ValueError, match="unknown metric group"):
		resolve_metric_groups("core,hydro")


def test_csv_line_keeps_lists_in_one_column():
	import csv, io
	line = csv_line(["run", [0.9, 0.95], None, 1.5])
	assert next(csv.reader(io.StringIO(line))) == ["run", "[0.9, 0.95]", "", "1.5"]
