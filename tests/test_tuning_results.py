"""Trial result rows describe the model that was actually built and its repeat."""
import contextlib
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predictor_main as pm
from output_paths import OutputPaths


def test_objective_records_effective_settings_and_repeat_ids(tmp_path, monkeypatch):
	parser = pm.build_parser()
	base = pm.clean_args(parser, parser.parse_args([
		"data.csv", "tuning", "--half-float", "False", "--test-num", "5",
		"--num-layers", "1", "--units-per-layer", "4", "--l1-regularization", "0.01",
		"--learning-rate", "0.003", "--max-epochs", "1"]))
	monkeypatch.setattr(pm, "args", base, raising=False)
	entries = [
		{"name": "num_layers", "type": "int", "low": 1, "high": 2},
		{"name": "n_units_exponent", "type": "int", "low": 2, "high": 3},
		{"name": "loss", "type": "categorical", "choices": ["mse", "huber"]},
		{"name": "use_l1", "type": "categorical", "choices": [True, False]},
		{"name": "l1_regularization", "type": "float", "low": 0.001, "high": 0.01, "only_if": "use_l1"},
		{"name": "batch_norm", "type": "categorical", "choices": [False, True]},
		{"name": "decoder", "type": "categorical", "choices": [False, True]},
	]
	trial = optuna.trial.FixedTrial({"num_layers": 2, "n_units_exponent": 3,
		"loss": "huber", "use_l1": False, "batch_norm": True, "decoder": True}, number=7)
	observed = np.arange(1.0, 9.0).reshape(4, 2, 1)
	window = SimpleNamespace(num_features=2, num_labels=1, label_columns=["x_Label"],
		label_columns_indices={"x_Label": 0}, test=None,
		train_label_means=np.array([4.0]), train_label_naive_mae=np.array([1.0]),
		observed_test_windows=lambda: (observed, observed[:, 0, :]))
	built, fitted = [], []
	def build(*args, **kwargs):
		built.append((args, kwargs))
		return SimpleNamespace(summary=lambda **_: None, predict=lambda _: observed + 0.1)
	def fit(model, window, **kwargs):
		fitted.append(kwargs)
		return SimpleNamespace(history={"loss": [0.1], "val_loss": [0.2], "val_mse": [0.3]}), 0, 0, 0
	monkeypatch.setattr(pm, "MODEL_BUILDERS", {"lstm": build})
	monkeypatch.setattr(pm, "compile_and_fit", fit)
	paths = OutputPaths(str(tmp_path)).create()
	strategy = SimpleNamespace(scope=contextlib.nullcontext)
	value = pm.objective(strategy, trial, "tuning", "min_std", "standard", 2, 4, 1, 2,
		window, 1, 0, num_repeat=2, paths=paths, search_entries=entries, args=pm.args)
	assert value == 0.3
	with open(paths.results_csv) as f:
		rows = list(csv.DictReader(f))
	assert len(rows) == 4  # two repeats, two horizons each
	for repeat, seed in [(1, 47), (2, 48)]:
		args, kwargs = built[repeat - 1]
		assert args[:2] == (2, 8)
		assert kwargs["l1_regularization"] is None and kwargs["decoder"] is True
		assert fitted[repeat - 1]["loss"] == "huber"
		selected = [r for r in rows if r["Trial Repeat"] == str(repeat)]
		assert len(selected) == 2
		for row in selected:
			assert row["Trial Number"] == "7" and row["Trial Seed"] == str(seed)
			assert row["Test Name"] == f"tuning_Trial-7_Repeat-{repeat}"
			assert (row["num_layers"], row["units_per_layer"], row["loss_function"]) == ("2", "8", "huber")
			assert row["l1_regularization"] == ""  # disabled by the sampled gate
			assert row["decoder"] == row["batch_normalization"] == "True"
			assert row["learning_rate"] == "0.003"  # not searched: retain config value
	assert base.units_per_layer == 4 and base.l1_regularization == 0.01
	with open(tmp_path / "optuna" / "param_search.csv") as f:
		params = list(csv.DictReader(f))
	assert [(r["Trial Number"], r["Trial Repeat"], r["Trial Seed"]) for r in params] == [("7", "1", "47"), ("7", "2", "48")]
	# Ordinary runs share the column schema, with empty trial metadata.
	pm.test(window, observed, observed + 0.1, paths, "ordinary", "min_std", "standard",
		0, 0, 0, 1, 0, base)
	with open(paths.results_csv) as f:
		rows = list(csv.DictReader(f))
	assert rows[-1]["Trial Number"] == rows[-1]["Trial Repeat"] == rows[-1]["Trial Seed"] == ""
