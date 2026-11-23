"""Tests for search_space: file validation and conversion of trials to run settings."""
import json
import os
import sys

import optuna
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import search_space

DEFAULT_FILE = os.path.join(ROOT, "configs", "optuna_search_space.json")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_switched_off_settings_are_none():
	entries = search_space.load_search_space(DEFAULT_FILE)
	trial = optuna.trial.FixedTrial({
		"num_layers": 2, "n_units_exponent": 7, "dropout_rate": 0.1, "use_input_dropout": False, "loss": "mse",
		"use_l1": False, "use_l2": True, "l2_regularization": 1e-6,
		"use_recurrent_l1_regularization": False, "use_recurrent_l2_regularization": False,
		"use_dense_l1_regularization": False, "use_dense_l2_regularization": False,
		"batch_norm": False, "use_noise": False, "patience": 5,
		"set_learning_rate": False, "use_gradient_clip": False})
	values = search_space.sample(trial, entries)
	assert values["l1_regularization"] is None
	assert values["l2_regularization"] == 1e-6
	assert values["learning_rate"] is None and values["gradient_clip"] is None
	assert values["noise"] is None and values["input_dropout"] is None


@pytest.mark.parametrize("params, message", [
	([{"name": "not_a_setting", "type": "int", "low": 1, "high": 2}], "unknown name"),
	([{"name": "num_layers", "type": "int", "low": 3, "high": 2}], "low"),
	([{"name": "loss", "type": "categorical", "choices": []}], "choices"),
	([{"name": "noise", "type": "uniform", "low": 0, "high": 1}], "type must be"),
	([{"name": "noise", "type": "float", "low": 0, "high": 1, "only_if": "use_noise"}], "only_if"),
	([{"name": "noise", "type": "float", "low": 0, "high": 1}] * 2, "defined twice"),
])
def test_invalid_files_are_rejected(tmp_path, params, message):
	path = tmp_path / "space.json"
	path.write_text(json.dumps({"parameters": params}))
	with pytest.raises(ValueError, match=message):
		search_space.load_search_space(str(path))


def test_best_settings_map_trial_parameters_to_run_settings():
	entries = search_space.load_search_space(DEFAULT_FILE)
	params = {"num_layers": 3, "n_units_exponent": 7, "dropout_rate": 0.2, "loss": "huber",
		"use_l1": False, "use_l2": True, "l2_regularization": 1e-6,
		"use_recurrent_l1_regularization": True, "recurrent_l1_regularization": 2e-7,
		"use_recurrent_l2_regularization": False, "use_dense_l1_regularization": False,
		"use_dense_l2_regularization": False, "batch_norm": True, "noise": 0.01, "patience": 7,
		"set_learning_rate": False, "use_gradient_clip": True, "gradient_clip": 2.5}
	s = search_space.best_settings(params, entries)
	assert s["units_per_layer"] == 128 and s["num_layers"] == 3 and s["loss_function"] == "huber"
	assert s["l1_regularization"] is None and s["l2_regularization"] == 1e-6
	assert s["l1_recurrent_regularization"] == 2e-7 and s["l2_recurrent_regularization"] is None
	assert s["batch_normalization"] is True and s["learning_rate"] is None and s["gradient_clip"] == 2.5
	assert not any(k.startswith(("use_", "set_")) for k in s)


def test_best_settings_reproduce_what_the_objective_samples():
	"""Settings from best_settings must equal the values objective() derives from the same trial."""
	entries = search_space.load_search_space(DEFAULT_FILE)
	trial = optuna.trial.FixedTrial({"num_layers": 2, "n_units_exponent": 6, "dropout_rate": 0.1,
		"use_input_dropout": False, "loss": "mse", "use_l1": True, "l1_regularization": 3e-7, "use_l2": False, "use_recurrent_l1_regularization": False,
		"use_recurrent_l2_regularization": False, "use_dense_l1_regularization": False,
		"use_dense_l2_regularization": False, "batch_norm": False, "use_noise": False, "patience": 5,
		"set_learning_rate": True, "learning_rate": 1e-3, "use_gradient_clip": False})
	sampled = search_space.sample(trial, entries)
	s = search_space.best_settings(trial.params, entries)
	assert s["units_per_layer"] == 2 ** sampled["n_units_exponent"]
	for name, dest in [("dropout_rate", "dropout_rate"), ("l1_regularization", "l1_regularization"),
			("learning_rate", "learning_rate"), ("gradient_clip", "gradient_clip"), ("patience", "patience")]:
		assert s[dest] == sampled[name]
