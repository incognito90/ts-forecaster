"""YAML config files: precedence, aliases, validation and round trip."""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as run_config
from predictor_main import build_parser, clean_args


def parse(argv):
	parser = build_parser()
	run_config.apply_config(parser, argv)
	args = parser.parse_args(argv)
	clean_args(parser, args)
	return args


def write(tmp_path, cfg, name="cfg.yaml"):
	path = tmp_path / name
	path.write_text(yaml.safe_dump(cfg))
	return str(path)


BASE = {"run": {"name": "exp"}, "data": {"file": "data.parquet", "input_length": 48, "output_length": 12}}


def test_precedence_default_config_cli(tmp_path):
	path = write(tmp_path, {**BASE, "model": {"units_per_layer": 64, "decoder": True}, "training": {"batch_size": 16}})
	args = parse(["--config", path, "--batch-size", "128"])
	assert (args.filename, args.testname) == ("data.parquet", "exp")      # aliases data.file, run.name
	assert args.input_length == 48 and args.units_per_layer == 64         # from config (typed)
	assert args.decoder is True                                           # YAML bool handled like 'True'
	assert args.batch_size == 128                                         # command line wins
	assert args.dropout_rate == build_parser().get_default("dropout_rate")  # built-in default


def test_positional_arguments_override_config(tmp_path):
	path = write(tmp_path, BASE)
	args = parse(["other.parquet", "other_name", "--config", path])
	assert (args.filename, args.testname) == ("other.parquet", "other_name")


def test_lists_none_and_tuning_aliases(tmp_path):
	cfg = {**BASE, "preprocessing": {"oversample_high_thresholds": [0.9, 0.95]},
		"model": {"l1_regularization": 1e-6, "l2_regularization": None},
		"tuning": {"trials": 20, "repeats": 3}}
	args = parse(["--config", write(tmp_path, cfg)])
	assert args.oversample_high_thresholds == [0.9, 0.95]
	assert args.l1_regularization == 1e-6 and args.l2_regularization is None
	assert args.hyperparameter_optimization == 20 and args.hyperparameter_optimization_repeat == 3


def test_inline_search_space(tmp_path):
	space = [{"name": "num_layers", "type": "int", "low": 1, "high": 2}]
	args = parse(["--config", write(tmp_path, {**BASE, "tuning": {"trials": 2, "search_space": space}})])
	assert args.search_space == space


@pytest.mark.parametrize("cfg, message", [
	({"dataa": {"file": "x"}}, "did you mean 'data'"),
	({"data": {"input_lenght": 3}}, "did you mean 'input_length'"),
	({"data": {"units_per_layer": 3}}, "belongs in section 'model'"),
])
def test_unknown_or_misplaced_keys_are_errors(tmp_path, cfg, message):
	with pytest.raises(ValueError, match=message):
		run_config.read_config(write(tmp_path, cfg))


def test_written_config_reproduces_the_run_settings(tmp_path):
	cfg = {**BASE, "model": {"units_per_layer": 64}, "preprocessing": {"oversample_high_thresholds": [0.9]},
		"training": {"learning_rate": -1, "half_float": False}}
	args = parse(["--config", write(tmp_path, cfg), "--dropout-rate", "0.3"])
	out = run_config.write_config(args, str(tmp_path / "used.yaml"), header="test")
	again = parse(["--config", out])
	keep = [d for dests in run_config.SECTIONS.values() for d in dests]
	assert {k: getattr(args, k) for k in keep} == {k: getattr(again, k) for k in keep}


def test_defaults_min_std_and_automatic_precision(tmp_path):
	import tensorflow as tf
	args = parse(["--config", write(tmp_path, BASE)])
	assert args.nan_fill == "min_std"
	expected = "True" if tf.config.list_physical_devices("GPU") else "False"
	assert args.half_float == expected                     # auto: mixed precision only with a GPU
	assert parse(["--config", write(tmp_path, BASE), "--half-float", "True"]).half_float == "True"


def test_written_config_keeps_the_explain_section(tmp_path):
	explain = {"run_shap": True, "member_group": "dam", "feature_groups": {"dam": ["_dam"], "river": ["_river"]}}
	args = parse(["--config", write(tmp_path, {**BASE, "explain": explain})])
	out = run_config.write_config(args, str(tmp_path / "used.yaml"))
	with open(out) as f:
		assert yaml.safe_load(f)["explain"] == explain
