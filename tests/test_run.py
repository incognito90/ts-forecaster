"""run.py: config and command-line precedence; mode dispatch."""
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run


def test_command_line_overrides_config(tmp_path):
	cfg = tmp_path / "cfg.yaml"
	cfg.write_text(yaml.safe_dump({"run": {"name": "exp"}, "data": {"file": "base.parquet", "input_length": 24, "output_length": 6}}))
	s = run.resolve_settings(["--config", str(cfg), "--output-length", "48"])
	assert (s.filename, s.input_length, s.testname) == ("base.parquet", 24, "exp")   # from the config
	assert s.output_length == 48                                                    # command line wins


@pytest.mark.parametrize("mode_from_config", [False, True])
def test_evaluate_dispatch_disables_tuning(tmp_path, monkeypatch, mode_from_config):
	cfg = tmp_path / "tuning.yaml"
	cfg.write_text(yaml.safe_dump({
		"run": {"name": "exp", "mode": "evaluate" if mode_from_config else "tune", "trained_model": "saved_model"},
		"data": {"file": "data.csv"},
		"tuning": {"trials": 10, "search_space": "missing-search-file.json"},
	}))
	argv = ["run.py", "--config", str(cfg)] + ([] if mode_from_config else ["--mode", "evaluate"])
	monkeypatch.setattr(sys, "argv", argv)
	commands = []
	monkeypatch.setattr(run, "run", commands.append)
	run.main()
	assert len(commands) == 1 and commands[0][1] == run.MAIN
	s = run.resolve_settings(commands[0][2:])
	assert s.train == "False" and s.hyperparameter_optimization is None
	assert s.trainedModelFile == "saved_model"


@pytest.mark.parametrize("options", [["--mode", "evaluate"], ["--train", "False"]])
def test_direct_evaluation_disables_search_before_validation(options):
	from predictor_main import build_parser, clean_args
	parser = build_parser()
	args = parser.parse_args(["data.csv", "exp", "--trainedModelFile", "saved_model",
		"--hyperparameter-optimization", "10", "--search-space", "missing-search-file.json",
		"--half-float", "False"] + options)
	clean_args(parser, args)
	assert args.train is False and args.hyperparameter_optimization is None
