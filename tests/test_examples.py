"""Every example config parses with the pipeline's own parsers and points at a data file in the repository."""
import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
import predictor_main

EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "configs", "examples", "*.yaml")))


def test_examples_found():
	assert len(EXAMPLES) >= 2


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_example_config_parses_and_data_exists(path, monkeypatch):
	monkeypatch.chdir(ROOT)                       # example paths are relative to the repository root
	parser = predictor_main.build_parser()
	config.apply_config(parser, ["--config", path])
	args = parser.parse_args(["--config", path])
	assert args.filename and os.path.isfile(args.filename), args.filename
	assert args.testname
