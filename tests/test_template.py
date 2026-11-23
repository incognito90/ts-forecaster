"""configs/template.yaml lists every option and is up to date (regenerate with tools/make_config_template.py)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import config
import make_config_template


def test_template_is_up_to_date():
	with open(make_config_template.TEMPLATE) as f:
		assert f.read() == make_config_template.render(), "run: python tools/make_config_template.py"


def test_template_is_a_valid_config():
	values = config.read_config(make_config_template.TEMPLATE)
	expected = {d for dests in config.ALL_SECTIONS.values() for d in dests}
	assert set(values) == expected
