"""Tests for the dataset-independent parts of shap_explain.py: feature groups
(default rules and explain.feature_groups), members, sub-types and categories
from column names, the member_categories CSV, focus-window clipping, the
command-line checks and the config mapping (--config, explain section)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as run_config
import shap_explain as se

TARGET = "Brantford-QR-8659042_river_Flow"
DAM_COLUMNS = [
	TARGET,
	"min_temp-TA-0_climate",
	"Shand_Dam-HK-12169042_dam",
	"Shand_Dam-LS-14238042_dam",
	"Guelph_Lake_Dam-LS-14246042_dam",
	"Shand_Dam-LS-11189042_dam",
	"Laurel_Creek_Dam-LS-14208042_dam",
	"Day sin",
]
NO_DAM_COLUMNS = [TARGET, "min_temp-TA-0_climate", "Day sin", "Day cos"]
GRWS_GROUPS = {"dam": ["_dam"], "river": ["_river"], "climate": ["_climate"]}
SHEF = r"-([A-Z]{2})-\d+"
BASE_ARGS = ["--filename", "x.parquet", "--input-length", "24", "--output-length", "6",
	"--nan-fill", "min_std", "--transform-method", "minmax", "--test-name", "unit"]


def grws(**kwargs):
	"""The grouping that reproduces the former GRWS-specific dam analysis."""
	settings = dict(feature_groups=GRWS_GROUPS, label_columns=[TARGET], member_group="dam", type_pattern=SHEF)
	settings.update(kwargs)
	return se.FeatureGrouping(**settings)


# --- feature groups ---------------------------------------------------------

def test_default_groups():
	grouping = se.FeatureGrouping(label_columns=[TARGET])
	columns = DAM_COLUMNS + ["Week cos", "Discharge", "Station-QR-1_flow", "trailing_"]
	assert [grouping.group(c) for c in columns] == [
		"target", "climate", "dam", "dam", "dam", "dam", "dam", "time", "time", "other", "flow", "other"]


def test_configured_groups_first_match_and_fallback():
	grouping = se.FeatureGrouping(feature_groups={"upstream": ["re:^Shand_"], **GRWS_GROUPS}, label_columns=[TARGET])
	assert grouping.group("Shand_Dam-LS-14238042_dam") == "upstream"      # first matching group wins
	assert grouping.group("Guelph_Lake_Dam-LS-14246042_dam") == "dam"
	assert grouping.group(TARGET) == "river"                              # as in the former code
	assert grouping.group("Day sin") == "time"                            # calendar fallback
	assert grouping.group("x_Label") == "other"                           # unmatched, not a target here
	assert se.FeatureGrouping(feature_groups=GRWS_GROUPS, label_columns=["x_Label"]).group("x_Label") == "target"


def test_parse_feature_groups(tmp_path):
	assert se.parse_feature_groups(None) is None
	assert se.parse_feature_groups("none") is None
	parsed = se.parse_feature_groups("{dam: [_dam], river: _river, climate: ['re:_clim']}")
	assert list(parsed.items()) == [("dam", ["_dam"]), ("river", ["_river"]), ("climate", ["re:_clim"])]
	path = tmp_path / "groups.yaml"
	path.write_text("b: [x]\na: [y]\n")
	assert list(se.parse_feature_groups(str(path))) == ["b", "a"]               # order kept
	assert se.parse_feature_groups(GRWS_GROUPS) == GRWS_GROUPS
	for bad in ["[a, b]", "{a: []}", "{a: ['re:(']}", "{a: [1]}"]:
		with pytest.raises(ValueError):
			se.parse_feature_groups(bad)


def test_capture_patterns_are_checked():
	se.check_capture_pattern(SHEF, "--type-pattern")
	with pytest.raises(ValueError, match="group in parentheses"):
		se.check_capture_pattern("-[A-Z]{2}-", "--type-pattern")
	with pytest.raises(ValueError, match="invalid"):
		se.check_capture_pattern("(", "--member-pattern")


# --- members, types, categories ---------------------------------------------

def test_member_names_derived_from_columns():
	assert grws().members(DAM_COLUMNS) == ["Shand Dam", "Guelph Lake Dam", "Laurel Creek Dam"]
	assert grws().members(NO_DAM_COLUMNS) == []
	assert se.FeatureGrouping().members(DAM_COLUMNS) == []                    # no member_group
	custom = grws(member_pattern=r"-(\d+)_dam$")
	assert custom.members(DAM_COLUMNS)[:2] == ["12169042", "14238042"]


def test_types_and_category_fallbacks():
	assert [grws().type(c) for c in DAM_COLUMNS] == ["QR", "TA", "HK", "LS", "LS", "LS", "LS", "unknown"]
	categories = [grws().category(c) if grws().is_member(c) else None for c in DAM_COLUMNS]
	assert categories == [None, None, "HK", "LS", "LS", "LS", "LS", None]   # type as category
	assert grws(type_pattern=None).category("Shand_Dam-LS-14238042_dam") == "all"
	assert grws(type_pattern=None).type("Shand_Dam-LS-14238042_dam") is None


def test_categories_from_csv(tmp_path):
	path = tmp_path / "cats.csv"
	path.write_text("pattern,category\n14238042,Rule Curve\n11189042,Storage\nre:-12169\\d+_dam,Elevation\n")
	grouping = grws(member_categories=se.load_member_categories(path))
	categories = [grouping.category(c) if grouping.is_member(c) else None for c in DAM_COLUMNS]
	assert categories == [None, None, "Elevation", "Rule Curve", None, "Storage", None, None]


def test_load_member_categories(tmp_path):
	path = tmp_path / "cats.csv"
	path.write_text("pattern, category ,note\n14238042,Rule Curve (Current),x\n 0123 , Storage ,\n14238042,Rule Curve (Current),dup\n")
	# whitespace is stripped; identical duplicates are allowed; file order is kept
	assert se.load_member_categories(path) == [("14238042", "Rule Curve (Current)"), ("0123", "Storage")]


@pytest.mark.parametrize("content, message", [
	("series_id,category\n1,a\n", "missing column"),
	("pattern,category\n1,a\n1,b\n", "two categories"),
	("pattern,category\n1,\n", "empty"),
])
def test_load_member_categories_errors(tmp_path, content, message):
	path = tmp_path / "cats.csv"
	path.write_text(content)
	with pytest.raises(ValueError, match=message):
		se.load_member_categories(path)


def test_annotation_table_columns():
	table = se.build_feature_annotation_table(DAM_COLUMNS, grws())
	assert list(table.columns) == ["feature_index", "feature", "group", "member", "type"]
	assert list(table["member"])[:3] == ["", "", "Shand Dam"]
	plain = se.build_feature_annotation_table(DAM_COLUMNS, se.FeatureGrouping(label_columns=[TARGET]))
	assert list(plain.columns) == ["feature_index", "feature", "group"]


# --- SHAP aggregation ----------------------------------------------------------

def test_clip_focus_window():
	assert se.clip_focus_window(0, 48, 72) == (0, 48)
	assert se.clip_focus_window(0, 48, 24) == (0, 24)
	assert se.clip_focus_window(12, 48, 24) == (12, 24)
	with pytest.raises(ValueError):
		se.clip_focus_window(30, 48, 24)
	with pytest.raises(ValueError):
		se.clip_focus_window(10, 10, 24)
	with pytest.raises(ValueError):
		se.clip_focus_window(-1, 10, 24)


def test_export_member_analysis_short_input(tmp_path):
	# 24 input steps with the default 48-step focus window used to crash
	rng = np.random.default_rng(0)
	shap_values = rng.normal(size=(5, 24, len(DAM_COLUMNS)))
	se.export_member_analysis(shap_values, DAM_COLUMNS, grws(), horizon_index=0, output_dir=str(tmp_path),
		file_prefix="t", sample_mode="random", focus_window_start=0, focus_window_end=48, timestep_seconds=3600)
	assert (tmp_path / "t_member_lag.png").exists()
	table = pd.read_csv(tmp_path / "t_member_categories.csv")
	assert sorted(table["category"]) == ["HK", "LS"]
	ls_indices = [3, 4, 5, 6]
	expected = np.mean(np.abs(shap_values[:, :, ls_indices].sum(axis=(1, 2))))
	assert table.set_index("category").loc["LS", "mean_abs_shap"] == pytest.approx(expected)


def test_member_summaries_by_group():
	rng = np.random.default_rng(1)
	shap_values = rng.normal(size=(3, 4, len(DAM_COLUMNS)))
	summary = se.summarize_shap_by_feature(shap_values, DAM_COLUMNS, grws())
	members = se.summarize_shap_by_group(shap_values, summary[summary["group"] == "dam"], "member")
	shand = [2, 3, 5]
	expected = np.mean(np.abs(shap_values[:, :, shand].sum(axis=2)))
	assert members.set_index("member").loc["Shand Dam", "mean_abs_shap"] == pytest.approx(expected)


def test_lag_bins_daily_data():
	shap_values = np.ones((2, 10, len(DAM_COLUMNS)))
	members = [2, 3, 4, 5, 6]
	hourly = se.summarize_shap_by_lag_bin(shap_values, members, 0, timestep_seconds=3600)
	assert list(hourly["lag_bin"]) == ["1-12 h"]
	assert list(hourly.columns) == ["lag_bin", "lag_low_hours", "lag_high_hours", "mean_abs_shap", "sum_abs_shap",
		"n_timesteps", "n_features", "share_abs_shap"]
	daily = se.summarize_shap_by_lag_bin(shap_values, members, 0, timestep_seconds=86400)
	# lags of 1..10 days = 24..240 hours
	assert list(daily["lag_bin"]) == ["13-24 h", "37-48 h", "2-4 d", "4-7 d", "1-2 weeks"]
	assert daily["n_timesteps"].sum() == 10
	with pytest.raises(ValueError):
		se.summarize_shap_by_lag_bin(shap_values, [], 0)


# --- command line and config ----------------------------------------------------

def test_parse_args_defaults_and_thresholds():
	args = se.parse_args(BASE_ARGS + ["--oversample-high-thresholds", "0.9,0.95", "--oversample-low-thresholds", "None"])
	assert args.oversample_high_thresholds == [0.9, 0.95]
	assert args.oversample_low_thresholds is None
	assert args.output_dir == os.path.join("outputs", "shap", "unit")
	assert args.feature_groups is None and args.member_group is None
	assert args.batch_size == 64 and args.half_float == "False"            # shap_explain's own defaults


def test_parse_args_required_options(capsys):
	with pytest.raises(SystemExit) as exc:
		se.parse_args(["--filename", "x.parquet"])
	assert exc.value.code == 2
	assert "--input-length, --output-length, --nan-fill, --transform-method" in capsys.readouterr().err


def write(tmp_path, cfg, name="cfg.yaml"):
	path = tmp_path / name
	path.write_text(yaml.safe_dump(cfg, sort_keys=False))
	return str(path)


EXPERIMENT = {
	"run": {"name": "exp", "output_dir": "outputs/exp_dir", "trained_model": "outputs/exp_dir/models/m", "test_num": 1},
	"data": {"file": "data.parquet", "label_tag": "_Flow", "input_length": 72, "output_length": 6},
	"preprocessing": {"nan_fill": "min_std", "oversample_high_thresholds": [0.9]},
	"model": {"units_per_layer": 64},
	"training": {"batch_size": 512, "max_epochs": 3, "half_float": False},
	"explain": {"run_shap": True, "run_prediction_test": False, "shap_horizon": [1, 6],
		"shap_sample_mode": ["random", "high_flow"], "feature_groups": {"dam": ["_dam"], "river": ["re:_river"]},
		"member_group": "dam", "type_pattern": SHEF},
}


def test_config_mapping(tmp_path):
	args = se.parse_args(["--config", write(tmp_path, EXPERIMENT), "--mode", "explain"])
	assert (args.filename, args.test_name, args.saved_model) == ("data.parquet", "exp", "outputs/exp_dir/models/m")
	assert args.output_dir == os.path.join("outputs/exp_dir", "shap")
	assert (args.label_tag, args.input_length, args.output_length, args.batch_size) == ("_Flow", 72, 6, 512)
	assert args.oversample_high_thresholds == [0.9] and args.units_per_layer == 64 and args.half_float == "False"
	assert args.run_shap is True and args.run_prediction_test is False
	assert args.shap_horizon == "1,6" and args.shap_sample_mode == "random,high_flow"
	assert list(args.feature_groups.items()) == [("dam", ["_dam"]), ("river", ["re:_river"])]   # passed through
	assert (args.member_group, args.type_pattern) == ("dam", SHEF)
	# options the config leaves out take predictor_main's defaults, as in the training run
	assert (args.transform_method, args.num_layers, args.dropout_rate, args.noise) == ("minmax", 2, 0.4, 0.0)
	assert not hasattr(args, "test_num") and not hasattr(args, "max_epochs")


def test_config_command_line_precedence(tmp_path):
	path = write(tmp_path, EXPERIMENT)
	args = se.parse_args(["--config", path, "--input-length", "48", "--output-dir", "elsewhere", "--test-name", "t2",
		"--member-group", "none_such", "--shap-horizon", "2", "--feature-groups", "{x: [_x]}"])
	assert (args.input_length, args.output_dir, args.test_name) == (48, "elsewhere", "t2")
	assert (args.member_group, args.shap_horizon, args.feature_groups) == ("none_such", "2", {"x": ["_x"]})
	assert args.batch_size == 512                                           # still from the config


def test_config_default_output_dir_and_half_float(tmp_path):
	cfg = {"run": {"name": "exp"}, "data": {"file": "d.parquet"}}
	args = se.parse_args(["--config", write(tmp_path, cfg)])
	assert args.output_dir == os.path.join("outputs", "exp", "shap")        # next to the training outputs
	assert (args.input_length, args.output_length, args.nan_fill) == (24, 2, "min_std")
	assert args.half_float in ("True", "False")                             # auto resolved


def test_training_defaults_match_predictor_main():
	from predictor_main import build_parser
	parser = build_parser()
	for dest, value in se.TRAINING_DEFAULTS.items():
		assert str(parser.get_default(dest)) == str(value), dest


def test_predictor_main_ignores_explain_section(tmp_path):
	from predictor_main import build_parser
	parser = build_parser()
	argv = ["--config", write(tmp_path, EXPERIMENT)]
	run_config.apply_config(parser, argv)
	args = parser.parse_args(argv)
	assert args.input_length == 72 and args.testname == "exp"
	present = [key for key in run_config.EXPLAIN_SECTIONS["explain"] if hasattr(args, key)]
	assert present == []


@pytest.mark.parametrize("cfg, message", [
	({"explain": {"shap_horizn": 1}}, "did you mean 'shap_horizon'"),
	({"data": {"run_shap": True}}, "belongs in section 'explain'"),
	({"explian": {"run_shap": True}}, "did you mean 'explain'"),
])
def test_explain_section_validation(tmp_path, cfg, message):
	with pytest.raises(ValueError, match=message):
		run_config.read_config(write(tmp_path, cfg))


def test_read_config_keeps_feature_groups_structured(tmp_path):
	values = run_config.read_config(write(tmp_path, {"explain": {"feature_groups": GRWS_GROUPS, "shap_horizon": [1, 6]}}))
	assert values["feature_groups"] == GRWS_GROUPS and values["shap_horizon"] == "1,6"
