"""YAML config files: every command-line option grouped into sections (configs/template.yaml).

Precedence: built-in default < config file (--config) < command line. Keys are the
option names with '-' -> '_' in their section (SECTIONS), plus a few shorter ALIASES;
unknown sections or keys stop the run. The explain section holds the shap_explain.py
settings. Every run writes the settings it used to <output-dir>/configs/.
"""
import difflib
import os

import yaml

# argparse dest names by config section
SECTIONS = {
	"run": ["mode", "testname", "output_dir", "test_num", "save_model", "train", "trainedModelFile", "suppress_warnings"],
	"data": ["filename", "date_column", "label_tag", "input_length", "output_length", "output_offset",
		"train_ratio", "val_ratio", "test_ratio", "time_features", "trend_period_years", "change_lags", "change_columns", "predict"],
	"preprocessing": ["nan_fill", "transform_method", "oversample_high_thresholds", "oversample_low_thresholds"],
	"model": ["model_type", "decoder", "num_layers", "units_per_layer", "dropout_rate", "input_dropout", "noise", "weight_init",
		"batch_normalization", "l1_regularization", "l2_regularization", "l1_recurrent_regularization",
		"l2_recurrent_regularization", "l1_dense_regularization", "l2_dense_regularization"],
	"training": ["loss_function", "learning_rate", "batch_size", "max_epochs", "min_epochs", "patience",
		"patience_metric", "gradient_clip", "half_float", "deterministic", "mirrored", "num_gpu_splits", "cur_gpu_split"],
	"evaluation": ["metrics", "seasonal_period"],
	"tuning": ["hyperparameter_optimization", "search_space", "hyperparameter_optimization_repeat",
		"hyperparameter_optimization_repeat_method", "hyperparameter_optimization_n_jobs",
		"hyperparameter_optimization_db_dir", "hyperparameter_optimization_timeout", "final_repeats"],
}

# shap_explain.py settings, kept apart from SECTIONS (predictor_main.py options,
# which a training run writes back to <output-dir>/configs/)
EXPLAIN_SECTIONS = {
	"explain": ["run_shap", "shap_sample_mode", "shap_flow_quantile", "shap_lowflow_quantile", "shap_horizon",
		"shap_label_index", "shap_seed", "shap_background_batches", "shap_background_size",
		"shap_explain_batches", "shap_explain_size", "shap_focus_window_start", "shap_focus_window_end",
		"feature_groups", "member_group", "member_pattern", "type_pattern", "member_categories",
		"run_prediction_test", "run_original_test",
		"max_test_batches_for_cache", "checkpoint"],
}

ALL_SECTIONS = {**SECTIONS, **EXPLAIN_SECTIONS}

# shorter config names: (section, key) -> argparse dest, also used when writing configs
ALIASES = {
	("run", "name"): "testname",
	("run", "trained_model"): "trainedModelFile",
	("data", "file"): "filename",
	("tuning", "trials"): "hyperparameter_optimization",
	("tuning", "repeats"): "hyperparameter_optimization_repeat",
	("tuning", "repeat_method"): "hyperparameter_optimization_repeat_method",
	("tuning", "n_jobs"): "hyperparameter_optimization_n_jobs",
	("tuning", "db_dir"): "hyperparameter_optimization_db_dir",
	("tuning", "timeout"): "hyperparameter_optimization_timeout",
}
_DEST_TO_ALIAS = {dest: key for (section, key), dest in ALIASES.items()}

# values that may be lists/dicts, passed through unchanged
STRUCTURED = {"search_space", "feature_groups"}


def _section_keys(section):
	return {_DEST_TO_ALIAS.get(d, d) for d in ALL_SECTIONS[section]} | set(ALL_SECTIONS[section])


def _dest_section(dest):
	return next(sec for sec, dests in SECTIONS.items() if dest in dests)


def _as_cli_value(value):
	# as typed on the command line, so the parser converts and checks both the same way
	if value is None:
		return None
	if isinstance(value, bool):
		return "True" if value else "False"
	if isinstance(value, (list, tuple)):
		return ",".join(str(v) for v in value)
	return str(value)


def read_config(path):
	"""Config file -> {argparse dest: value}; ValueError (with a suggestion) for unknown keys."""
	with open(path) as f:
		raw = yaml.safe_load(f) or {}
	if not isinstance(raw, dict):
		raise ValueError(f"{path}: expected sections such as 'data:' and 'model:' at the top level")
	values = {}
	for section, entries in raw.items():
		if section not in ALL_SECTIONS:
			hint = difflib.get_close_matches(str(section), ALL_SECTIONS, n=1)
			raise ValueError(f"{path}: unknown section '{section}'" + (f" (did you mean '{hint[0]}'?)" if hint else "")
				+ f". Sections: {', '.join(ALL_SECTIONS)}")
		if entries is None:
			continue
		if not isinstance(entries, dict):
			raise ValueError(f"{path}: section '{section}' must contain 'key: value' lines")
		for key, value in entries.items():
			dest = ALIASES.get((section, key), key)
			if dest not in ALL_SECTIONS[section]:
				other = next((s for s in ALL_SECTIONS if dest in ALL_SECTIONS[s] or (s, key) in ALIASES), None)
				if other:
					raise ValueError(f"{path}: '{key}' belongs in section '{other}', not '{section}'")
				hint = difflib.get_close_matches(str(key), sorted(_section_keys(section)), n=1)
				raise ValueError(f"{path}: unknown key '{section}.{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
			values[dest] = value if dest in STRUCTURED else _as_cli_value(value)
	return values


def apply_config(parser, argv=None, translate=None):
	"""If argv has --config, use the file's values as parser defaults; return the path or None.

	Only options `parser` defines are set (predictor_main.py writes every argument to
	results.csv). translate maps read_config's dict to `parser`'s names (shap_explain.py).
	"""
	pre = parser.parse_known_args(argv)[0]
	path = getattr(pre, "config", None)
	if path:
		try:
			values = read_config(path)
		except (OSError, ValueError, yaml.YAMLError) as e:
			parser.error(f"--config: {e}")
		if translate is not None:
			values = translate(values)
		known = {action.dest for action in parser._actions}
		parser.set_defaults(**{dest: value for dest, value in values.items() if dest in known})
	return path


def _plain(value):
	# numpy values and tuples -> plain yaml values
	if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
		return value.item()
	if hasattr(value, "tolist"):
		return value.tolist()
	if isinstance(value, tuple):
		return list(value)
	if isinstance(value, str) and value.lower() in ("true", "false"):
		return value.lower() == "true"  # the parser keeps booleans as text
	return value


def config_from_args(args):
	"""Parsed arguments -> sectioned config dict."""
	out = {}
	for section, dests in SECTIONS.items():
		block = {}
		for dest in dests:
			if hasattr(args, dest):
				block[_DEST_TO_ALIAS.get(dest, dest)] = _plain(getattr(args, dest))
		if block:
			out[section] = block
	return out


def write_config(args, path, header=None, overrides=None):
	"""Write the run's settings as a config that can be run again.

	overrides: {argparse dest: value} applied on top of args (e.g. the best trial's settings).
	"""
	cfg = config_from_args(args)
	for dest, value in (overrides or {}).items():
		section = _dest_section(dest)
		cfg.setdefault(section, {})[_DEST_TO_ALIAS.get(dest, dest)] = _plain(value)
	# copy the explain section from the source config so the file still works with --mode explain
	source = getattr(args, "config", None)
	if source and os.path.exists(source):
		with open(source) as f:
			original = yaml.safe_load(f) or {}
		for section in EXPLAIN_SECTIONS:
			if original.get(section):
				cfg[section] = original[section]
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w") as f:
		if header:
			f.write("".join(f"# {line}\n" for line in header.splitlines()))
		yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
	return path
