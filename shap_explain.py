"""SHAP explanations of a trained ts-forecaster model.

Rebuilds the data window of the training run (same split, gap filling and
scaling), loads the trained model (--saved-model, or --checkpoint weights put
into a rebuilt architecture), optionally re-runs the test-set evaluation and,
with --run-shap, computes SHAP values (shap.GradientExplainer) for one target
column at one forecast horizon. Background windows come from the training
split, explained windows from the test split. Every combination of sample
mode, quantile and horizon gets its own run.

Run from the repo root with the config of the training run. The SHAP settings
go in its `explain` section; options on the command line override the config
(run.py --config ... --mode explain does the same):

    python shap_explain.py --config my_experiment.yaml --run-shap True

Without --config every data option must be given and match the training run.

Sample modes (--shap-sample-mode):
  random     random test windows
  high_flow  test windows with target >= the --shap-flow-quantile quantile
  low_flow   test windows with target <= the --shap-lowflow-quantile quantile
The names come from river flow, but any target works.

Each input column gets a group (--feature-groups) for the group summary. With
--member-group the columns of one group (e.g. dams) are also broken down per
member, sub-type (--type-pattern), lag bin and category (--member-categories).

Outputs go to <run.output_dir>/shap, or outputs/<run.name>/shap (without
--config: outputs/shap/<test-name>):
  feature_columns.json, label_columns.json
  checkpoint_test_metrics.csv and prediction .npy files (--run-prediction-test)
  shap/*.npy   background/explained windows and SHAP values, shaped
               (samples, input steps, features), scaled space
  shap/*.csv   mean |SHAP| per feature and group (and member, type, lag bin)
  *_summary_plot_*.png, *_member_lag.png, *_member_categories.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

import re

import copy

import config as run_config
import data_handling as dh

try:
	import shap
except ImportError as exc:
	raise ImportError(
		"SHAP is not installed in this environment. Install it before running --run-shap True."
	) from exc

import datetime

from model import (
	MODEL_BUILDERS,
	get_lr_metric,
	ConditionalNoise,
)
from metrics import (
	rmse,
	nse,
	nse_loss,
	kge_loss,
	mse_loss,
	rmse_metric,
	kge_metric_components,
	mae_metric,
	pbias_metric,
	peak_prediction_accuracy,
)


SHAP_SAMPLE_MODES = ["random", "high_flow", "low_flow"]


def str_to_bool(value):
	"""argparse type for True/False options."""
	if isinstance(value, bool):
		return value

	value = str(value).strip().lower()

	if value in ["true", "1", "yes", "y"]:
		return True

	if value in ["false", "0", "no", "n"]:
		return False

	raise argparse.ArgumentTypeError(f"Cannot convert {value} to bool.")


def none_or_float(value):
	"""argparse type: none/null/nan/'' -> None, otherwise float."""
	if value is None:
		return None

	value = str(value).strip()

	if value.lower() in ["none", "null", "nan", ""]:
		return None

	return float(value)


def none_or_float_list(value):
	"""argparse type: 'None' or comma-separated floats, as in predictor_main.py."""
	if value.lower() == "none":
		return None

	return list(map(float, value.split(",")))


def build_window(args):
	"""Rebuild the data window of the training run with dh.readData()."""
	half_float = str_to_bool(args.half_float)

	if half_float:
		policy = tf.keras.mixed_precision.Policy("mixed_float16")
		tf.keras.mixed_precision.set_global_policy(policy)
	# floatx stays float32, as in predictor_main.py
	tf.keras.backend.set_floatx("float32")

	window = dh.readData(
		args.test_name,
		args.filename,
		label_tag=args.label_tag,
		input_seq_length=args.input_length,
		output_seq_length=args.output_length,
		output_seq_offset=args.output_offset,
		batch_size=args.batch_size,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		oversample_high_thresholds=args.oversample_high_thresholds,
		oversample_low_thresholds=args.oversample_low_thresholds,
		nan_imputation_method=args.nan_fill,
		transform_method=args.transform_method,
		log_dir=str(args.output_dir),
		trend_period_years=args.trend_period_years,
		date_column=args.date_column,
		time_features=args.time_features,
		predict=args.predict,
		change_lags=args.change_lags,
		change_columns=args.change_columns,
	)

	return window


def build_model(args, window):
	"""Rebuild the untrained architecture so checkpoint weights can be loaded into it."""
	if args.model_type not in MODEL_BUILDERS:
		raise ValueError(f"Unknown model_type: {args.model_type} (available: {', '.join(MODEL_BUILDERS)})")
	return MODEL_BUILDERS[args.model_type](
		args.num_layers,
		args.units_per_layer,
		args.dropout_rate,
		window.num_features,
		window.input_steps,
		window.num_outputs,
		window.label_steps,
		weight_init=args.weight_init,
		batch_normalization=args.batch_normalization,
		l1_regularization=args.l1_regularization,
		l2_regularization=args.l2_regularization,
		recurrent_l1_regularization=args.l1_recurrent_regularization,
		recurrent_l2_regularization=args.l2_recurrent_regularization,
		dense_l1_regularization=args.l1_dense_regularization,
		dense_l2_regularization=args.l2_dense_regularization,
		noise_std=args.noise,
		decoder=args.decoder,
		input_dropout=args.input_dropout,
	)


def normalize_checkpoint_path(checkpoint_path):
	"""Strip the .index / .data-* suffix from a checkpoint path."""
	checkpoint_path = str(checkpoint_path)

	for suffix in [".index", ".data-00000-of-00001"]:
		if checkpoint_path.endswith(suffix):
			checkpoint_path = checkpoint_path[: -len(suffix)]

	return checkpoint_path


def revert_predictions(window, predictions):
	"""Model outputs for the test windows -> target levels in original units.

	Change outputs (--predict change) are turned into levels first, so the
	metrics are always computed on levels.
	"""
	return window.revert_label_predictions(window.outputs_to_level(predictions, "level"))


def finite_pair(simulated, observed):
	"""Keep only the positions where both are finite."""
	valid = np.isfinite(simulated) & np.isfinite(observed)

	return simulated[valid].astype(np.float64), observed[valid].astype(np.float64)


def summarize_metrics(y_test, pred, label_columns):
	"""Test metrics, one row per (label, horizon).

	y_test, pred: (samples, horizons, labels) in original units.
	"""
	rows = []

	for label_index, label_name in enumerate(label_columns):
		for horizon_index in range(y_test.shape[1]):
			observed = y_test[:, horizon_index, label_index]
			simulated = pred[:, horizon_index, label_index]
			simulated, observed = finite_pair(simulated, observed)

			if len(observed) == 0:
				rows.append({
					"label": label_name,
					"horizon": horizon_index + 1,
					"n_valid": 0,
					"mse": np.nan,
					"nmse": np.nan,
					"rmse": np.nan,
					"kge": np.nan,
					"correlation": np.nan,
					"alpha": np.nan,
					"beta": np.nan,
					"mae": np.nan,
					"pbias": np.nan,
					"recall_99": np.nan,
					"peak_fdr_99": np.nan,
				})
				continue

			data_variance = np.var(observed)
			mse = mse_loss(observed, simulated, cast=False)
			rmse = rmse_metric(observed, simulated, cast=False)
			kge, correlation, alpha, beta = kge_metric_components(observed, simulated, cast=False)
			mae = mae_metric(observed, simulated, cast=False)
			pbias = pbias_metric(observed, simulated, cast=False)
			recall_99, peak_fdr_99 = peak_prediction_accuracy(observed, simulated, 99, cast=False)

			rows.append({
				"label": label_name,
				"horizon": horizon_index + 1,
				"n_valid": len(observed),
				"mse": float(mse),
				"nmse": float(mse / data_variance) if data_variance != 0 else np.nan,
				"rmse": float(rmse),
				"kge": float(kge),
				"correlation": float(correlation),
				"alpha": float(alpha),
				"beta": float(beta),
				"mae": float(mae),
				"pbias": float(pbias),
				"recall_99": float(recall_99),
				"peak_fdr_99": float(peak_fdr_99),
			})

	return pd.DataFrame(rows)


def dataset_to_numpy(dataset, max_batches=None):
	"""Concatenate the (x, y) batches of a dataset into numpy arrays."""
	x_parts = []
	y_parts = []

	for batch_index, batch in enumerate(dataset):
		if max_batches is not None and batch_index >= max_batches:
			break

		x_batch, y_batch = batch
		x_parts.append(x_batch.numpy())
		y_parts.append(y_batch.numpy())

	return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def write_feature_metadata(window, output_dir):
	"""Write feature_columns.json and label_columns.json."""
	feature_columns = list(window.column_indices.keys()) if hasattr(window, "column_indices") else list(window.train_df.columns)
	label_columns = list(window.label_columns)

	with open(output_dir / "feature_columns.json", "w") as f:
		json.dump(feature_columns, f, indent=2)

	with open(output_dir / "label_columns.json", "w") as f:
		json.dump(label_columns, f, indent=2)

	return feature_columns, label_columns

def get_custom_objects():
	"""custom_objects for loading models saved by predictor_main.py."""
	opt = tf.keras.optimizers.Adam()
	lr_metric = get_lr_metric(opt)

	return {
		"lr": lr_metric,
		"ConditionalNoise": ConditionalNoise,
		"rmse": rmse,
		"nse": nse,
		"kge_loss": kge_loss,
		"nse_loss": nse_loss,
	}

def sample_dataset_windows(dataset, max_batches, sample_size, seed, valid_windows=None):
	"""Draw up to sample_size (x, y) windows from the first max_batches batches.

	valid_windows: optional boolean mask over the windows in dataset order,
	applied before sampling (max_batches still counts the original batches).
	"""
	x_all = []
	y_all = []
	window_start = 0
	if valid_windows is not None:
		valid_windows = np.asarray(valid_windows, dtype=bool)
		if valid_windows.ndim != 1:
			raise ValueError("valid_windows must be a one-dimensional window mask")

	for batch_index, batch in enumerate(dataset):
		print(f"sample batch{batch_index} of {max_batches}")
		if max_batches is not None and batch_index >= max_batches:
			break

		x_batch, y_batch = batch
		x_batch, y_batch = x_batch.numpy(), y_batch.numpy()
		if valid_windows is not None:
			window_end = window_start + len(x_batch)
			keep = valid_windows[window_start:window_end]
			if len(keep) != len(x_batch):
				raise ValueError("valid_windows does not cover the dataset windows")
			window_start = window_end
			x_batch, y_batch = x_batch[keep], y_batch[keep]
		if len(x_batch):
			x_all.append(x_batch)
			y_all.append(y_batch)

	if len(x_all) == 0:
		raise ValueError(
			"No eligible samples found in the requested dataset batches. "
			"Explanation samples require observed targets throughout the prediction window."
		)

	x_all = np.concatenate(x_all, axis=0)
	y_all = np.concatenate(y_all, axis=0)

	if sample_size is not None and x_all.shape[0] > sample_size:
		rng = np.random.default_rng(seed)
		indices = rng.choice(x_all.shape[0], size=sample_size, replace=False)
		x_all = x_all[indices]
		y_all = y_all[indices]

	return x_all, y_all


def make_scalar_output_model(model, horizon_index, label_index):
	"""Wrap the model so it outputs one (horizon, label) only, shape (batch, 1)."""
	output = tf.keras.layers.Lambda(
		lambda y: y[:, horizon_index:horizon_index + 1, label_index],
		name=f"shap_output_h{horizon_index + 1}_label{label_index}"
	)(model.output)

	return tf.keras.Model(inputs=model.input, outputs=output)


def normalize_shap_values(shap_values):
	"""Coerce GradientExplainer output to (samples, timesteps, features)."""
	if isinstance(shap_values, list):
		shap_values = shap_values[0]

	shap_values = np.asarray(shap_values)

	if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
		shap_values = np.squeeze(shap_values, axis=-1)

	if shap_values.ndim != 3:
		raise ValueError(f"Expected SHAP values shaped (samples, timesteps, features), got {shap_values.shape}")

	return shap_values


# member name = text before the first '-' of the column name
DEFAULT_MEMBER_PATTERN = r"^([^-]+)"


def pattern_matches(pattern, feature_name):
	"""Substring match, or re.search if the pattern starts with 're:'."""
	if pattern.startswith("re:"):
		return re.search(pattern[3:], feature_name) is not None

	return pattern in feature_name


def parse_feature_groups(value):
	"""Parse --feature-groups (mapping, YAML/JSON text or file) into {group: [patterns]}."""
	if value is None or (isinstance(value, str) and value.strip().lower() in ["", "none", "null"]):
		return None

	if isinstance(value, str):
		text = value
		if os.path.isfile(value):
			with open(value) as f:
				text = f.read()
		try:
			value = yaml.safe_load(text)
		except yaml.YAMLError as exc:
			raise ValueError(f"feature_groups is not valid YAML/JSON: {exc}") from exc

	if not isinstance(value, dict) or len(value) == 0:
		raise ValueError("feature_groups must be a mapping {group: [patterns]}, e.g. {dam: [_dam], river: [_river]}")

	groups = {}

	for group, patterns in value.items():
		if isinstance(patterns, str):
			patterns = [patterns]

		if not isinstance(patterns, (list, tuple)) or len(patterns) == 0 or not all(isinstance(p, str) and p != "" for p in patterns):
			raise ValueError(f"feature_groups: group '{group}' needs a list of non-empty patterns (text, or 're:<regex>')")

		for pattern in patterns:
			if pattern.startswith("re:"):
				try:
					re.compile(pattern[3:])
				except re.error as exc:
					raise ValueError(f"feature_groups: invalid regular expression '{pattern}' in group '{group}': {exc}") from exc

		groups[str(group)] = list(patterns)

	return groups


def check_capture_pattern(pattern, name):
	"""Raise ValueError unless `pattern` is a valid regex with a group 1."""
	try:
		compiled = re.compile(pattern)
	except re.error as exc:
		raise ValueError(f"{name}: invalid regular expression '{pattern}': {exc}") from exc

	if compiled.groups < 1:
		raise ValueError(f"{name}: the regular expression '{pattern}' needs a group in parentheses, e.g. '^([^-]+)'")


def load_member_categories(path):
	"""Read the --member-categories CSV into a list of (pattern, category).

	Header pattern,category, e.g. a row 14238042,Rule Curve (Current).
	Patterns match like the feature_groups patterns; the first match wins.
	"""
	table = pd.read_csv(path, dtype=str, keep_default_na=False)
	table.columns = [str(column).strip() for column in table.columns]

	missing = [column for column in ["pattern", "category"] if column not in table.columns]
	if missing:
		raise ValueError(f"{path}: missing column(s) {missing}; expected a header with pattern,category")

	categories = {}

	for row_number, (pattern, category) in enumerate(zip(table["pattern"], table["category"]), start=1):
		pattern = pattern.strip()
		category = category.strip()

		if pattern == "" or category == "":
			raise ValueError(f"{path}: empty pattern or category in data row {row_number}")

		if categories.get(pattern, category) != category:
			raise ValueError(f"{path}: pattern {pattern} has two categories ('{categories[pattern]}' and '{category}')")

		if pattern.startswith("re:"):
			try:
				re.compile(pattern[3:])
			except re.error as exc:
				raise ValueError(f"{path}: invalid regular expression '{pattern}' in data row {row_number}: {exc}") from exc

		categories[pattern] = category

	return list(categories.items())


class FeatureGrouping:
	"""Group, member, sub-type and category of each input column."""

	def __init__(self, feature_groups=None, label_columns=(), member_group=None, member_pattern=DEFAULT_MEMBER_PATTERN, type_pattern=None, member_categories=None):
		self.feature_groups = feature_groups or None
		self.label_columns = list(label_columns)
		self.member_group = member_group
		self.member_pattern = member_pattern or DEFAULT_MEMBER_PATTERN
		self.type_pattern = type_pattern
		self.member_categories = member_categories

	def group(self, feature_name):
		"""First matching configured group, else time/target, else the text
		after the last '_' (only without feature_groups), else 'other'."""
		if self.feature_groups:
			for group, patterns in self.feature_groups.items():
				if any(pattern_matches(pattern, feature_name) for pattern in patterns):
					return group

		if feature_name in dh.TIME_FEATURE_COLUMNS:
			return "time"

		if feature_name in self.label_columns:
			return "target"

		if self.feature_groups or "_" not in feature_name:
			return "other"

		return feature_name.rsplit("_", 1)[1] or "other"

	def is_member(self, feature_name):
		"""True if the column is in member_group."""
		return self.member_group is not None and self.group(feature_name) == self.member_group

	def member(self, feature_name):
		"""Group 1 of member_pattern ('_' -> ' '), the whole name if no match."""
		match = re.search(self.member_pattern, feature_name)
		name = feature_name if match is None else match.group(1)

		return name.replace("_", " ")

	def type(self, feature_name):
		"""Group 1 of type_pattern, 'unknown' if no match."""
		if self.type_pattern is None:
			return None

		match = re.search(self.type_pattern, feature_name)

		return "unknown" if match is None else match.group(1)

	def category(self, feature_name):
		"""member_categories category (None if no row matches), else the sub-type, else 'all'."""
		if self.member_categories is not None:
			return next((category for pattern, category in self.member_categories if pattern_matches(pattern, feature_name)), None)

		if self.type_pattern is not None:
			return self.type(feature_name)

		return "all"

	def members(self, feature_columns):
		"""Distinct member names, in column order."""
		members = []

		for feature_name in feature_columns:
			if self.is_member(feature_name):
				member = self.member(feature_name)
				if member not in members:
					members.append(member)

		return members


def build_feature_annotation_table(feature_columns, grouping):
	"""One row per feature column: index, name, group (and member / type if set)."""
	rows = []

	for feature_index, feature_name in enumerate(feature_columns):
		row = {
			"feature_index": feature_index,
			"feature": feature_name,
			"group": grouping.group(feature_name),
		}

		if grouping.member_group is not None:
			row["member"] = grouping.member(feature_name) if row["group"] == grouping.member_group else ""

		if grouping.type_pattern is not None:
			row["type"] = grouping.type(feature_name)

		rows.append(row)

	return pd.DataFrame(rows)


def clip_focus_window(focus_window_start, focus_window_end, n_steps):
	"""Clip the focus window (steps back from the newest, end exclusive) to n_steps."""
	if focus_window_start < 0 or focus_window_end <= focus_window_start:
		raise ValueError(
			f"Invalid focus window {focus_window_start}-{focus_window_end}: "
			f"need 0 <= --shap-focus-window-start < --shap-focus-window-end."
		)

	clipped_start = min(focus_window_start, n_steps)
	clipped_end = min(focus_window_end, n_steps)

	if clipped_end <= clipped_start:
		raise ValueError(
			f"Focus window {focus_window_start}-{focus_window_end} lies outside the "
			f"{n_steps} input steps; lower --shap-focus-window-start."
		)

	return clipped_start, clipped_end


def step_unit(timestep_seconds):
	"""(abbreviation, name) of the time step, for labels."""
	if timestep_seconds == 3600:
		return "h", "hours"

	if timestep_seconds == 86400:
		return "d", "days"

	return "steps", "time steps"

def sample_high_flow_test_windows(window, sample_size, label_index, horizon_index, quantile=0.9, seed=123):
	"""Draw test windows with target >= the quantile, and as many from the rest.

	Returns (x_high, y_high, x_rest, y_rest, threshold), threshold in scaled units.
	"""
	# only windows with every target observed at every forecast step (checked on
	# the raw data, before gap filling)
	x_all, y_all = sample_dataset_windows(
		window.test,
		max_batches=None,
		sample_size=None,
		seed=seed,
		valid_windows=window.complete_test_window_mask(),
	)

	target = y_all[:, horizon_index, label_index]
	threshold = np.nanquantile(target, quantile)

	high_indices = np.where(target >= threshold)[0]
	normal_indices = np.where(target < threshold)[0]

	rng = np.random.default_rng(seed)

	high_indices = rng.choice(
		high_indices,
		size=min(sample_size, len(high_indices)),
		replace=False,
	)

	normal_indices = rng.choice(
		normal_indices,
		size=min(sample_size, len(normal_indices)),
		replace=False,
	)

	return (
		x_all[high_indices],
		y_all[high_indices],
		x_all[normal_indices],
		y_all[normal_indices],
		threshold,
	)

def sample_low_flow_test_windows(window, sample_size, label_index, horizon_index, quantile=0.1, seed=123):
	"""Same as sample_high_flow_test_windows for target <= the quantile."""
	x_all, y_all = sample_dataset_windows(
		window.test,
		max_batches=None,
		sample_size=None,
		seed=seed,
		valid_windows=window.complete_test_window_mask(),
	)

	target = y_all[:, horizon_index, label_index]
	threshold = np.nanquantile(target, quantile)

	low_indices = np.where(target <= threshold)[0]
	normal_indices = np.where(target > threshold)[0]

	rng = np.random.default_rng(seed)

	low_indices = rng.choice(
		low_indices,
		size=min(sample_size, len(low_indices)),
		replace=False,
	)

	normal_indices = rng.choice(
		normal_indices,
		size=min(sample_size, len(normal_indices)),
		replace=False,
	)

	return (
		x_all[low_indices],
		y_all[low_indices],
		x_all[normal_indices],
		y_all[normal_indices],
		threshold,
	)

def summarize_shap_by_feature(shap_values, feature_columns, grouping):
	"""Mean |SHAP| per feature, with its group (and member / type)."""
	abs_shap = np.abs(shap_values)
	feature_importance = abs_shap.mean(axis=(0, 1))

	result = pd.DataFrame({
		"feature_index": np.arange(len(feature_columns)),
		"feature": feature_columns,
		"mean_abs_shap": feature_importance,
	})

	annotations = build_feature_annotation_table(feature_columns, grouping)

	return result.merge(annotations, on=["feature_index", "feature"], how="left")


def summarize_shap_by_group(shap_values, feature_summary, group_column):
	"""Mean |SHAP| per value of group_column (signed values summed per group first)."""
	unique_groups = feature_summary[group_column].unique()
	
	group_results = []
	for g in unique_groups:
		# Get the exact original feature indices for all features in this group
		# from the 'feature_index' column of the dataframe you passed
		indices = feature_summary[feature_summary[group_column] == g]["feature_index"].astype(int).tolist()
		
		if len(indices) > 0:
			# FIX: Sum the RAW SHAP values for these specific features before taking the absolute
			raw_group_shap = np.sum(shap_values[:, :, indices], axis=2)
			# Take the absolute mean of the aggregated group impact
			true_mean_abs = np.mean(np.abs(raw_group_shap))
			group_results.append({group_column: g, 'mean_abs_shap': true_mean_abs})
			
	return pd.DataFrame(group_results).sort_values("mean_abs_shap", ascending=False)

def summarize_shap_by_lag_bin(shap_values, feature_indices, horizon_index, timestep_seconds=3600):
	"""|SHAP| of the given features per lag bin (lag to the target, in hours)."""
	abs_shap = np.abs(shap_values)
	n_steps = shap_values.shape[1]

	feature_indices = np.asarray(feature_indices, dtype=int)

	if len(feature_indices) == 0:
		raise ValueError("No features given for the lag-bin summary.")

	# last bin ends at the largest lag so every input step is counted
	step_hours = timestep_seconds / 3600.0
	lag_hours = ((n_steps - 1) - np.arange(n_steps) + (horizon_index + 1)) * step_hours
	max_lag_hours = int(np.ceil(lag_hours.max()))
	# only call it "1-3 months" if it ends within ~92 days
	last_bin_label = "1-3 months" if max_lag_hours <= 92 * 24 else "> 4 weeks"

	lag_bins = [
		("1-12 h", 1, 12),
		("13-24 h", 13, 24),
		("25-36 h", 25, 36),
		("37-48 h", 37, 48),
		("2-4 d", 49, 96),
		("4-7 d", 97, 168),
		("1-2 weeks", 169, 336),
		("2-4 weeks", 337, 672),
		(last_bin_label, 673, max_lag_hours),
	]

	rows = []

	for label, low, high in lag_bins:
		time_indices = np.where((lag_hours >= low) & (lag_hours <= high))[0]

		if len(time_indices) == 0:
			continue

		bin_values = abs_shap[:, time_indices, :][:, :, feature_indices]

		rows.append({
			"lag_bin": label,
			"lag_low_hours": low,
			"lag_high_hours": high,
			"mean_abs_shap": float(bin_values.mean()),
			"sum_abs_shap": float(bin_values.sum()),
			"n_timesteps": len(time_indices),
			"n_features": len(feature_indices),
		})

	result = pd.DataFrame(rows, columns=["lag_bin", "lag_low_hours", "lag_high_hours", "mean_abs_shap", "sum_abs_shap", "n_timesteps", "n_features"])
	total = result["sum_abs_shap"].sum()

	if total > 0:
		result["share_abs_shap"] = result["sum_abs_shap"] / total
	else:
		result["share_abs_shap"] = np.nan

	return result

def parse_csv_int_list(value):
	"""Parse '1,6,12' (or an int) into a list of ints."""
	if isinstance(value, int):
		return [value]

	return [
		int(item.strip())
		for item in str(value).split(",")
		if item.strip() != ""
	]


def parse_csv_float_list(value):
	"""Parse '0.9,0.95' (or a float) into a list of floats."""
	if isinstance(value, float):
		return [value]

	return [
		float(item.strip())
		for item in str(value).split(",")
		if item.strip() != ""
	]


def parse_csv_str_list(value, allowed_values=None):
	"""Parse 'a,b' into a list of strings, optionally checked against allowed_values."""
	values = [
		item.strip()
		for item in str(value).split(",")
		if item.strip() != ""
	]

	if allowed_values is not None:
		for item in values:
			if item not in allowed_values:
				raise ValueError(f"Invalid value '{item}'. Allowed values: {allowed_values}")

	return values


import matplotlib.pyplot as plt

def create_recent_shap_summary_plot(shap_values, x_sample, feature_columns, output_dir, file_prefix,focus_window_start=0, focus_window_end=36, unit_abbr="h"):
	"""
	Generates a SHAP beeswarm summary plot focusing ONLY on the recent temporal 
	window where the features are active, avoiding noise from the full input window.
	"""
	# 1. Slice the arrays to only include the most recent 'focus_window' timesteps
	# Assuming shape is (samples, timesteps, features) and the latest time is at the end [-focus_window:]
	end_idx = -focus_window_start if focus_window_start > 0 else None
	
	recent_shap = shap_values[:, -focus_window_end:end_idx, :]
	recent_x = x_sample[:, -focus_window_end:end_idx, :]
	
	# 2. Collapse only the focused timesteps. 
	# Sum the SHAP values to get the total impact over the focused timesteps.
	shap_values_2d = np.sum(recent_shap, axis=1)
	
	# Mean the feature values over the focused timesteps to get the representative state for coloring
	x_sample_2d = np.mean(recent_x, axis=1)
	
	# 3. Create DataFrame for plotting
	df_features = pd.DataFrame(x_sample_2d, columns=feature_columns)
	
	# 4. Generate and save plot
	plt.figure(figsize=(10, 8))
	shap.summary_plot(shap_values_2d, df_features, show=False)
	
	plot_path = os.path.join(output_dir, f"{file_prefix}_summary_plot_{focus_window_start}_{focus_window_end}{unit_abbr}.png")
	plt.savefig(plot_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Saved focused SHAP summary plot to: {plot_path}")


def export_member_analysis(shap_values, feature_columns, grouping, horizon_index, output_dir, file_prefix, sample_mode, focus_window_start=0, focus_window_end=None, timestep_seconds=None):
	"""Write the per-member lag plot and the member category table (focus window only)."""
	target_horizon = horizon_index + 1
	n_steps = shap_values.shape[1]

	if focus_window_end is None:
		focus_window_end = n_steps

	focus_window_start, focus_window_end = clip_focus_window(focus_window_start, focus_window_end, n_steps)
	shap_values = shap_values[:, n_steps - focus_window_end:n_steps - focus_window_start, :]

	# Lag relative to target. Oldest kept step = target_horizon + focus_window_end - 1,
	# newest kept step = target_horizon + focus_window_start
	lags = target_horizon + np.arange(focus_window_end - 1, focus_window_start - 1, -1)
	unit_abbr, unit_name = step_unit(timestep_seconds)

	# ---- 1. PLOT MEMBER ATTRIBUTION VS TARGET LAG ----
	plt.figure(figsize=(12, 6))
	members = grouping.members(feature_columns)

	for member in members:
		member_indices = [j for j, f in enumerate(feature_columns) if grouping.is_member(f) and grouping.member(f) == member]
		if len(member_indices) > 0:
			# Sum the RAW shap values of all features of this member at each timestep
			# (so opposing contributions cancel instead of inflating the total)
			raw_member_shap = np.sum(shap_values[:, :, member_indices], axis=2)
			# Take absolute value, then mean across samples
			true_mean_abs_lag = np.mean(np.abs(raw_member_shap), axis=0)
			plt.plot(lags, true_mean_abs_lag, label=member, linewidth=2)

	plt.xlim(max(lags), min(lags)) # Older lags on left, newer on right
	plt.title(f"{sample_mode}: SHAP attribution per '{grouping.member_group}' member relative to the prediction target (+{target_horizon}{unit_abbr})")
	plt.xlabel(f'Lag in {unit_name.capitalize()} Relative to Prediction Target')
	plt.ylabel('Mean Absolute SHAP (member features summed)')
	plt.legend(loc='upper left')
	plt.grid(True, linestyle='--', alpha=0.7)

	plot_path = os.path.join(output_dir, f"{file_prefix}_member_lag.png")
	plt.savefig(plot_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Saved member lag plot to: {plot_path}")

	# ---- 2. EXPORT MEMBER CATEGORY TABLE ----
	categories = [grouping.category(f) if grouping.is_member(f) else None for f in feature_columns]

	n_uncategorised = sum(1 for f, c in zip(feature_columns, categories) if grouping.is_member(f) and c is None)
	if n_uncategorised > 0:
		print(f"NOTE: {n_uncategorised} '{grouping.member_group}' feature(s) match no row of member_categories; they are left out of the member category table.")

	unique_cats = sorted(set([c for c in categories if c is not None]))
	cat_results = []

	for cat in unique_cats:
		cat_indices = [j for j, c in enumerate(categories) if c == cat]
		if len(cat_indices) > 0:
			# Sum the RAW shap values of the category first across all timesteps and features
			raw_cat_shap = np.sum(shap_values[:, :, cat_indices], axis=2) # sum features
			raw_cat_shap_total = np.sum(raw_cat_shap, axis=1) # sum timesteps
			true_mean_abs = np.mean(np.abs(raw_cat_shap_total)) # absolute mean across samples
			cat_results.append({'category': cat, 'mean_abs_shap': true_mean_abs})

	df_cat = pd.DataFrame(cat_results, columns=['category', 'mean_abs_shap']).sort_values("mean_abs_shap", ascending=False)
	cat_csv_path = os.path.join(output_dir, f"{file_prefix}_member_categories.csv")
	df_cat.to_csv(cat_csv_path, index=False)
	print(f"Saved member categories to: {cat_csv_path}")


def run_shap_grid(model, window, args, output_dir, shuffle_seed=42, grouping=None):
	"""Run run_shap_analysis for every sample mode x quantile x horizon."""
	shap_sample_modes = parse_csv_str_list(
		args.shap_sample_mode,
		allowed_values=SHAP_SAMPLE_MODES,
	)
	shap_horizons = parse_csv_int_list(args.shap_horizon)

	for shap_sample_mode in shap_sample_modes:
		if shap_sample_mode in ["random", "peak", "baseflow", "rising_limb", "falling_limb"]:
			quantiles_for_mode = [None]
		elif shap_sample_mode == "high_flow":
			shap_flow_quantiles = parse_csv_float_list(args.shap_flow_quantile)
			quantiles_for_mode = shap_flow_quantiles
		elif shap_sample_mode == "low_flow":
			shap_lowflow_quantiles = parse_csv_float_list(args.shap_lowflow_quantile)
			quantiles_for_mode = shap_lowflow_quantiles

		for shap_flow_quantile in quantiles_for_mode:
			for shap_horizon in shap_horizons:
				print("\n" + "=" * 80)
				print("Starting SHAP run")
				print(f"\tmode: {shap_sample_mode}")
				print(f"\tquantile: {shap_flow_quantile}")
				print(f"\thorizon: {shap_horizon}")
				print("=" * 80 + "\n")

				run_args = copy.copy(args)
				run_args.shap_sample_mode = shap_sample_mode
				run_args.shap_horizon = shap_horizon

				if shap_flow_quantile is not None:
					if shap_sample_mode == "high_flow":
						run_args.shap_flow_quantile = shap_flow_quantile
					elif shap_sample_mode == "low_flow":
						run_args.shap_lowflow_quantile = shap_flow_quantile

				run_shap_analysis(
					model=model,
					window=window,
					args=run_args,
					output_dir=output_dir,
					shuffle_seed=shuffle_seed,
					grouping=grouping,
				)

def run_shap_analysis(model, window, args, output_dir, shuffle_seed=42, grouping=None):
	"""One SHAP run for a single sample mode, quantile, horizon and label."""
	output_dir = Path(output_dir)

	print(f"Running SHAP analysis with mode={args.shap_sample_mode}")
	print(f"output_dir: {output_dir}")

	shap_dir = output_dir / "shap"
	shap_dir.mkdir(parents=True, exist_ok=True)

	feature_columns = list(window.column_indices.keys()) if hasattr(window, "column_indices") else list(window.train_df.columns)

	if "Date" in feature_columns:
		feature_columns = [col for col in feature_columns if col != "Date"]

	if len(feature_columns) != window.num_features:
		raise ValueError(
			f"Feature-column count mismatch. feature_columns={len(feature_columns)}, window.num_features={window.num_features}"
		)

	if grouping is None:
		grouping = FeatureGrouping(label_columns=window.label_columns)

	member_names = grouping.members(feature_columns)

	if grouping.member_group is None:
		print("No member_group set: per-member SHAP outputs are not written.")
	elif len(member_names) == 0:
		print(f"NOTE: no feature column is in the member group '{grouping.member_group}'; the member SHAP outputs (top member features, member/type/lag summaries, member lag plot, member categories) are skipped.")
	else:
		print(f"Members of group '{grouping.member_group}' in the feature columns: {', '.join(member_names)}")

	timestep_seconds = window.timestep_seconds
	unit_abbr, unit_name = step_unit(timestep_seconds)

	horizon_index = args.shap_horizon - 1
	label_index = args.shap_label_index

	# quantile in the file name so runs with several quantiles don't overwrite each other
	mode_name = args.shap_sample_mode
	if mode_name == "high_flow":
		mode_name += f"_q{args.shap_flow_quantile}"
	elif mode_name == "low_flow":
		mode_name += f"_q{args.shap_lowflow_quantile}"
	file_prefix = f"{args.test_name}_shap_{mode_name}_h{horizon_index + 1}_label{label_index}"


	if horizon_index < 0 or horizon_index >= window.label_steps:
		raise ValueError(f"--shap-horizon must be between 1 and {window.label_steps}")

	if label_index < 0 or label_index >= window.num_labels:
		raise ValueError(f"--shap-label-index must be between 0 and {window.num_labels - 1}")

	# high_flow / low_flow scan the whole test split, only random uses --shap-explain-batches
	print("Sampling SHAP background windows from training set...")
	background_x, _ = sample_dataset_windows(
		window.train,
		max_batches=args.shap_background_batches,
		sample_size=args.shap_background_size,
		seed=args.shap_seed,
	)

	explain_x, explain_y = None, None

	if args.shap_sample_mode == "random":
		sample_mode="Random"
		print("Sampling random SHAP explanation windows from test set...")
		explain_x, explain_y = sample_dataset_windows(
			window.test,
			max_batches=args.shap_explain_batches,
			sample_size=args.shap_explain_size,
			seed=args.shap_seed + 1,
			valid_windows=window.complete_test_window_mask(),
		)
		sample_description = "random"

	elif args.shap_sample_mode == "high_flow":
		sample_mode="High Flow"
		print("Sampling high-flow SHAP explanation windows from test set...")
		explain_x, explain_y, normal_x, normal_y, high_flow_threshold = sample_high_flow_test_windows(
			window=window,
			sample_size=args.shap_explain_size,
			label_index=label_index,
			horizon_index=horizon_index,
			quantile=args.shap_flow_quantile,
			seed=args.shap_seed + 1,
		)

		print(f"High-flow threshold in scaled label space: {high_flow_threshold}")
		sample_description = f"high_flow_q{args.shap_flow_quantile}"

	elif args.shap_sample_mode == "low_flow":
		sample_mode="Low Flow"
		print("Sampling low-flow SHAP explanation windows from test set...")
		explain_x, explain_y, normal_x, normal_y, low_flow_threshold = sample_low_flow_test_windows(
			window=window,
			sample_size=args.shap_explain_size,
			label_index=label_index,
			horizon_index=horizon_index,
			quantile=args.shap_lowflow_quantile,
			seed=args.shap_seed + 1,
		)

		print(f"Low-flow threshold in scaled label space: {low_flow_threshold}")
		sample_description = f"low_flow_q{args.shap_lowflow_quantile}"

	if explain_x is None or explain_y is None:
		raise ValueError(f"Failed to sample explanation windows for mode {args.shap_sample_mode}")
	
	background_x = background_x.astype(np.float32)
	explain_x = explain_x.astype(np.float32)

	print("SHAP background shape:", background_x.shape)
	print("SHAP explain shape:", explain_x.shape)

	scalar_model = make_scalar_output_model(model, horizon_index, label_index)

	print("Creating SHAP GradientExplainer...")
	explainer = shap.GradientExplainer(
		scalar_model,
		background_x,
		batch_size=8,
	)

	print("Computing SHAP values...")
	shap_values = explainer.shap_values(explain_x)
	shap_values = normalize_shap_values(shap_values)

	print("SHAP values computed and normalized.")

	if len(member_names) > 0:
		export_member_analysis(
			shap_values=shap_values,
			feature_columns=feature_columns,
			grouping=grouping,
			horizon_index=horizon_index,
			output_dir=output_dir,
			file_prefix=file_prefix,
			sample_mode=args.shap_sample_mode,
			focus_window_start=args.shap_focus_window_start,
			focus_window_end=args.shap_focus_window_end,
			timestep_seconds=timestep_seconds,
		)

	print("SHAP values shape:", shap_values.shape)

	np.save(shap_dir / f"{file_prefix}_background_x.npy", background_x)
	np.save(shap_dir / f"{file_prefix}_explain_x.npy", explain_x)
	np.save(shap_dir / f"{file_prefix}_explain_y_scaled.npy", explain_y)
	np.save(shap_dir / f"{file_prefix}_shap_values.npy", shap_values)
	
	# beeswarm plots for fixed windows back from the newest input step (hourly:
	# 12 h, 1 d, 2 d, 1 week, 4 weeks, 8 weeks), clipped to the input length
	starts=[0, 12, 24, 48, 168, 28*24, 56*24]
	n_steps = shap_values.shape[1]

	for i in range(len(starts)-1):
		if starts[i] >= n_steps:
			break
		window_end = min(starts[i+1], n_steps)
		print(f"Creating focused SHAP summary plot for window {starts[i]} to {window_end} {unit_name}...")
		create_recent_shap_summary_plot(
			shap_values=shap_values,
			x_sample=explain_x,
			feature_columns=feature_columns,
			output_dir=output_dir,
			file_prefix=file_prefix,
			focus_window_start=starts[i],
			focus_window_end=window_end,
			unit_abbr=unit_abbr,
		)


	feature_summary = summarize_shap_by_feature(shap_values, feature_columns, grouping)
	feature_summary.to_csv(shap_dir / f"shap_feature_summary_{sample_description}_h{args.shap_horizon}_label{label_index}.csv", index=False)

	group_summary = summarize_shap_by_group(shap_values, feature_summary, "group")
	group_summary.to_csv(shap_dir / f"shap_group_summary_{sample_description}_h{args.shap_horizon}_label{label_index}.csv", index=False)

	print("\nSHAP group summary:")
	print(group_summary)

	if len(member_names) == 0:
		print(f"\nSaved SHAP outputs to: {shap_dir}")
		return

	member_features = feature_summary[feature_summary["group"] == grouping.member_group]
	file_suffix = f"{sample_description}_h{args.shap_horizon}_label{label_index}.csv"

	top_member_features = (
		member_features
		.sort_values("mean_abs_shap", ascending=False)
		.head(20)
	)

	top_member_features.to_csv(shap_dir / f"shap_top_member_features_{file_suffix}", index=False)

	print(f"\nTop '{grouping.member_group}' features:")
	print(top_member_features)

	member_summary = summarize_shap_by_group(shap_values, member_features, "member")
	member_summary.to_csv(shap_dir / f"shap_member_summary_{file_suffix}", index=False)

	print(f"\nSHAP summary per '{grouping.member_group}' member:")
	print(member_summary)

	if grouping.type_pattern is not None:
		type_summary = summarize_shap_by_group(shap_values, member_features, "type")
		type_summary.to_csv(shap_dir / f"shap_member_type_summary_{file_suffix}", index=False)

		print(f"\nSHAP summary per type of the '{grouping.member_group}' features:")
		print(type_summary)

	member_indices = [j for j, f in enumerate(feature_columns) if grouping.is_member(f)]
	lag_summary = summarize_shap_by_lag_bin(shap_values, member_indices, horizon_index, timestep_seconds)
	lag_summary.to_csv(shap_dir / f"shap_member_lag_summary_{file_suffix}", index=False)

	print(f"\nSHAP lag summary of the '{grouping.member_group}' features:")
	print(lag_summary)

	print(f"\nSaved SHAP outputs to: {shap_dir}")

def main(args):
	"""Rebuild the window, load the model, then run the tests and/or SHAP."""
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if args.saved_model is None and args.checkpoint is None:
		raise ValueError("Provide either --saved-model or --checkpoint.")

	member_categories = None
	if args.member_categories is not None:
		member_categories = load_member_categories(args.member_categories)
		print(f"Loaded {len(member_categories)} member category patterns from {args.member_categories}")
		if args.member_group is None:
			print("NOTE: --member-categories is ignored without --member-group.")

	if args.saved_model is not None and args.checkpoint is not None:
		print("Both --saved-model and --checkpoint were provided. Using --saved-model and ignoring --checkpoint.")

	print("TensorFlow:", tf.__version__)
	print("Building window/data pipeline...")
	window = build_window(args)

	print("Window summary:")
	print(f"\tinput_steps: {window.input_steps}")
	print(f"\tlabel_steps: {window.label_steps}")
	print(f"\tnum_features: {window.num_features}")
	print(f"\tnum_labels: {window.num_labels}")
	print(f"\tlabel_columns: {window.label_columns}")

	feature_columns, label_columns = write_feature_metadata(window, output_dir)

	grouping = FeatureGrouping(
		feature_groups=args.feature_groups,
		label_columns=label_columns,
		member_group=args.member_group,
		member_pattern=args.member_pattern,
		type_pattern=args.type_pattern,
		member_categories=member_categories,
	)

	if args.saved_model is not None:
		print(f"Loading full saved model from: {args.saved_model}")
		model = tf.keras.models.load_model(
			args.saved_model,
			compile=False,
			custom_objects=get_custom_objects(),
		)
		model.summary(expand_nested=True)
	else:
		print("Building model...")
		model = build_model(args, window)
		model.summary(expand_nested=True)

		checkpoint_path = normalize_checkpoint_path(args.checkpoint)
		print(f"Loading checkpoint weights from: {checkpoint_path}")

		# expect_partial: the optimizer state is not needed. But every model variable
		# must be restored, otherwise a model rebuilt with the wrong settings silently
		# keeps some layers at their initial weights.
		status = model.load_weights(checkpoint_path).expect_partial()
		try:
			status.assert_existing_objects_matched()
		except AssertionError as e:
			sys.exit(f"Error: the checkpoint {checkpoint_path} does not match the model rebuilt from these "
				f"settings (some layers would keep their initial weights). Use the settings of the training "
				f"run, e.g. its <output-dir>/configs/<run>.yaml or best_config.yaml.\n{e}")

	expected_input_shape = (None, window.input_steps, window.num_features)
	expected_output_shape = (None, window.label_steps, window.num_outputs)

	if tuple(model.input_shape) != expected_input_shape:
		raise ValueError(
			f"Model input shape does not match recreated window. "
			f"Model: {model.input_shape}, expected: {expected_input_shape}"
		)

	if tuple(model.output_shape) != expected_output_shape:
		raise ValueError(
			f"Model output shape does not match recreated window. "
			f"Model: {model.output_shape}, expected: {expected_output_shape}"
		)

	if args.run_prediction_test:
		print("Running raw model prediction on window.test...")
		raw_predictions = model.predict(window.test)

		print("Reverting predictions and labels to original scale...")
		pred = revert_predictions(window, raw_predictions)
		y_test, _ = window.observed_test_windows()

		print("Shapes:")
		print(f"\traw_predictions: {raw_predictions.shape}")
		print(f"\tpred: {pred.shape}")
		print(f"\ty_test: {y_test.shape}")


		if args.run_original_test:
			print("Running original predictor_main.test() for exact metric reproduction...")
			# imported here, predictor_main pulls in optuna, seaborn etc.
			from predictor_main import test as original_test
			from output_paths import OutputPaths

			original_test(
				window,
				y_test,
				pred,
				OutputPaths(str(output_dir)).create(),
				"original_test",
				args.nan_fill,
				args.transform_method,
				trainTime=0,
				peak_vram=0,
				patience=0,
				epochs_used=0,
				best_epoch=0,
				args=args,
			)
			print(f"Original test output written to {output_dir}/results.csv")

		if np.isnan(pred).all():
			raise RuntimeError("All reverted predictions are NaN. Check model/load/preprocessing settings.")

		metrics_df = summarize_metrics(y_test, pred, label_columns)
		metrics_path = output_dir / "checkpoint_test_metrics.csv"
		metrics_df.to_csv(metrics_path, index=False)
		print(f"Saved metrics: {metrics_path}")

		

		np.save(output_dir / "raw_predictions_scaled.npy", raw_predictions)
		np.save(output_dir / "predictions_reverted.npy", pred)
		np.save(output_dir / "y_test_reverted.npy", y_test)

		print("Collecting a small scaled X_test sample for SHAP/IG later...")
		x_test_scaled, y_test_scaled = dataset_to_numpy(window.test, max_batches=args.max_test_batches_for_cache)
		np.save(output_dir / "x_test_scaled_sample.npy", x_test_scaled)
		np.save(output_dir / "y_test_scaled_sample.npy", y_test_scaled)

		print("Saved:")
		print(f"\t{output_dir / 'feature_columns.json'}")
		print(f"\t{output_dir / 'label_columns.json'}")
		print(f"\t{output_dir / 'x_test_scaled_sample.npy'}")
		print(f"\t{output_dir / 'y_test_scaled_sample.npy'}")

		print("\nTop metric rows:")
		print(metrics_df.head(min(12, len(metrics_df))))

	
	if args.run_shap:
		run_shap_grid(
			model=model,
			window=window,
			args=args,
			output_dir=output_dir,
			shuffle_seed=args.shap_seed,
			grouping=grouping,
		)

# predictor_main.py option names that are called differently here
CONFIG_DESTS = {
	"testname": "test_name",
	"trainedModelFile": "saved_model",
}

# predictor_main.py defaults, used with --config for options the config leaves
# out (so the window matches the training run). Checked against
# predictor_main.build_parser() in tests/test_shap_explain.py.
TRAINING_DEFAULTS = {
	"label_tag": "_Label",
	"date_column": "Date",
	"input_length": 24,
	"output_length": 2,
	"output_offset": 0,
	"train_ratio": 0.7,
	"val_ratio": 0.15,
	"time_features": "auto",
	"trend_period_years": 0,
	"change_lags": "1,3,6",
	"change_columns": None,
	"predict": "change",
	"nan_fill": "min_std",
	"transform_method": "minmax",
	"oversample_high_thresholds": None,
	"oversample_low_thresholds": None,
	"batch_size": 32,
	"half_float": "auto",
	"model_type": "lstm",
	"decoder": "False",
	"num_layers": 2,
	"units_per_layer": 512,
	"dropout_rate": 0.4,
	"weight_init": "glorot_uniform",
	"batch_normalization": "False",
	"l1_regularization": None,
	"l2_regularization": None,
	"l1_recurrent_regularization": None,
	"l2_recurrent_regularization": None,
	"l1_dense_regularization": None,
	"l2_dense_regularization": None,
	"noise": 0.0,
	"input_dropout": 0.0,
	"metrics": "core",
	"seasonal_period": 0,
}

# required on the command line or in the config
REQUIRED_OPTIONS = ["filename", "input_length", "output_length", "nan_fill", "transform_method"]


def translate_config(values):
	"""Map config values to this script's options, on top of TRAINING_DEFAULTS."""
	translated = dict(TRAINING_DEFAULTS)

	for dest, value in values.items():
		dest = CONFIG_DESTS.get(dest, dest)

		if dest == "output_dir" and value is not None:
			value = os.path.join(value, "shap")

		translated[dest] = value

	return translated


def build_parser():
	parser = argparse.ArgumentParser(description="SHAP explanations of a trained ts-forecaster model.")

	parser.add_argument("--config", default=None, help="YAML config of the training run; SHAP settings go in its explain section. Command-line options override it.")
	parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)  # passed on by run.py; ignored

	parser.add_argument("--filename", default=None, help="Dataset file the model was trained on. Required.")
	parser.add_argument("--saved-model", default=None, help="Path to a fully saved Keras model (SavedModel folder). If provided, --checkpoint is ignored.")
	parser.add_argument("--output-dir", default=None, help="Output folder. Default: outputs/shap/<test-name> (with --config: <run.output_dir>/shap or outputs/<run.name>/shap).")
	parser.add_argument("--test-name", default="checkpoint_explainability_test", help="Run name, used in the output file names. Default: checkpoint_explainability_test.")

	parser.add_argument("--label-tag", default="_Label", help="Text that marks the target columns. Must match the training run. Default: _Label.")
	parser.add_argument("--date-column", default="Date", help="Name of the time column. Must match the training run. Default: Date.")
	parser.add_argument("--input-length", type=int, default=None, help="Number of input time steps. Must match the training run. Required.")
	parser.add_argument("--output-length", type=int, default=None, help="Number of forecast time steps. Must match the training run. Required.")
	parser.add_argument("--output-offset", type=int, default=0)
	parser.add_argument("--batch-size", type=int, default=64, help="Batch size of the rebuilt datasets. Default: 64 (with --config: training.batch_size).")
	parser.add_argument("--train-ratio", type=float, default=0.7)
	parser.add_argument("--val-ratio", type=float, default=0.15)

	parser.add_argument("--nan-fill", default=None, help="Gap filling method. Must match the training run. Required.")
	parser.add_argument("--transform-method", default=None, help="Scaling method. Must match the training run. Required.")
	parser.add_argument("--trend-period-years", type=int, default=0, help="Must match the training run (see predictor_main.py --trend-period-years). Default: 0.")
	parser.add_argument("--predict", default="change", choices=["level", "change", "both"], help="level, change or both. Must match the training run; SHAP explains the output as trained (e.g. the change). Default: change.")
	parser.add_argument("--time-features", default="auto", help="Must match the training run (see predictor_main.py --time-features). Default: auto.")
	parser.add_argument("--change-lags", default="1,3,6", help="Must match the training run (see predictor_main.py --change-lags). Default: 1,3,6.")
	parser.add_argument("--change-columns", default=None, help="Must match the training run (see predictor_main.py --change-columns). Default: None.")
	parser.add_argument("--half-float", default="False", help="Mixed precision: True, False or auto (on if a GPU is found). Default: False (with --config: training.half_float).")

	parser.add_argument("--run-prediction-test", type=str_to_bool, default=True, help="Evaluate the model on the test period (checkpoint_test_metrics.csv). Default: True.")

	parser.add_argument("--run-shap", type=str_to_bool, default=False, help="Compute SHAP values. Default: False.")
	parser.add_argument("--shap-sample-mode", default="random", help="Comma-separated sample modes: random, high_flow, low_flow. Default: random.")
	parser.add_argument("--shap-flow-quantile", default="0.9", help="Comma-separated quantiles of the target, used only for high_flow mode, e.g. 0.9,0.95. Default: 0.9.")
	parser.add_argument("--shap-lowflow-quantile", default="0.1", help="Comma-separated quantiles of the target, used only for low_flow mode, e.g. 0.1,0.05. Default: 0.1.")
	parser.add_argument("--shap-horizon", default="1", help="Comma-separated 1-indexed output horizons, e.g. 1,6,12. Default: 1.")
	parser.add_argument("--shap-focus-window-start", type=int, default=0, help="Start of the focus window of the member lag plot and category table, in input steps back from the newest (0). Default: 0.")
	parser.add_argument("--shap-focus-window-end", type=int, default=48, help="End (exclusive) of that focus window, clipped to --input-length. Default: 48.")
	parser.add_argument("--shap-background-batches", type=int, default=4, help="Training batches scanned for the SHAP background windows. Default: 4.")
	parser.add_argument("--shap-background-size", type=int, default=16, help="Background windows drawn from those batches. Default: 16.")
	parser.add_argument("--shap-explain-batches", type=int, default=4, help="Test batches scanned for the explained windows (random mode only). Default: 4.")
	parser.add_argument("--shap-explain-size", type=int, default=32, help="Test windows explained per run. Default: 32.")
	parser.add_argument("--shap-label-index", type=int, default=0, help="Index of the target column to explain (see label_columns.json). Default: 0.")
	parser.add_argument("--shap-seed", type=int, default=123, help="Seed for drawing the background and explained windows. Default: 123.")

	parser.add_argument("--feature-groups", default=None, help="Ordered mapping {group: [patterns]} as YAML/JSON text or file; a pattern is a substring of the column name or 're:<regex>', first match wins. Default: calendar columns -> time, targets -> target, else the text after the last '_'.")
	parser.add_argument("--member-group", default=None, help="Group whose columns are also analysed per member, e.g. dam. Default: none.")
	parser.add_argument("--member-pattern", default=DEFAULT_MEMBER_PATTERN, help="Regex whose group 1 is the member name of a column. Default: ^([^-]+), the text before the first '-'.")
	parser.add_argument("--type-pattern", default=None, help="Regex whose group 1 is the sub-type of a column, e.g. '-([A-Z]{2})-\\d+' for SHEF codes. Default: none.")
	parser.add_argument("--member-categories", default=None, help="CSV (columns pattern,category) assigning member columns to categories; first match wins, unmatched columns are left out. Default: the sub-type, or 'all'.")


	parser.add_argument("--oversample-high-thresholds", type=none_or_float_list, default=None, help="Must match the training run. None or comma-separated floats, e.g. 0.9,0.95. Default: None.")
	parser.add_argument("--oversample-low-thresholds", type=none_or_float_list, default=None, help="Must match the training run. None or comma-separated floats, e.g. 0.1,0.05. Default: None.")


	parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint weights. Can include or omit suffixes like .index or .data-00000-of-00001.")

	parser.add_argument("--model-type", default="lstm", help="Model architecture, a name in model.MODEL_BUILDERS (only with --checkpoint). Default: lstm.")
	parser.add_argument("--decoder", type=str_to_bool, default=False, help="Whether the model was trained with --decoder True (only with --checkpoint). Default: False.")
	parser.add_argument("--num-layers", type=int, default=-1)
	parser.add_argument("--units-per-layer", type=int, default=-1)
	parser.add_argument("--dropout-rate", type=float, default=-1)
	parser.add_argument("--weight-init", default="glorot_uniform")
	parser.add_argument("--batch-normalization", type=str_to_bool, default=False)

	parser.add_argument("--l1-regularization", type=none_or_float, default=None)
	parser.add_argument("--l2-regularization", type=none_or_float, default=None)
	parser.add_argument("--l1-recurrent-regularization", type=none_or_float, default=None)
	parser.add_argument("--l2-recurrent-regularization", type=none_or_float, default=None)
	parser.add_argument("--l1-dense-regularization", type=none_or_float, default=None)
	parser.add_argument("--l2-dense-regularization", type=none_or_float, default=None)
	parser.add_argument("--noise", type=float, default=0.001)
	parser.add_argument("--input-dropout", type=float, default=0.0, help="Must match the training run (only with --checkpoint). Default: 0.0.")

	parser.add_argument("--max-test-batches-for-cache", type=int, default=4, help="Test batches saved as x/y_test_scaled_sample.npy. Default: 4.")

	parser.add_argument("--run-original-test", type=str_to_bool, default=True, help="Also run predictor_main.test(), which appends to results.csv (only with --run-prediction-test). Default: True.")
	parser.add_argument("--metrics", default="core", help="Metric groups for --run-original-test (see predictor_main.py --metrics). Default: core.")
	parser.add_argument("--seasonal-period", type=int, default=0, help="Seasonal-naive reference period for --run-original-test, 0 = off. Default: 0.")

	return parser


def parse_args(argv=None):
	"""Parse the config and command line and check the options."""
	parser = build_parser()
	config_path = run_config.apply_config(parser, argv, translate=translate_config)

	args = parser.parse_args(argv)

	missing = ["--" + dest.replace("_", "-") for dest in REQUIRED_OPTIONS if getattr(args, dest) is None]
	if missing:
		parser.error(f"the following arguments are required: {', '.join(missing)} (on the command line or in --config)")

	if str(args.half_float).strip().lower() == "auto":
		args.half_float = "True" if tf.config.list_physical_devices("GPU") else "False"
		print(f"--half-float auto -> {args.half_float}")
	try:
		str_to_bool(args.half_float)
	except argparse.ArgumentTypeError:
		parser.error(f"--half-float must be True, False or auto, not {args.half_float}")

	try:
		sample_modes = parse_csv_str_list(args.shap_sample_mode, allowed_values=SHAP_SAMPLE_MODES)
	except ValueError as exc:
		parser.error(f"--shap-sample-mode: {exc}")

	try:
		focus_window = clip_focus_window(args.shap_focus_window_start, args.shap_focus_window_end, args.input_length)
	except ValueError as exc:
		parser.error(str(exc))
	if focus_window != (args.shap_focus_window_start, args.shap_focus_window_end):
		print(f"NOTE: focus window {args.shap_focus_window_start}-{args.shap_focus_window_end} clipped to {focus_window[0]}-{focus_window[1]} (--input-length {args.input_length}).")

	try:
		args.feature_groups = parse_feature_groups(args.feature_groups)
		check_capture_pattern(args.member_pattern, "--member-pattern")
		if args.type_pattern is not None:
			check_capture_pattern(args.type_pattern, "--type-pattern")
	except ValueError as exc:
		parser.error(str(exc))

	if args.output_dir is None:
		if config_path:
			args.output_dir = os.path.join("outputs", args.test_name, "shap")
		else:
			args.output_dir = os.path.join("outputs", "shap", args.test_name)

	return args


if __name__ == "__main__":
	now = datetime.datetime.now()
	print(f"Starting shap_explain.py at: {str(now)}")
	main(parse_args())
