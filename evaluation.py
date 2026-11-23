"""Scores written to results.csv, chosen by group with --metrics (evaluation.metrics).

Groups (comma-separated, or "all"): core (default), hydrology, peaks, relative;
see METRIC_GROUPS and the README. Every score is computed on the positions where
both the observation and the forecast exist, and also for the reference forecasts.
"""
import numpy as np

import HydroErr as he

from metrics import (
	fdc_metric,
	kge_metric_components,
	mae_metric,
	mape_metric,
	model_efficiency_metric,
	mse_loss,
	msle_loss,
	pbias_metric,
	peak_flow_error,
	peak_prediction_accuracy,
	rmse_metric,
	rve_metric,
	to_scalar,
)

PEAK_PERCENTILES = (75, 90, 95, 99)

METRIC_GROUPS = {
	"core": ["mae", "rmse", "mse", "r2", "smape", "mase", "bias", "Pearson Correlation"],
	"hydrology": ["NSE", "KGE", "alpha", "beta", "pbias", "rve", "fdc", "nmse", "model efficiency"],
	"peaks": [name for q in PEAK_PERCENTILES for name in (f"recall ({q}th)", f"peak false discovery rate ({q}th)")]
		+ [f"peak_flow_error_{q}" for q in PEAK_PERCENTILES],
	"relative": ["mape", "msle"],  # need strictly positive data
}


def resolve_metric_groups(spec):
	"""'all', 'core,hydrology' or a list of group names -> ordered list of metric names."""
	if isinstance(spec, str):
		groups = [g.strip() for g in spec.split(",") if g.strip()]
	else:
		groups = [str(g).strip() for g in spec]
	if not groups:
		raise ValueError("no metric group given")
	if "all" in groups:
		groups = list(METRIC_GROUPS)
	unknown = [g for g in groups if g not in METRIC_GROUPS]
	if unknown:
		raise ValueError(f"unknown metric group(s) {unknown}; choose from {', '.join(METRIC_GROUPS)} or all")
	names = []
	for g in groups:
		names.extend(n for n in METRIC_GROUPS[g] if n not in names)
	return names


def paired_values(observed, simulated):
	"""Keep the positions where both series are finite; return float64 arrays."""
	observed = np.asarray(observed, dtype=np.float64)
	simulated = np.asarray(simulated, dtype=np.float64)
	valid = np.isfinite(observed) & np.isfinite(simulated)
	return observed[valid], simulated[valid]


def compute_metrics(observed, simulated, names, naive_mae=np.nan):
	"""Scores `names` for one target and lead time; returns (scores, data_variance, obs_std).

	naive_mae is the MAE of the one-step naive forecast on the training period (for mase).
	"""
	o, s = paired_values(observed, simulated)
	if len(o) == 0:
		return {n: np.nan for n in names}, np.nan, np.nan
	data_variance = float(np.var(o))
	obs_std = float(np.std(o))

	def wanted(group):
		return any(n in names for n in METRIC_GROUPS[group])

	m = {}
	if wanted("core") or wanted("hydrology"):
		mse = to_scalar(mse_loss(o, s, cast=False))
		KGE, correlation, alpha, beta = [to_scalar(v) for v in kge_metric_components(o, s, cast=False)]
		if np.ptp(o) == 0 or np.ptp(s) == 0:
			# correlation with a constant series (e.g. climatology) is undefined, so is KGE
			KGE, correlation = np.nan, np.nan

	if wanted("core"):
		mae = to_scalar(mae_metric(o, s, cast=False))
		m["mae"] = mae
		m["rmse"] = to_scalar(rmse_metric(o, s, cast=False))
		m["mse"] = mse
		m["r2"] = 1.0 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2)
		m["smape"] = _smape(o, s)
		m["mase"] = mae / naive_mae if naive_mae and np.isfinite(naive_mae) else np.nan
		m["bias"] = float(np.mean(s - o))
		m["Pearson Correlation"] = correlation

	if wanted("hydrology"):
		m["NSE"] = he.nse(s, o)
		m["KGE"] = KGE
		m["alpha"] = alpha
		m["beta"] = beta
		m["pbias"] = to_scalar(pbias_metric(o, s, cast=False))
		m["rve"] = to_scalar(rve_metric(o, s, cast=False))
		m["fdc"] = to_scalar(fdc_metric(o, s, cast=False))
		m["nmse"] = mse / data_variance if data_variance > 0 else np.nan
		m["model efficiency"] = to_scalar(model_efficiency_metric(o, s, cast=False))

	if wanted("peaks"):
		for q in PEAK_PERCENTILES:
			recall, fdr = peak_prediction_accuracy(o, s, q, cast=False)
			m[f"recall ({q}th)"] = to_scalar(recall)
			m[f"peak false discovery rate ({q}th)"] = to_scalar(fdr)
			m[f"peak_flow_error_{q}"] = to_scalar(peak_flow_error(o, s, q, cast=False))

	if wanted("relative"):
		m["mape"] = to_scalar(mape_metric(o, s, cast=False))
		m["msle"] = to_scalar(msle_loss(o, s, cast=False))

	return {n: float(m[n]) for n in names}, data_variance, obs_std


def _smape(o, s):
	# 100 * mean 2|s - o| / (|o| + |s|), 0 where both are 0
	denom = np.abs(o) + np.abs(s)
	terms = np.where(denom == 0, 0.0, 2.0 * np.abs(s - o) / np.where(denom == 0, 1.0, denom))
	return 100.0 * float(np.mean(terms))
