"""Pruning of Optuna trials that train several repeats (seeds).

Optuna ignores repeated reports for the same epoch, so repeats 2+ could not be
judged; the curves are kept here and only one is reported when the trial ends.
With repeat_method "worst" every repeat is judged and a finished trial stores its
worst repeat's curve: prune when a repeat's best value so far is worse than the
median of the finished trials at that epoch (a trial can't score better than any
of its repeats). Other repeat methods judge only the first repeat.
"""
import math

import numpy as np
import optuna
from optuna.trial import TrialState


class RepeatPruning:
	"""Per-repeat validation curves of one trial; warm-up and interval restart every repeat."""

	def __init__(self, trial, n_startup_trials=5, n_warmup_steps=0, interval_steps=1, prune_all_repeats=True):
		self.trial = trial
		self.n_startup_trials = n_startup_trials
		self.n_warmup_steps = n_warmup_steps
		self.interval_steps = max(1, int(interval_steps))
		self.prune_all_repeats = prune_all_repeats
		self.curves = []        # one {epoch: value} per repeat
		self.pruned = False
		self._reference = None  # finished trials' curves, read once per repeat

	def start_repeat(self):
		"""Call before each repeat's training."""
		self.curves.append({})
		self._reference = None

	def reset_repeat(self):
		"""Forget the current repeat's values (its training is retried)."""
		self.curves[-1] = {}
		self.pruned = False

	def _median_at(self, epoch):
		if self._reference is None:
			finished = self.trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
			self._reference = {"n": len(finished), "trials": [t.intermediate_values for t in finished]}
		if self._reference["n"] < self.n_startup_trials:
			return None
		values = [v[epoch] for v in self._reference["trials"] if epoch in v and v[epoch] is not None and math.isfinite(v[epoch])]
		return float(np.median(values)) if values else None

	def epoch_end(self, epoch, value):
		"""Record this repeat's value at `epoch` (0-based); return True if the trial should be pruned."""
		curve = self.curves[-1]
		curve[epoch] = float(value) if value is not None else float("nan")
		if self.pruned:
			return True
		if not self.prune_all_repeats and len(self.curves) > 1:
			return False
		if not math.isfinite(curve[epoch]):
			self.pruned = True  # NaN is pruned, as in MedianPruner
			return True
		if epoch < self.n_warmup_steps or (epoch - self.n_warmup_steps) % self.interval_steps != 0:
			return False
		median = self._median_at(epoch)
		if median is None:
			return False
		best_so_far = min(v for v in curve.values() if math.isfinite(v))
		self.pruned = best_so_far > median
		return self.pruned

	def report_curve(self, repeat_index):
		"""Report repeat `repeat_index`'s curve (0-based) as the trial's intermediate values."""
		for epoch, value in sorted(self.curves[repeat_index].items()):
			if math.isfinite(value):
				self.trial.report(value, epoch)

	def finish(self, scores, aggregated, repeat_method):
		"""Report the worst repeat's curve ("worst") or the one closest to `aggregated`; return its index."""
		s = np.asarray(scores, dtype=float)
		if repeat_method == "worst":
			idx = int(np.nanargmax(s))
		else:
			idx = int(np.nanargmin(np.abs(s - aggregated)))
		self.report_curve(idx)
		return idx

	def prune(self):
		"""Report the pruned repeat's curve (TPE uses it) and raise optuna.TrialPruned."""
		self.report_curve(len(self.curves) - 1)
		raise optuna.TrialPruned()
