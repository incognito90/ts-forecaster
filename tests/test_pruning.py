"""pruning.RepeatPruning: every repeat is judged, against finished trials' worst-repeat curves."""
import optuna
import pytest

from pruning import RepeatPruning

optuna.logging.set_verbosity(optuna.logging.WARNING)


def finished_trial(study, repeat_curves, method="worst"):
	"""Run one trial whose repeats have the given {epoch: value} curves; score = last value."""
	def objective(trial):
		p = RepeatPruning(trial, n_startup_trials=1, n_warmup_steps=2, interval_steps=1)
		scores = []
		for curve in repeat_curves:
			p.start_repeat()
			for e, v in curve.items():
				p.epoch_end(e, v)
			scores.append(min(curve.values()))
		value = max(scores)
		p.finish(scores, value, method)
		return value
	study.optimize(objective, n_trials=1)
	return study.trials[-1]


def test_worst_repeat_curve_is_stored():
	study = optuna.create_study()
	good = {e: 1.0 - 0.1 * e for e in range(5)}
	bad = {e: 2.0 - 0.1 * e for e in range(5)}
	t = finished_trial(study, [good, bad, good])
	assert t.intermediate_values == bad  # repeat 2 had the worst score, its curve is stored


def test_any_repeat_can_prune():
	study = optuna.create_study()
	finished_trial(study, [{e: 1.0 for e in range(5)}] * 2)  # reference: worst curve = 1.0 everywhere

	def objective(trial):
		p = RepeatPruning(trial, n_startup_trials=1, n_warmup_steps=2, interval_steps=1)
		p.start_repeat()
		for e in range(5):
			assert not p.epoch_end(e, 0.5)  # repeat 1 is better than the median: kept
		p.start_repeat()
		assert not p.epoch_end(0, 3.0)  # warm-up restarts in repeat 2
		assert not p.epoch_end(1, 3.0)
		assert p.epoch_end(2, 3.0)  # repeat 2 is worse than the median at epoch 2
		p.prune()

	study.optimize(objective, n_trials=1)
	t = study.trials[-1]
	assert t.state == optuna.trial.TrialState.PRUNED
	assert t.intermediate_values == {0: 3.0, 1: 3.0, 2: 3.0}  # the pruned repeat's curve


def test_only_first_repeat_judged_for_other_methods():
	study = optuna.create_study()
	finished_trial(study, [{e: 1.0 for e in range(5)}])

	def objective(trial):
		p = RepeatPruning(trial, n_startup_trials=1, n_warmup_steps=0, interval_steps=1, prune_all_repeats=False)
		p.start_repeat()
		p.epoch_end(0, 0.5)
		p.start_repeat()
		assert not p.epoch_end(0, 3.0)
		return 0.0

	study.optimize(objective, n_trials=1)


def test_no_pruning_before_startup_trials():
	study = optuna.create_study()

	def objective(trial):
		p = RepeatPruning(trial, n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
		p.start_repeat()
		assert not p.epoch_end(0, 100.0)
		return 0.0

	study.optimize(objective, n_trials=1)
