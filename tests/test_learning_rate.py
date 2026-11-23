"""--learning-rate semantics: omitted -> Adam default, -1 -> cyclical schedule, x -> constant x."""
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import OneCycleScheduler, create_callbacks

WINDOW = SimpleNamespace(num_train_samples=1000, batch_size=100)


def build(learning_rate, grad_clip_norm=None):
	callbacks, _, _, _, opt = create_callbacks(
		log_dir=None, min_epochs=1, max_epochs=5, learning_rate=learning_rate, patience=-1,
		patience_metric="val_loss", grad_clip_norm=grad_clip_norm, window=WINDOW,
		logger=logging.getLogger("test"))
	has_schedule = any(isinstance(c, OneCycleScheduler) for c in callbacks)
	return float(np.asarray(opt.learning_rate)), has_schedule, opt


@pytest.mark.parametrize("learning_rate, expected_rate, expected_schedule", [
	(None, 0.001, False),      # omitted -> Adam default, constant
	(-1, 0.001, True),         # schedule (the scheduler overrides the rate every batch)
	(0.0005, 0.0005, False),   # given -> used as a constant rate
])
def test_learning_rate_modes(learning_rate, expected_rate, expected_schedule):
	rate, has_schedule, _ = build(learning_rate)
	assert rate == pytest.approx(expected_rate)
	assert has_schedule is expected_schedule


def test_gradient_clipping_is_kept_with_every_mode():
	for learning_rate in (None, -1, 0.0005):
		_, _, opt = build(learning_rate, grad_clip_norm=1.0)
		assert opt.clipnorm == 1.0
