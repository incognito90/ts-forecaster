"""Output folder of a run (--output-dir, default outputs/<testname>) and locked CSV appends.

results.csv holds one row per target and lead time.
Several processes (SLURM array tasks, parallel runs) may share one folder, so the
shared CSVs are only appended to with append_lines_locked, which holds an
exclusive file lock so rows from different processes never interleave.
"""
import csv
import io
import os

try:
	import fcntl  # POSIX (Linux, macOS, HPC clusters)
except ImportError:  # pragma: no cover - Windows
	fcntl = None
	import msvcrt


class OutputPaths:
	"""Paths of everything a run writes, all below one root folder."""

	# images: training curves, history: per-epoch CSVs, checkpoints: best weights and epoch logs,
	# predictions: per-sample forecasts, optuna: search CSVs, database and trial logs
	SUBDIRS = ("images", "models", "history", "checkpoints", "predictions", "optuna", "logs")

	def __init__(self, root):
		self.root = root
		self.results_csv = os.path.join(root, "results.csv")
		for sub in self.SUBDIRS:
			setattr(self, sub, os.path.join(root, sub))

	def create(self):
		for path in (self.root,) + tuple(getattr(self, sub) for sub in self.SUBDIRS):
			os.makedirs(path, exist_ok=True)
		return self


def append_lines_locked(path, lines, header=None):
	"""Append lines (each ending in a newline) to `path` under an exclusive lock.

	header is written to an empty file; an existing file must have the same header
	(checked under the lock), otherwise ValueError and nothing is appended.
	"""
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "a+", newline="") as f:
		_lock(f)
		try:
			f.seek(0, os.SEEK_END)
			if header is not None:
				if f.tell() == 0:
					f.write(header)
				else:
					f.seek(0)
					existing = next(csv.reader(f))
					expected = next(csv.reader(io.StringIO(header)))
					if existing != expected:
						raise ValueError(
							f"CSV header mismatch for {path}: columns or column order differ. "
							"Use a separate output directory for these results; the existing file was not changed.")
					f.seek(0, os.SEEK_END)
			for line in lines:
				f.write(line)
			f.flush()
			os.fsync(f.fileno())
		finally:
			_unlock(f)


def check_csv_header(path, header):
	"""Raise ValueError if `path` exists with another header (checked before a run starts)."""
	if not os.path.exists(path) or os.path.getsize(path) == 0:
		return
	with open(path, newline="") as f:
		existing = next(csv.reader(f), None)
	if existing != next(csv.reader(io.StringIO(header))):
		raise ValueError(
			f"{path} already holds results with other columns (e.g. made with another --metrics or "
			"--seasonal-period, or by another version). Use a separate output directory for this run.")


def _lock(f):
	if fcntl is not None:
		fcntl.flock(f, fcntl.LOCK_EX)
	else:  # pragma: no cover - windows: lock the first byte
		f.seek(0)
		msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)


def _unlock(f):
	if fcntl is not None:
		fcntl.flock(f, fcntl.LOCK_UN)
	else:  # pragma: no cover
		f.seek(0)
		msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
