"""Tests for output_paths: folder layout and concurrent-safe CSV appends."""
import multiprocessing as mp
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output_paths import OutputPaths, append_lines_locked

HEADER = "writer,row,payload\n"
N_WRITERS = 8
ROWS_PER_WRITER = 200


def _writer(path, writer_id):
	for row in range(ROWS_PER_WRITER):
		# a long payload makes interleaved (corrupted) writes easy to detect
		append_lines_locked(path, [f"{writer_id},{row},{'x' * 500}\n"], header=HEADER)


def test_output_paths_layout(tmp_path):
	paths = OutputPaths(str(tmp_path / "run")).create()
	assert paths.results_csv == str(tmp_path / "run" / "results.csv")
	for sub in OutputPaths.SUBDIRS:
		assert os.path.isdir(getattr(paths, sub))


def test_parallel_appends_do_not_interleave(tmp_path):
	path = str(tmp_path / "results.csv")
	procs = [mp.Process(target=_writer, args=(path, i)) for i in range(N_WRITERS)]
	for p in procs:
		p.start()
	for p in procs:
		p.join()
		assert p.exitcode == 0

	with open(path) as f:
		lines = f.read().splitlines()
	assert lines[0] == HEADER.strip()
	assert lines.count(HEADER.strip()) == 1
	rows = lines[1:]
	assert len(rows) == N_WRITERS * ROWS_PER_WRITER
	assert all(len(r.split(",")) == 3 and r.endswith("x" * 500) for r in rows)
	assert len(set(rows)) == len(rows)


@pytest.mark.parametrize("header", ["b,a\n", "a,b,c\n"])
def test_incompatible_header_does_not_modify_existing_file(tmp_path, header):
	path = tmp_path / "results.csv"
	append_lines_locked(path, ["1,2\n"], header="a,b\n")
	before = path.read_bytes()
	with pytest.raises(ValueError, match="CSV header mismatch"):
		append_lines_locked(path, ["3,4\n"], header=header)
	assert path.read_bytes() == before
	# A rejected append releases the lock; compatible writers can continue.
	append_lines_locked(path, ["5,6\n"], header="a,b\n")
	assert path.read_text() == "a,b\n1,2\n5,6\n"


def test_equivalent_csv_header_quoting_and_newlines_are_accepted(tmp_path):
	path = tmp_path / "results.csv"
	path.write_bytes(b'"a",b\r\n1,2\r\n')
	append_lines_locked(path, ["3,4\n"], header="a,b\n")
	assert path.read_text() == "\"a\",b\n1,2\n3,4\n"
