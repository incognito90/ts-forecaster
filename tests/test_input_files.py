"""Input files: parquet / CSV / TSV give the same data; date column name; non-numeric columns."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_handling as dh


def frame(n=300):
	rng = np.random.default_rng(0)
	return pd.DataFrame({
		"Date": pd.date_range("2021-01-01", periods=n, freq="h"),
		"rain": rng.gamma(1.0, size=n),
		"x_Label": 10 + np.sin(np.arange(n) / 8) + rng.normal(0, 0.1, n),
	})


def read(path, **kw):
	return dh.readData("t", str(path), label_tag="_Label", input_seq_length=6, output_seq_length=2,
		batch_size=8, nan_imputation_method="min_std", transform_method="minmax", **kw)


def windows(w):
	x = np.concatenate([b[0].numpy() for b in w.test_ds])
	observed, _ = w.observed_test_windows()
	return x, observed


def test_parquet_csv_and_tsv_give_identical_windows(tmp_path):
	df = frame()
	df.to_parquet(tmp_path / "d.parquet", engine="fastparquet")
	df.to_csv(tmp_path / "d.csv", index=False)
	df.to_csv(tmp_path / "d.tab", index=False, sep="\t")
	ref_x, ref_obs = windows(read(tmp_path / "d.parquet"))
	for name in ("d.csv", "d.tab"):
		x, obs = windows(read(tmp_path / name))
		np.testing.assert_allclose(x, ref_x, rtol=1e-6)
		np.testing.assert_allclose(obs, ref_obs, rtol=1e-12)


def test_custom_date_column_and_text_columns(tmp_path, capsys):
	df = frame().rename(columns={"Date": "timestamp"})
	df["station"] = "Nongzhanguan"                    # text column: dropped with a warning
	df.to_csv(tmp_path / "d.csv", index=False)
	w = read(tmp_path / "d.csv", date_column="timestamp")
	assert "station" not in w.input_columns and "rain" in w.input_columns
	assert "dropping non-numeric column(s) ['station']" in capsys.readouterr().out


def test_missing_date_column_and_text_target_are_errors(tmp_path):
	df = frame()
	df.to_csv(tmp_path / "d.csv", index=False)
	with pytest.raises(SystemExit, match="no time column 'time'"):
		read(tmp_path / "d.csv", date_column="time")
	df["x_Label"] = "high"
	df.to_csv(tmp_path / "e.csv", index=False)
	with pytest.raises(SystemExit, match="not numeric"):
		read(tmp_path / "e.csv")


def test_unsupported_extension(tmp_path):
	(tmp_path / "d.xlsx").write_text("x")
	with pytest.raises(SystemExit, match="unsupported data file type"):
		read(tmp_path / "d.xlsx")
