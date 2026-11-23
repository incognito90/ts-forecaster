"""Run an experiment from one config file.

    python run.py --config configs/my_experiment.yaml [--mode MODE] [overrides ...]

  train     train one model and evaluate it on the test period (default)
  tune      Optuna search, then write best_config.yaml and train the final
            model(s) from it into <output-dir>/final/
  evaluate  evaluate a saved model (run.trained_model) on the test period
  explain   SHAP explanations of a saved model (explain section, shap_explain.py)

Options after the config override it for this run (e.g. --max-epochs 5).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "predictor_main.py")
EXPLAIN = os.path.join(HERE, "shap_explain.py")


def resolve_settings(argv):
	"""Settings of the run (config + overrides), parsed as predictor_main.py does."""
	sys.path.insert(0, HERE)
	import config as run_config
	from predictor_main import build_parser
	parser = build_parser()
	run_config.apply_config(parser, argv)
	return parser.parse_args(argv)


def config_mode(path):
	# None if unset or unreadable (resolve_settings then reports the problem)
	sys.path.insert(0, HERE)
	import config as run_config
	try:
		return run_config.read_config(path).get("mode")
	except Exception:
		return None


def run(cmd):
	print("\n$ " + " ".join(cmd), flush=True)
	result = subprocess.run(cmd)
	if result.returncode != 0:
		sys.exit(f"run.py: step failed (exit code {result.returncode}): {' '.join(cmd)}")


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", required=True, help="YAML config of the experiment (see configs/template.yaml)")
	parser.add_argument("--mode", choices=["train", "tune", "evaluate", "explain"], help="overrides run.mode of the config")
	args, overrides = parser.parse_known_args()

	explain = (args.mode or config_mode(args.config)) == "explain"
	argv = ["--config", args.config] + overrides + (["--mode", args.mode] if args.mode else [])
	if explain:
		# shap_explain.py reads the config itself; overrides are its own options
		print("run.py: mode=explain")
		run([sys.executable, EXPLAIN] + argv)
		return
	settings = resolve_settings(argv)
	mode = settings.mode
	output_dir = settings.output_dir or os.path.join("outputs", settings.testname or "")
	python = sys.executable
	print(f"run.py: mode={mode}, output folder={output_dir}")

	if mode == "train":
		run([python, MAIN] + argv)

	elif mode == "tune":
		if not settings.hyperparameter_optimization:
			sys.exit("run.py: --mode tune needs the number of trials: set tuning.trials in the config "
				"(or pass --hyperparameter-optimization N)")
		run([python, MAIN] + argv)  # search, writes best_config.yaml
		best = os.path.join(output_dir, "best_config.yaml")
		if not os.path.exists(best):
			sys.exit(f"run.py: the search did not write {best}")
		for repeat in range(1, settings.final_repeats + 1):
			run([python, MAIN, "--config", best, "--test-num", str(repeat), "--mode", "train"])
		print(f"\nrun.py: done. Best settings: {best}\n        Final model(s) and results: {os.path.join(output_dir, 'final')}")

	elif mode == "evaluate":
		if not settings.trainedModelFile:
			sys.exit("run.py: --mode evaluate needs the saved model: set run.trained_model in the config "
				"(or pass --trainedModelFile PATH)")
		run([python, MAIN] + argv + ["--train", "False", "--hyperparameter-optimization", "None"])


if __name__ == "__main__":
	main()
