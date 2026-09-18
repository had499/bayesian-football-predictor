"""A fake stand-in for run_cv_window.py's CLI contract, used only by
tests/test_run_cv_window.py to test run_windows_concurrent's concurrency and
merge-safety WITHOUT any real PyMC/NUTS sampling (which would make a
concurrency test slow and, worse, flaky on timing). Mimics just enough of
the real script: reads --checkpoint-path (always a fresh/private path in
the concurrent case, per run_windows_concurrent's design), writes one
deterministic result + one match-prediction for --window-index, after a
small artificial delay so concurrency is actually observable via wall time.
"""
import argparse
import json
import pickle
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--window-index", required=True, type=int)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--fail", action="store_true", help="simulate a failed window")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    time.sleep(args.delay)

    if args.fail:
        raise RuntimeError(f"simulated failure for window {args.window_index}")

    overrides = json.loads(args.config_json)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = {
        "results": [{"window": args.window_index, "mae": 0.1 * args.window_index, "overrides": overrides}],
        "cv_match_predictions": [{"window": args.window_index, "lambda_home": 1.0, "lambda_away": 1.0}],
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)


if __name__ == "__main__":
    main()
