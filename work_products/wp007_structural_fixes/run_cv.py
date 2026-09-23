"""Run the WP007 arms x windows sweep, resumably, N subprocesses at a time.

    python work_products/wp007_structural_fixes/run_cv.py [--workers 4] [--arms A B ...]

Each (arm, window) is its own `scripts/run_cv_window.py` subprocess writing
its own checkpoint under `checkpoints/<arm>/w<NN>.pkl`, so tasks are
independent (no shared-file races), a killed/hung task is simply retried on
the next invocation, and slow arms don't hold up fast ones. The analysis
notebook merges the per-window checkpoints.

Data: `cv_shared_data.pkl` (df_cv + windows) is built on first run from
Understat — set REQUESTS_CA_BUNDLE first if you're behind TLS interception.
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

WP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WP_DIR.parents[1]
sys.path.insert(0, str(WP_DIR))
from arms import ARMS, SAMPLER  # noqa: E402

DATA_PATH = WP_DIR / "cv_shared_data.pkl"
CKPT_DIR = WP_DIR / "checkpoints"
SCRIPT = REPO_ROOT / "scripts" / "run_cv_window.py"

YEARS = [str(y) for y in range(2014, 2026)]   # 12 EPL seasons of history
FIRST_TEST_SEASON = "2020"                    # test period matches WP001-WP006 (2020-21 onward)
STEP = 7                                      # a window every 7 rounds ...
TEST_LEN = 4                                  # ... each forecasting the next 4 rounds
TASK_TIMEOUT_S = 40 * 60


def build_shared_data():
    from football_model.data.get_data import get_understat_data
    from football_model.features.add_metadata import add_home_away_goals_xg, add_match_ids, add_rounds_to_data

    df = get_understat_data(leagues=["EPL"], years=YEARS)
    df = add_home_away_goals_xg(add_match_ids(add_rounds_to_data(df)))
    df = df.sort_values("datetime").reset_index(drop=True)

    first_test_round = int(df.loc[df["season"] == FIRST_TEST_SEASON, "round"].min()) + 2
    last_round = int(df["round"].max())
    windows, train_end = [], first_test_round - 1
    while train_end + 1 <= last_round:
        test_start = train_end + 1
        windows.append(dict(train_start=1, train_end=train_end, test_start=test_start,
                            test_end=min(test_start + TEST_LEN - 1, last_round)))
        train_end += STEP
    return {"df_cv": df, "windows": windows}


def load_shared():
    if not DATA_PATH.exists():
        shared = build_shared_data()
        with open(DATA_PATH, "wb") as f:
            pickle.dump(shared, f)
        print(f"built {DATA_PATH.name}: {len(shared['df_cv'])} rows, {len(shared['windows'])} windows")
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


NO_CXX = False   # set from --no-cxx in main(); makes subprocesses skip PyTensor's C compiler


def run_task(arm: str, window_index: int, n_windows: int):
    cfg = ARMS[arm]
    ckpt = CKPT_DIR / arm / f"w{window_index:03d}.pkl"
    if ckpt.exists():
        return arm, window_index, "cached", 0.0
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT),
        "--data-path", str(DATA_PATH), "--checkpoint-path", str(ckpt),
        "--window-index", str(window_index),
        "--use-xg", str(cfg["use_xg"]).lower(), "--use-dc", str(cfg["use_dc"]).lower(),
        "--config-json", __import__("json").dumps(cfg["overrides"]),
        "--predict-mode", cfg["predict_mode"],
        "--draws", str(SAMPLER["draws"]), "--tune", str(SAMPLER["tune"]),
        "--chains", str(SAMPLER["chains"]), "--target-accept", str(SAMPLER["target_accept"]),
    ]
    env = dict(os.environ)
    if NO_CXX:
        env["PYTENSOR_FLAGS"] = "cxx="
    # One single-threaded XLA device per chain. Left at defaults, every device
    # starts a full-width Eigen thread pool; with several tasks running at once
    # that oversubscribes the machine (measured: load avg 50 on 18 cores and
    # ~5x slower tasks). The model's scan over rounds is sequential, tiny-op
    # work that gains nothing from intra-op threads anyway.
    env.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={SAMPLER['chains']} "
                                "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    t0 = time.time()
    log = CKPT_DIR / arm / f"w{window_index:03d}.log"
    try:
        with open(log, "w") as lf:
            subprocess.run(cmd, check=True, timeout=TASK_TIMEOUT_S, env=env, stdout=lf, stderr=subprocess.STDOUT)
        return arm, window_index, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001 — a failed task must not kill the sweep
        if ckpt.exists():
            ckpt.unlink()
        return arm, window_index, f"FAILED {type(e).__name__}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--max-windows", type=int, default=None, help="smoke-test: only the first N windows")
    ap.add_argument("--no-cxx", action="store_true",
                    help="set PYTENSOR_FLAGS=cxx= in every task (use if tasks die in seconds with a clang "
                         "'linker command failed' error in checkpoints/<arm>/wNNN.log)")
    args = ap.parse_args()
    global NO_CXX
    NO_CXX = args.no_cxx

    shared = load_shared()
    n_w = len(shared["windows"]) if args.max_windows is None else min(args.max_windows, len(shared["windows"]))
    # interleave arms window-by-window so every arm progresses together and a
    # partial run still yields a paired comparison on the early windows
    tasks = [(a, w) for w in range(1, n_w + 1) for a in args.arms]
    print(f"{len(tasks)} tasks = {len(args.arms)} arms x {n_w} windows, {args.workers} at a time", flush=True)

    done = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_task, a, w, n_w) for a, w in tasks]
        for fu in as_completed(futs):
            arm, w, status, secs = fu.result()
            done += 1
            failed += status.startswith("FAILED")
            print(f"[{done}/{len(tasks)}] {arm} w{w:03d}: {status} ({secs:.0f}s)  elapsed {(time.time()-t0)/60:.0f}m", flush=True)
    print(f"finished: {done - failed} ok/cached, {failed} failed")


if __name__ == "__main__":
    main()
