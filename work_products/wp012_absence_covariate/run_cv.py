"""WP012 — run the absence-covariate arm.

Only ONE new arm is fitted: L7_absence = L5_xgchannel + use_absence=True.
L5's own 30 windows already exist from WP007 and are a valid paired baseline —
verified that the WP012 data file yields bit-identical ModelData apart from the
absence arrays, which L5 (use_absence=False) ignores.

    python work_products/wp012_absence_covariate/run_cv.py [--workers 4]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "wp007_structural_fixes"))
from arms import ARMS, SAMPLER  # noqa: E402

DATA = HERE / "cv_shared_data_absence.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
ARM = "L7_absence"
OVERRIDES = {**ARMS["L5_xgchannel"]["overrides"], "use_absence": True, "absence_sd": 0.5}


def run_task(w, n):
    ck = CKPT / ARM / f"w{w:03d}.pkl"
    if ck.exists():
        return ARM, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(OVERRIDES), "--predict-mode", "posterior",
           "--draws", str(SAMPLER["draws"]), "--tune", str(SAMPLER["tune"]),
           "--chains", str(SAMPLER["chains"]), "--target-accept", str(SAMPLER["target_accept"])]
    env = dict(os.environ)
    env["PYTENSOR_FLAGS"] = "cxx="
    env.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={SAMPLER['chains']} "
                                "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    t0 = time.time()
    try:
        with open(ck.parent / f"w{w:03d}.log", "w") as lf:
            subprocess.run(cmd, check=True, timeout=40 * 60, env=env, stdout=lf, stderr=subprocess.STDOUT)
        return ARM, w, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        ck.unlink(missing_ok=True)
        return ARM, w, f"FAILED {type(e).__name__}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import pickle
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    print(f"{n} windows, {args.workers} at a time  (arm: {ARM})", flush=True)
    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(run_task, w, n) for w in range(1, n + 1)]):
            _, w, status, secs = fu.result()
            done += 1
            failed += status.startswith("FAILED")
            print(f"[{done}/{n}] w{w:03d}: {status} ({secs:.0f}s)  elapsed {(time.time()-t0)/60:.0f}m", flush=True)
    print(f"finished: {done - failed} ok/cached, {failed} failed")


if __name__ == "__main__":
    main()
