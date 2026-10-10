"""WP013 — isolate the xG measurement channel.

WP007's ladder was cumulative: L5_xgchannel = L0 + posterior prediction +
intercept/centering + random walk + promotion offset + xG channel. Only the
last rung showed an effect, but the other four were never separated out, so
"L0 + xG channel alone" has never been measured. This runs exactly that, with
L0's own prediction mode, so the only difference from L0 is the xG channel.

    python work_products/wp013_xg_channel_robustness/run_cv.py [--workers 4]
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WP007 = HERE.parent / "wp007_structural_fixes"
sys.path.insert(0, str(WP007))
sys.path.insert(0, str(HERE.parent))
from notify import done as _notify_done  # noqa: E402
from arms import SAMPLER_LEGACY as SAMPLER  # noqa: E402

DATA = WP007 / "cv_shared_data.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
ARM = "S3_xgchannel_only"
# L0's config verbatim (clip_theta/center come from the harness defaults) + ONLY the xG channel
OVERRIDES = {"use_xg_channel": True, "xg_channel_weight": 0.5}
PREDICT_MODE = "plugin"   # same as L0, so the xG channel is the only difference


def run_task(w):
    ck = CKPT / ARM / f"w{w:03d}.pkl"
    if ck.exists():
        return w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(OVERRIDES), "--predict-mode", PREDICT_MODE,
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
        return w, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        ck.unlink(missing_ok=True)
        return w, f"FAILED {type(e).__name__}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    print(f"{n} windows, {args.workers} at a time  (arm: {ARM}, predict={PREDICT_MODE})", flush=True)
    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(run_task, w) for w in range(1, n + 1)]):
            w, status, secs = fu.result()
            done += 1
            failed += status.startswith("FAILED")
            print(f"[{done}/{n}] w{w:03d}: {status} ({secs:.0f}s)  elapsed {(time.time()-t0)/60:.0f}m", flush=True)
    print(f"finished: {done - failed} ok/cached, {failed} failed")
    _notify_done(HERE.name, f"{done - failed}/{len(tasks)} fits ok, {failed} failed, "
                            f"{(time.time()-t0)/60:.0f}m")


if __name__ == "__main__":
    main()
