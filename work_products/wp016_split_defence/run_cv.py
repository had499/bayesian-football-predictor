"""WP012 — run the absence-covariate arm.

Only ONE new arm is fitted: L7_absence = L5_xgchannel + use_absence=True.
L5's own 30 windows already exist from WP007 and are a valid paired baseline —
verified that the WP012 data file yields bit-identical ModelData apart from the
absence arrays, which L5 (use_absence=False) ignores.

    python work_products/wp016_split_defence/run_cv.py [--workers 4]
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
sys.path.insert(0, str(HERE.parent))
from notify import done as _notify_done  # noqa: E402
from arms import ARMS, SAMPLER_LEGACY as SAMPLER  # noqa: E402

DATA_MATCH = HERE.parent / "wp007_structural_fixes" / "cv_shared_data.pkl"
DATA = DATA_MATCH   # every WP015 arm uses the original match-level xG
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
# Two arms, both against WP013's shipping config (L0 + xG channel only):
#   B_npxg          — channel observes non-penalty xG instead of raw xG
#   C_kappa_season  — channel keeps raw xG but calibrates kappa per season
# All three are WP013's shipping config (L0 + xG channel) with ONE thing changed.
#   D_shotxg       — same shot feed, penalties INCLUDED  (source control)
#   B_npxg         — same shot feed, penalties REMOVED   (the hypothesis)
#   C_kappa_season — match-level xG, kappa fitted per season
# B vs D isolates penalties with the data source held constant; D vs A (WP013's
# S3_xgchannel_only) says whether swapping source matters at all.
# All arms are WP013's shipping config (L0 + xG channel) PLUS hard centering,
# which time_varying_level requires. E_level_only isolates hard-centering+drifting
# level from the HFA change, so a win can be attributed to the right one.
# On top of WP013's shipping config (L0 + xG channel). H is the structural split;
# H_loose lets shot-stopping move as freely as suppression, to check the tighter
# sigma prior isn't itself the reason for any null.
ARMS_WP014 = {
    "H_split":       {"use_xg_channel": True, "xg_channel_weight": 0.5,
                      "split_defence": True, "stop_sigma_scale": 0.3},
    "H_split_loose": {"use_xg_channel": True, "xg_channel_weight": 0.5,
                      "split_defence": True, "stop_sigma_scale": 1.0},
}
DATA_FOR = {a: DATA_MATCH for a in ARMS_WP014}   # all use the original match-level xG


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA_FOR[arm]), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS_WP014[arm]), "--predict-mode", "plugin",
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
        return arm, w, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        ck.unlink(missing_ok=True)
        return arm, w, f"FAILED {type(e).__name__}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import pickle
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    tasks = [(a, w) for w in range(1, n + 1) for a in ARMS_WP014]
    print(f"{len(tasks)} tasks = {len(ARMS_WP014)} arms x {n} windows, {args.workers} at a time", flush=True)
    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(run_task, a, w) for a, w in tasks]):
            a, w, status, secs = fu.result()
            done += 1
            failed += status.startswith("FAILED")
            print(f"[{done}/{len(tasks)}] {a} w{w:03d}: {status} ({secs:.0f}s)  "
                  f"elapsed {(time.time()-t0)/60:.0f}m", flush=True)
    print(f"finished: {done - failed} ok/cached, {failed} failed")
    _notify_done(HERE.name, f"{done - failed}/{len(tasks)} fits ok, {failed} failed, "
                            f"{(time.time()-t0)/60:.0f}m")


if __name__ == "__main__":
    main()
