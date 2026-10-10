"""WP019 — does observing SHOT COUNT on top of xG help?

Three arms, all WP013's shipping config (L0 + xG channel) with one thing changed.
The baseline is WP013's existing `S3_xgchannel_only` checkpoints — no refit.

  E_shots_added   xG channel + shot-count channel      the hypothesis
  F_shots_only    shot-count channel INSTEAD of xG     which channel carries it
  G_shots_heavy   both, shot channel at weight 1.0     is 0.5 undertrusting it?

F is the control that makes E interpretable (BACKLOG caution 2). If E beats the
baseline but F matches it, shot count is adding information total xG lacked. If F
alone matches the xG channel, the two are substitutes and E's gain is just extra
tempering weight — which G separates.

Cost: the shot channel adds TWO scalars and ZERO latents, so per-window cost
should track WP014's ~31 min/arm rather than WP016's blowup (BACKLOG caution 4).
90 tasks, ~95 min at 4 workers.

    python work_products/wp019_shot_channel/run_cv.py [--workers 4]
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
sys.path.insert(0, str(HERE.parent / "wp007_structural_fixes"))
sys.path.insert(0, str(HERE.parent))
from notify import done as _notify_done  # noqa: E402
from arms import SAMPLER_LEGACY as SAMPLER  # noqa: E402

DATA = HERE / "cv_shared_data_shots.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"

ARMS_WP019 = {
    "E_shots_added":  {"use_xg_channel": True, "xg_channel_weight": 0.5,
                       "use_shot_channel": True, "shot_channel_weight": 0.5},
    "F_shots_only":   {"use_xg_channel": False,
                       "use_shot_channel": True, "shot_channel_weight": 0.5},
    "G_shots_heavy":  {"use_xg_channel": True, "xg_channel_weight": 0.5,
                       "use_shot_channel": True, "shot_channel_weight": 1.0},
}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS_WP019[arm]), "--predict-mode", "plugin",
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
    ap.add_argument("--arms", default=",".join(ARMS_WP019))
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    tasks = [(a, w) for w in range(1, n + 1) for a in arms]
    print(f"{len(tasks)} tasks = {len(arms)} arms x {n} windows, {args.workers} at a time", flush=True)
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
