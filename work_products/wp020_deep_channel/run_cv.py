"""WP020 — does observing DEEP COMPLETIONS help?

Three arms against WP013's existing `S3_xgchannel_only` checkpoints (no refit).

  H_deep_added   xG channel + deep channel         the hypothesis
  I_deep_only    deep channel INSTEAD of xG        which channel carries it
  J_deep_heavy   both, deep channel at weight 1.0  the weight ladder

The ladder is included because it was the most diagnostic part of WP019: a channel
carrying real information should improve as tempering is relaxed, at least
initially. WP019's shot channel degraded monotonically (0.20481 -> 0.20497 ->
0.20577), which said more than any single CI did.

`I_deep_only` is the control that makes `H` interpretable (BACKLOG caution 2).

PREDICTION ON RECORD (made before running): H beats the baseline by MORE than
0.0005 RPS, and the gain concentrates in the high-line-movement quartile where
WP010 localised the deficit. Point estimate -0.0015; the per-match premise check
extrapolates to -0.0030 [-0.0062, -0.0006] and I am shading it toward zero because
that framework is pooled across teams, so part of deep's measured value may be team
IDENTIFICATION the attack/defence latents already carry. A null would indicate
exactly that, and would discredit the premise-check approach.

Cost: two scalars, zero new latents, so ~WP019's 141 min at 4 workers.

    python work_products/wp020_deep_channel/run_cv.py [--workers 4]
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

DATA = HERE / "cv_shared_data_deep.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"

ARMS_WP020 = {
    "H_deep_added": {"use_xg_channel": True, "xg_channel_weight": 0.5,
                     "use_deep_channel": True, "deep_channel_weight": 0.5},
    "I_deep_only":  {"use_xg_channel": False,
                     "use_deep_channel": True, "deep_channel_weight": 0.5},
    "J_deep_heavy": {"use_xg_channel": True, "xg_channel_weight": 0.5,
                     "use_deep_channel": True, "deep_channel_weight": 1.0},
}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS_WP020[arm]), "--predict-mode", "plugin",
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
    ap.add_argument("--arms", default=",".join(ARMS_WP020))
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
