"""WP021 — the high-power confirmation run for the deep-completion channel.

WP020's result (-0.00096 RPS [-0.00248, +0.00063]) was the best since the xG channel
but sat BELOW that design's +-0.00156 detection threshold. 115 windows drops the
threshold to ~0.00080, so if the effect is real at WP020's size it should now clear
zero: predicted CI roughly [-0.00176, -0.00016].

Three arms, all against a baseline refitted on the SAME grid (old checkpoints are at
different train_end values and cannot be reused):

  S3_xgchannel_only  goals + xG                    the shipping config / baseline
  H_deep_added       goals + xG + deep             WP020's result, re-tested at power
  K_deep_shots       goals + xG + deep + shots     are the three channels complementary?

K is included because WP019's shot channel was null ALONE but its per-match
information was not negative, and `deep`-on-top-of-shots retained 98% of its
premise-check value. If the three are complementary, K is the best config available.

Tasks are interleaved by window, so stopping early still leaves a valid paired
comparison on the windows completed so far.

Cost: ~3x WP020's per-arm total training rows. Estimate ~7h at 4 workers, but treat
that as soft — BACKLOG caution 4 (WP016 was predicted at 50 min and took 702).
Fully resumable.

    python work_products/wp021_power_confirmation/run_cv.py [--workers 4]
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

DATA = HERE / "cv_shared_data_big.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"

ARMS_WP021 = {
    "S3_xgchannel_only": {"use_xg_channel": True, "xg_channel_weight": 0.5},
    "H_deep_added":      {"use_xg_channel": True, "xg_channel_weight": 0.5,
                          "use_deep_channel": True, "deep_channel_weight": 0.5},
    "K_deep_shots":      {"use_xg_channel": True, "xg_channel_weight": 0.5,
                          "use_deep_channel": True, "deep_channel_weight": 0.5,
                          "use_shot_channel": True, "shot_channel_weight": 0.5},
}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS_WP021[arm]), "--predict-mode", "plugin",
           "--draws", str(SAMPLER["draws"]), "--tune", str(SAMPLER["tune"]),
           "--chains", str(SAMPLER["chains"]), "--target-accept", str(SAMPLER["target_accept"])]
    env = dict(os.environ)
    env["PYTENSOR_FLAGS"] = "cxx="
    env.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={SAMPLER['chains']} "
                                "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    t0 = time.time()
    try:
        with open(ck.parent / f"w{w:03d}.log", "w") as lf:
            subprocess.run(cmd, check=True, timeout=45 * 60, env=env, stdout=lf, stderr=subprocess.STDOUT)
        return arm, w, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        ck.unlink(missing_ok=True)
        return arm, w, f"FAILED {type(e).__name__}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", default=",".join(ARMS_WP021))
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
            el = (time.time() - t0) / 60
            eta = el / done * (len(tasks) - done) if done else 0
            print(f"[{done}/{len(tasks)}] {a} w{w:03d}: {status} ({secs:.0f}s)  "
                  f"elapsed {el:.0f}m  eta {eta:.0f}m", flush=True)
    print(f"finished: {done - failed} ok/cached, {failed} failed")
    _notify_done(HERE.name, f"{done - failed}/{len(tasks)} fits ok, {failed} failed, "
                            f"{(time.time()-t0)/60:.0f}m")


if __name__ == "__main__":
    main()
