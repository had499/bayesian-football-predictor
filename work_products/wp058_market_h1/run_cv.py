"""WP058 — the MARKET CHANNEL on the 343-window horizon-1 grid.

WHY THIS RUN EXISTS. WP039 measured the market channel at -0.00079 (10/10 LOSO) on the
115-window TEST_LEN=3 grid, but it was never shipped: rhat 1.41 and P(improves)~90%. The
gap to Pinnacle's opening line is now quoted on the TEST_LEN=1 grid (+0.00282), so the
channel's effect size and the gap it is meant to close are currently measured on DIFFERENT
protocols. This run puts them on the same one.

CONFIGURATION, combining the two things later work established:
  weight 0.5          WP039's weight sweep: 94% of the gain at weight 1.0 with materially
                      better conditioning (the cost and the rhat problem share a root
                      cause -- ill-conditioning, tree depth 6->8, steps/draw 64->256).
  shape prior (8,2)   WP051's T1500 tightening, which was the fix for the rhat.

BASELINE is wp055's `H1_CHAMP`, already on disk, run from THIS SAME data file and grid —
so the contrast is paired and within-protocol. Only the market arm needs fitting.

NOT SCREENABLE ON THE CLOSE. WP057's close-line metric is 4-5x more powerful but is
CIRCULAR for a market-fed arm (its close-gain/outcome-gain ratio is 0.77 against xG's
0.32 — that is the inflation). This arm must be judged on outcome-RPS, which is exactly
why it needs a full 343-window sweep rather than cheap post-processing.
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
sys.path.insert(0, str(HERE.parent))
from notify import done as _notify_done  # noqa: E402
sys.path.insert(0, str(HERE.parent / "wp007_structural_fixes"))
from arms import SAMPLER, sampler_args  # noqa: E402

DATA = HERE.parent / "wp055_horizon1" / "cv_shared_data_h1.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
# The NEW champion: xG at 0.75 (WP032), deep 0.5, stakes effort.
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.75,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
# Weight points, because WP032 proved the 0.5 hedge was UNDER-set for xG and this
# channel has no reason to share xG's optimum. M050 is the direct analogue of the
# other channels' default; M100/M025 bracket it.
ARMS = {"H1_MKT050": {**CHAMPION, "use_market_channel": True,
                      "market_channel_weight": 0.5,
                      "market_channel_shape_alpha": 8.0,
                      "market_channel_shape_beta": 2.0}}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS[arm]), "--predict-mode", "plugin",
           *sampler_args()]
    env = dict(os.environ)
    env["PYTENSOR_FLAGS"] = "cxx="
    env.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={SAMPLER['chains']}")
    t0 = time.time()
    try:
        with open(ck.parent / f"w{w:03d}.log", "w") as lf:
            subprocess.run(cmd, check=True, timeout=90 * 60, env=env, stdout=lf, stderr=subprocess.STDOUT)
        return arm, w, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        ck.unlink(missing_ok=True)
        return arm, w, f"FAILED {type(e).__name__}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--windows", type=str, default="")
    ap.add_argument("--arms", type=str, default="")
    args = ap.parse_args()
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    wins = ([int(x) for x in args.windows.split(",")] if args.windows else list(range(1, n + 1)))
    arms = args.arms.split(",") if args.arms else list(ARMS)
    # Window-major so every arm advances together: a partial sweep stays a usable
    # paired comparison. (It must still NOT be read for direction -- WP032's partial
    # read at 21/115 windows pointed the opposite way to the final result.)
    tasks = [(a, w) for w in wins for a in arms]
    print(f"{len(tasks)} tasks = {len(arms)} arms x {len(wins)} windows, "
          f"{args.workers} at a time", flush=True)
    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(run_task, a, w) for a, w in tasks]):
            a, w, status, secs = fu.result()
            done += 1
            failed += status.startswith("FAILED")
            el = (time.time() - t0) / 60
            print(f"[{done}/{len(tasks)}] {a} w{w:03d}: {status} ({secs:.0f}s)  "
                  f"elapsed {el:.0f}m  eta {el/done*(len(tasks)-done):.0f}m", flush=True)
    print(f"finished: {done - failed} ok/cached, {failed} failed")
    _notify_done(HERE.name, f"{done - failed}/{len(tasks)} fits ok, {failed} failed, "
                            f"{(time.time()-t0)/60:.0f}m")


if __name__ == "__main__":
    main()
