"""WP051 — the market channel with a LONGER WARM-UP. The untried standard remedy.

THE SITUATION. WP039's market channel is worth **-0.00085 [-0.00207,+0.00031], better in
10/10 leave-one-season-out folds** -- three times the size of the xG-weight change that
WAS shipped (-0.00029). It was not shipped for one reason: convergence. 38 of 115 windows
exceed rhat 1.1 against the champion's 3/115 (median 1.073 vs 1.032), and fits are 3.5x
slower.

WHAT WAS NEVER TRIED. The project has run 600 tune / 600 draws since WP023 and never
moved off it. For a stiff posterior, MORE WARM-UP is the textbook first remedy -- the
sampler needs longer to adapt its mass matrix and step size. Instead I jumped straight to
re-specifying the shape prior (WP039b, arm T100), which fixed the single worst window on
a one-window screen and did NOT generalise: across 115 windows the median rhat was
unchanged and windows above 1.1 went UP, 32 -> 38.

The stiffness is real and intrinsic: the market's implied rate tracks the model's lambda
to ~+-10%, so `mkt_shape` fits at 100+ and the Gamma likelihood is sharp almost by
construction. That is exactly the geometry a longer warm-up exists to handle.

ONE CHANGE FROM T100. T100 is the best market arm (-0.00085, max rhat 1.309, shape prior
Gamma(8,2)); this is T100 with tune 600 -> 1500 and nothing else, so the comparison
isolates warm-up length.

PREDICTION ON RECORD. I expect rhat to come down materially -- that is what warm-up does
-- and the RPS point estimate to move very little, because the posterior being sampled is
the same one. The purpose is to make the -0.00085 TRUSTWORTHY, not larger. If rhat does
NOT improve, the geometry is beyond warm-up and the market channel should be abandoned
rather than tuned further.
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

DATA = HERE.parent / "wp039_market_channel" / "cv_shared_data_market.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
# The NEW champion: xG at 0.75 (WP032), deep 0.5, stakes effort.
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.75,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
# Weight points, because WP032 proved the 0.5 hedge was UNDER-set for xG and this
# channel has no reason to share xG's optimum. M050 is the direct analogue of the
# other channels' default; M100/M025 bracket it.
ARMS = {
    "T1500": {**CHAMPION, "use_market_channel": True, "market_channel_weight": 1.0,
              "market_channel_shape_alpha": 8.0, "market_channel_shape_beta": 2.0},
}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS[arm]), "--predict-mode", "plugin",
           *sampler_args({**SAMPLER, "tune": 1500})]
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
