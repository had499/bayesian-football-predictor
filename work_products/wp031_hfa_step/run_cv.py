"""WP031 — home advantage: is the defect a crowd STEP, or just a bad PRIOR?

WP015 tested time-varying HFA as a slow RANDOM WALK: null. The untested specification
is a STEP with a known external breakpoint, motivated by the COVID regime shift. The
premise check (premise.py) both supported and demolished parts of that:

  champion out-of-sample HOME TILT  (goals/match; >0 = under-rates home teams)
    pre-covid   n=1408   +0.0518 [-0.0320,+0.1336]   home win 0.471
    CROWDLESS   n= 434   -0.2017 [-0.3789,-0.0454]   home win 0.392   <- REAL, CI excl 0
    post-covid  n=1938   -0.0366 [-0.1277,+0.0530]   home win 0.442

  SUPPORTED: the crowdless window is a real, large, CI-excluding-zero defect.
  NOT SUPPORTED: a PERMANENT step down. pre and post CIs both span zero and overlap
    heavily (difference -0.088 goals, well inside the noise).
  DEMOLISHED by the season table: 2024 had FULL crowds and a tilt of -0.2251, almost
    as extreme as crowdless 2020's -0.2757, while 2022 was +0.1250. Season-level HFA
    variation is large and NOT aligned with crowd status, so "COVID" is not a special
    explanation -- it is one draw from a volatile distribution. That is why the
    season-level arm is NOT included: it is WP015's rejected drift in another coat.

  THE ACTUAL FINDING, which the COVID story was obscuring:
    home_mu posterior sits at 0.2100 against a prior of Normal(0.13, 0.03).
    That is 2.67 PRIOR SDs above the centre -- the prior is fighting the data in
    every one of the 115 windows, and shrinking the HFA estimate toward a value the
    data has never supported. Cheap to fix and it applies to EVERY match, not just
    the 434 crowdless ones.

ARMS
  B_hfa_wide   home_sd 0.03 -> 0.15, i.e. widen the prior on the league-wide HFA mean
               and let each window's data decide. DELIBERATELY WIDENED, NOT RECENTRED:
               recentring on 0.21 would import a number derived from fits whose
               training sets overlap earlier windows' test sets. Widening is leak-free
               and tests the same defect. No code change -- a pure config override.

  A_crowd      an explicit crowdless indicator on the HFA (needs code + a predict-side
               mirror, unlike WP030: crowd status IS observable at the opening price,
               so predict.py SHOULD implement it).

PREDICTION ON RECORD. B_hfa_wide is the better bet and the one I would ship if either
works. A_crowd fixes a real but FINISHED regime -- its only forward value is that it
stops 434 crowdless matches from dragging home_mu down for 2025-26, which is a second-
order effect on 3,780 test matches. Expect A to show a visible gain CONFINED to 2020-21
and near-zero elsewhere; if it gains outside that window, something is wrong.

    python work_products/wp031_hfa_step/run_cv.py [--workers 6]
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

DATA = HERE.parent / "wp030_squad_intactness" / "cv_shared_data_intact.pkl"
DATA_CROWD = HERE / "cv_shared_data_crowd.pkl"   # same file + the crowdless column
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.5,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
ARMS = {
    "B_hfa_wide": {**CHAMPION, "home_sd": 0.15},
    "A_crowd":    {**CHAMPION, "use_crowd_regime": True},
}
DATA_FOR = {"B_hfa_wide": DATA, "A_crowd": DATA_CROWD}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA_FOR[arm]), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS[arm]), "--predict-mode", "plugin", *sampler_args()]
    env = dict(os.environ)
    env["PYTENSOR_FLAGS"] = "cxx="
    env.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={SAMPLER['chains']}")
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
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--windows", type=str, default="")
    ap.add_argument("--arms", type=str, default="")
    args = ap.parse_args()
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    wins = ([int(x) for x in args.windows.split(",")] if args.windows else list(range(1, n + 1)))
    arms = args.arms.split(",") if args.arms else list(ARMS)
    tasks = [(a, w) for w in wins for a in arms]
    print(f"{len(tasks)} tasks = {len(arms)} arm(s) x {len(wins)} windows, "
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
