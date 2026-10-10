"""WP035 — the season boundary: how much extra uncertainty at a season start?

THE LEAD, and it is the biggest one-window signal the audit has produced. WP034's
second screen (12 fits, one window, ~25 min) swept the structural constants:

    season_start_sigma_mult  3.0 -> 1.0    dRPS -0.00270     <- 2.8x the MDE
    season_start_sigma_mult  3.0 -> 6.0    dRPS +0.00053
    init_scale               0.2 -> 0.5    dRPS -0.00261     <- 2.8x the MDE
    init_scale               0.2 -> 0.05   dRPS +0.00175
    season_start_window      3 or 8        dRPS +0.00000  (DEAD CONFIG, see below)
    xg_channel_shape         tighter       no posterior movement

Both leads have a CLEAN DIRECTION with the opposite perturbation confirming it, which
is what screen 1 lacked -- there every dRPS was ~1x MDE and the implied direction turned
out to be WRONG for the one parameter that moved. These are ~2.8x MDE. Still one window,
so this sweep decides, not the screen.

WHAT THE TWO CONSTANTS DO.
  `season_start_sigma_mult` = 3.0 multiplies the latent walk's innovation SD during each
  season's first 5 rounds, to let teams jump after summer transfers. The screen says
  that inflation is TOO BIG: removing it entirely (1.0) is worth -0.00270 on one window,
  and sigma_att rises 0.0204 -> 0.0265 to compensate, i.e. the model would rather have a
  uniformly slightly-more-volatile walk than a big seasonal jolt.

  `init_scale` = 0.2 is the SD of a team's latent at t=0. The screen says it is TOO
  SMALL: 0.5 is worth -0.00261. A tight t=0 prior shrinks every team toward average at
  the start of history, and that error propagates forward through the AR(1).

WHY THIS IS A PLAUSIBLE PLACE FOR A REAL WIN, unlike the ~20 null arms: both constants
govern ENTRY and SEASON-BOUNDARY uncertainty, and that is the one population where this
project has already measured a large effect -- WP025's Elo entry prior fitted at 169% of
its prior SD (it was declined on external-data grounds, not on performance). These two
constants address the same weakness with NO external data.

    SSM_1    season_start_sigma_mult 1.0  (no inflation at all)
    SSM_2    season_start_sigma_mult 2.0  (between 1.0 and the current 3.0)
    INIT_50  init_scale 0.5
    INIT_100 init_scale 1.0

A DEAD CONFIG WAS FOUND AND FIXED. `ModelConfig.season_start_window` was declared but
NEVER READ: `prepare_model_data` used its own default of 5 and the harness never passed
the config value, so setting it did nothing -- which is exactly why the screen measured
+0.00000 for both 3 and 8. Now wired at both call sites, behaviour-preserving since both
defaults are 5, and guarded by two tests. It also validates the screen: a genuinely
inert parameter reads as exact zero.

PREDICTION ON RECORD. I expect at least one of these to clear the bar -- this is the
first time in the audit that a screen has shown a multi-MDE effect with a confirming
opposite direction. My guess is SSM_1 or SSM_2 wins and init_scale turns out smaller
than the screen implies, because init_scale only affects t=0 whose influence decays
through the AR(1), whereas season_start_sigma_mult fires at EVERY season boundary in
every window. I have been wrong on direction in 4 of the last 5 predictions, so treat
that ordering as a guess, not a forecast.

CAVEAT, same as the xG/deep 2D problem: these two constants both govern early-season
uncertainty and may interact. Sweeping each against the champion gives two 1D slices,
not a joint optimum.

    python work_products/wp035_season_boundary/run_cv.py [--workers 6]
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
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
# The NEW champion: xG at 0.75 (WP032), deep 0.5, stakes effort.
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.75,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
ARMS = {
    "SSM_1":    {**CHAMPION, "season_start_sigma_mult": 1.0},
    "SSM_2":    {**CHAMPION, "season_start_sigma_mult": 2.0},
    "INIT_50":  {**CHAMPION, "init_scale": 0.5},
    "INIT_100": {**CHAMPION, "init_scale": 1.0},
}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
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
