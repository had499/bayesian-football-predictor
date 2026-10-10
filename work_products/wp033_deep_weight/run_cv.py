"""WP033 — tune the DEEP channel's weight, now that xG's has moved.

THE LEAD, continued from WP032. `deep_channel_weight = 0.5` is the SAME unexamined
hedge WP032 just disproved for xG: goals and deep completions share possession-level
information, so 0.5 was chosen to avoid double-counting and was NEVER measured at any
value. WP020 selected the channel and WP021 re-tested it -- both at weight 0.5 only.

WHY IT IS NOT JUST A REPEAT OF WP032. Two things changed:

1. **xG's weight went UP (0.5 -> 0.75).** WP032 showed the channels COMPETE: upweighting
   xG past ~1.0 degrades the fit because it over-constrains the same latents deep is
   also informing, and the damage shows in `xg_kappa` drifting off 1.0. With xG now
   taking more of the available constraint, deep's optimum has probably moved DOWN.

2. **deep was never established in the first place.** It went into the champion on
   expected value: -0.00083 [-0.00177, +0.00013], better in only 6/10 seasons, and the
   effect HALVED when 2016 was dropped (WP021 could not confirm it at 115 windows).

So `D000` is the arm that matters most: with xG at 0.75, does deep still earn its place
at all? A null-or-better result there SIMPLIFIES the champion, which is worth as much as
a small RPS gain and is the one outcome nobody has tested.

ARMS, all at the NEW champion's xG weight of 0.75:
    D000  deep OFF entirely  (use_deep_channel=False; equivalent to weight 0 but
          without building a Potential that contributes nothing)
    D025  0.25
    0.50  the new champion -- ALREADY FITTED as WP032's W075, used as the baseline
    D075  0.75
    D100  1.0   -- ADDED AFTER THE SMOKE, see below

BASELINE is WP032's `W075` (xG 0.75 + deep 0.5 + stakes), which IS the current champion,
115 windows already on disk and bit-identical in every other respect.

SMOKE TEST (window 70) ALREADY CONTRADICTS MY PREDICTION:

    deep OFF  0.18873   d0.25  0.18623   d0.50  0.18470   d0.75  0.18464

Two things. First, deep clearly EARNS its place -- switching it off costs +0.0040 on this
window, so the D000 "simplify the champion" outcome looks unlikely. Second, the curve is
monotone RISING to 0.75 and flat between 0.5 and 0.75, i.e. deep's optimum has NOT moved
down as I predicted. D100 was added to probe above 0.75 rather than assume the peak.

D000 is deliberately KEPT in the full sweep despite the +0.0040 smoke. Unlike WP032 --
where w1.75/w3.00 were killed at 13-17x the MDE -- +0.0040 is only ~4x MDE on a single
window, and deep's whole established effect is just -0.00083, so most of that gap is
noise around a much smaller truth. It is also the most decision-relevant arm, since
WP021 could not confirm deep at 115 windows.

PREDICTION ON RECORD, REVISED. I expect deep's optimum to be at or above 0.5 and the whole
spread to be small -- most likely all three arms null, because deep's own effect is only
-0.00083 and a weight change can only move a fraction of that. My specific guess is
D025 comes in slightly negative and D000 slightly positive, i.e. deep is worth keeping
but at less than 0.5. The outcome I would find most useful is D000 landing null, because
dropping a channel that is not paying for itself is a real simplification.

A caveat I cannot resolve here: this is a 1D slice through a 2D (xG, deep) surface at
xG=0.75. If deep's optimum moves, the joint optimum may not be at xG=0.75 either, and a
proper 2D sweep is 16+ arms / 13h. Not worth it for effects this size -- but the
single-arm-at-a-time result should not be called "the optimum".

    python work_products/wp033_deep_weight/run_cv.py [--workers 6]
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
    "D000": {**CHAMPION, "use_deep_channel": False},
    "D025": {**CHAMPION, "deep_channel_weight": 0.25},
    "D075": {**CHAMPION, "deep_channel_weight": 0.75},
    "D100": {**CHAMPION, "deep_channel_weight": 1.0},
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
