"""WP032 — tune the xG channel's WEIGHT. The one component that works has never
been tuned, and the evidence that was used to stop tuning it was under-powered.

THE LEAD. WP007 tested exactly two weights and compared them on its 30-window grid:

    xG weight 1.0 vs 0.5   -0.00052  [-0.0019, +0.0009]   "no clear difference"

The point estimate says MORE xG is BETTER. It was filed as null because the CI spanned
zero -- but that grid's MDE was +-0.0019, nearly four times the effect. WP013 then
explicitly declined to revisit it ("weight 0.5 vs 1.0 was already null in WP007"), so
0.5 has been frozen ever since on the strength of an under-powered null. At 115 windows
the MDE is +-0.00095, so a -0.00052 effect is now inside reach.

This matters because the xG channel is the ONLY established win in the project
(-0.00258, 6/6 leave-one-season-out folds) and the whole remaining gap to Pinnacle's
opening line is +0.00344. Getting its one free parameter wrong is the cheapest possible
way to be leaving performance on the table: no new data, no new code, pure config.

WHAT THE WEIGHT MEANS. The channel is a tempered Potential: `weight * logp`. Goals and
xG share the same shot-level information, so weight=1 ("full trust") treats xG as a
fully independent second observation and double-counts. 0.5 was chosen as a hedge
against that double-counting, NOT measured. Weight > 1 is equivalently a DOWN-weighting
of the goals likelihood relative to xG, which is defensible if xG is the better signal
of the latent rate -- and WP007's own premise says it is (first-half xG predicts
second-half goals better than first-half goals do: r=0.73 vs 0.68 attack, 0.62 vs 0.55
defence).

SMOKE TEST (window 70) REDIRECTED THIS. All five points were fitted first:

    w0.25  0.18415    w0.50  0.18412 (champion)    w1.00  0.18736
    w1.75  0.20016    w3.00  0.19702
    xg_kappa drifts 1.018 -> 1.014 -> 1.007 -> 0.958 -> 0.933

Higher weight is clearly WORSE, not better, and the mechanism is visible: as the
channel overpowers the goals likelihood it drags its own calibration constant kappa
away from 1, i.e. it starts explaining xG at the expense of explaining goals.

W175 and W300 were therefore DROPPED without a full sweep: +0.016 and +0.013 on one
window is 13-17x the MDE, far outside single-window noise, and two independent arms
agree. Note the asymmetry -- one window is enough to RULE OUT an effect that large,
but would not be enough to crown a winner (BACKLOG caution 15).

WHY WP007's RESULT PROBABLY DOES NOT TRANSFER: its L5 config had NO deep channel. The
champion now carries TWO measurement channels, so upweighting xG over-constrains the
same latents that deep is already informing. The 0.5 default may have been lucky, but
it is not obviously wrong, and the live question is now much narrower.

THE ACTUAL SWEEP, bracketing 0.5 tightly where the optimum must now lie:
    0.25  |  0.5 (champion, already fitted)  |  0.75  |  1.0

  W075 is the real candidate: the smoke curve is flat between 0.25 and 0.5 and turns
       down by 1.0, so any remaining gain is in 0.5-1.0.
  W100 settles WP007's dismissed claim with proper power rather than leaving it open.
  W025 is the falsification on the low side: if LESS xG is also better than 0.5, the
       weight is not doing what this work package assumes and the whole reading is wrong.

PREDICTION ON RECORD, REVISED AFTER THE SMOKE. My first prediction was that the curve
peaks in 1.0-2.0 and W100 lands near -0.0005. The smoke falsified that outright. I now
expect all three arms to come back NULL, with the true optimum flat somewhere in
0.3-0.6 and no exploitable gain -- i.e. the 0.5 default was fine and WP007's -0.00052
was the noise it was called. Running it anyway because "the weight has never been
tuned with power" is worth CLOSING properly rather than leaving as a known unknown, and
because the flat region could still hide a -0.0005 at w0.75.

The deep channel's weight is held at 0.5 throughout. If xG's optimum moves, deep's
probably should be retuned too -- that is a follow-up, not this sweep.

    python work_products/wp032_xg_weight/run_cv.py [--workers 6]
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
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.5,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
WEIGHTS = {"W025": 0.25, "W075": 0.75, "W100": 1.0, "W175": 1.75, "W300": 3.0}
ARMS = {k: {**CHAMPION, "xg_channel_weight": w} for k, w in WEIGHTS.items()}


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
    # Interleave windows so every arm advances together -- a partial sweep is then
    # still a usable paired comparison instead of one finished arm and three empty.
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
