"""WP030 — squad intactness, the one arm.

THE IDEA. Every absence test so far (WP011 lineups, WP012 absence covariate, WP016
keepers) asked the same question: does knowing who is missing help PREDICT this match?
All null. This asks the opposite question: does knowing who was missing help the model
learn correctly from the PAST? When a depleted side loses, the attack latent currently
absorbs the whole result as evidence of poor quality. It should not. The covariate
gives that result somewhere else to go, so the latent stays a measure of the team's
real strength.

WHY IT IS USABLE AT OPENING PRICES -- the constraint that killed the lineup direction.
The term is b*(intact - 1). At prediction a normal XI is assumed, intact = 1, and the
term is EXACTLY ZERO; `predict.py` therefore does not implement it at all (a test
guards that). No team-sheet information about the priced match is ever consumed. The
covariate is a TRAINING-ROW CORRECTION, not a feature.

CONSTRUCTION. EWMA of each player's minutes share (half-life 15 matches, chosen on the
downstream objective, NOT on "predict this match's minutes" -- that criterion picked
half-life 3, retaining 31% after five missed matches, i.e. it defines an injured player
as unimportant). Non-appearances expand to zero minutes, because an injured player is
simply absent from the roster. Team intactness is the available importance divided by
the team's OWN prior EWMA norm -- self-normalised, because summing whole-squad
importance measures squad DEPTH, which is exactly WP011's `absent_share` error.
Result: mean 0.987, p5 0.80, 15.4% of matches below 0.9.

PREMISE (residual correlation, n=3,780, team-clustered) -- the strongest in the project:
  intactness differential     r = +0.1469 [+0.1093, +0.1875]   REAL
  either side below 0.9       r = +0.201   (where depletion actually happens)
  both sides near normal      r = +0.128
  for scale: Elo +0.047, stakes +0.040, congestion -0.020 (null)

PREDICTION ON RECORD. Genuinely uncertain. The premise is 3x anything that has worked,
but an ADDITIVE term on theta is the exact form that has failed four times, fitting at
~2% of its prior SD each time. My defence is that those arms competed with latents that
had already explained the match, whereas this one corrects the latents themselves using
information they cannot access. But it is the same algebra, and that should make us
nervous. Expect the coefficient to fit POSITIVE and well away from zero (unlike the
four nulls) even if RPS moves little -- a near-zero coefficient would mean the premise
correlation is being absorbed by the walk after all.

BASELINE is WP028's I_effort, which IS the champion (L0 + xG + deep + stakes_effort),
115 windows already fitted. Verified: this data file is bit-identical to WP028's apart
from the intact/rest/load columns, and rest/load are inert with use_congestion=False.

    python work_products/wp030_squad_intactness/run_cv.py [--workers 6]
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

DATA = HERE / "cv_shared_data_intact.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.5,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
ARMS = {
    "M_intact":  {**CHAMPION, "use_intactness": True},
    # WP030b -- the same term centred on the TRAINING MEAN (0.989) instead of 1.0.
    # M_intact lost +0.00090 UNIFORMLY across every slice, and the diagnosis is that
    # centring on 1.0 predicts every match at FULL strength while the training mean
    # is 0.989: a constant +1.1% inflation of both rates. An unknown XI means AVERAGE
    # availability, not a full squad. Expect this to recover part of the loss -- but
    # only part, because the second mechanism is unfixable here: depletion PERSISTS
    # (r=+0.09..+0.15; after a depleted match 24% are still depleted vs 13% base), so
    # the champion's "contaminated" latents carry a real forecast of next-match
    # availability, and cleaning them discards it with nothing to put back.
    "N_intact_centred": {**CHAMPION, "use_intactness": True,
                         "intactness_center_mean": True},
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
    args = ap.parse_args()
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    wins = ([int(x) for x in args.windows.split(",")] if args.windows
            else list(range(1, n + 1)))
    tasks = [(a, w) for w in wins for a in ARMS]
    print(f"{len(tasks)} tasks = {len(ARMS)} arm x {len(wins)} windows, "
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
