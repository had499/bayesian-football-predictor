"""WP028 — end-of-season stakes, two mechanisms.

WHY THIS IDEA IS DIFFERENT. Incentive exists only NOW. A mid-table side in May has
nothing to play for, and the latent walk learns "they played badly in April" only
AFTER those matches -- it is forward-looking in a way the walk cannot anticipate. And
it needs no new data: the league table is already in df_cv.

STAKES = how much THIS result moves the team's end-of-season probabilities. For each
(season, round) the remaining fixtures are simulated 3,000 times; the team's current
result is conditioned on (win/draw/loss) and P(outcome) recomputed. Stakes is the
spread, summed over relegation / Europe / title. That derives "nothing to play for"
from the arithmetic instead of hand-drawn categories, and separates "five points clear
with three to play" from "level on points". 23.9% of matches are dead rubbers.

PREMISE (residual correlation, n=3,780, team-clustered):
  total stakes differential   r = +0.0403 [+0.0111, +0.0673]   REAL
  April/May only (n=773)      r = +0.0629 [+0.0161, +0.1111]   REAL
  each boundary alone                                           null
  dead-rubber variance ratio  1.160  (dead rubbers MORE variable)

TWO ARMS, both deliberately avoiding an additive covariate -- congestion, absence,
lineups and keeper all fitted at ~2% of a prior SD because an additive term competes
with latents that already explain the variance.

  I_effort   theta = intercept + (1+g*stakes_h)*attack[h] - (1+g*stakes_a)*defence[a] + hfa
             MULTIPLICATIVE: stakes scale how much of its quality a team expresses.
             Dead rubber -> both sides compress toward the intercept, i.e. an even,
             average-scoring game. Smoke fit: g = -0.108 (NEGATIVE -- high stakes
             expressing LESS quality, plausible as cagey tense matches, but one window).

  J_disp     goals ~ NegBinomial(mu=lambda, alpha=alpha0*exp(b*stakes_match))
             Touches the VARIANCE, not theta, so it cannot compete with the latents.
             This is the mechanism the premise actually supports. predict.py's
             scoreline_grid takes a matching alpha -- an overdispersed likelihood with
             a Poisson predictive would measure nothing. Smoke fit: b = +0.475, the
             predicted sign (low stakes -> more dispersed -> flatter probabilities).

PREDICTION ON RECORD: J_disp is the better bet. Calibrating off Elo (premise r=+0.047
-> -0.00021 RPS), a premise of r=+0.040 suggests ~-0.0002, so expect small. But the
variance channel is the one the evidence points at and the one that does not fight the
latents, so if either clears zero it should be that one.

    python work_products/wp028_stakes/run_cv.py [--workers 4]
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

DATA = HERE / "cv_shared_data_stakes.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.5,
            "use_deep_channel": True, "deep_channel_weight": 0.5}
ARMS = {
    "I_effort": {**CHAMPION, "use_stakes_effort": True},
    "J_disp":   {**CHAMPION, "use_stakes_dispersion": True},
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
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    tasks = [(a, w) for w in range(1, n + 1) for a in ARMS]
    print(f"{len(tasks)} tasks = {len(ARMS)} arms x {n} windows, {args.workers} at a time", flush=True)
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
