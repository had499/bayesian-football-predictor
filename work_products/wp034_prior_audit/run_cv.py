"""WP034 — tune the SHRINKAGE PRIORS the project has never tuned.

WP032 showed the cheapest win available came from re-measuring an ASSUMED constant
(xG weight 0.5 -> 0.75, -0.00029). This continues that audit, but for prior SCALES,
and it starts from a screen rather than a guess.

THE SCREEN (screen.py, 6 fits on window 70, ~15 min). BACKLOG caution 21: a posterior
far from its prior centre does NOT mean the prior binds -- WP031 proved that for
home_mu. So the only test that counts is: widen the prior, does the POSTERIOR move?

    widened                 posterior shift                  1-window dRPS
    home_adv_sd x5          home_sd  0.0198 -> 0.0357 (+80%)     +0.00021
    stakes_effort_sd x3     stakes   -0.244 -> -0.344  (+41%)     +0.00020
    sigma_att/def x6        sigma_att 0.0204 -> 0.0242 (+19%)     +0.00099
    rho -> Beta(5,1)        NO MOVEMENT                           -0.00043
    home_sd x5 (CONTROL)    NO MOVEMENT  <- matches WP031's null, screen is calibrated

`rho`'s Beta(29,1) does NOT bind, which kills my prior hunch that it was pinning
persistence. Screening cost 15 minutes and saved a 2-3 hour sweep.

THE KEY OBSERVATION: all three binding priors got WORSE when LOOSENED. These are
shrinkage priors doing real regularisation work on ~29,000 latents, and the project has
never tuned that strength. If looser hurts, TIGHTER may help -- a direction nobody has
looked. So each parameter is swept BOTH ways around its champion value.

ARMS (baseline = WP033's/WP032's champion, 115 windows already fitted as W075):
    SIG_T   sigma_att/def 0.004  (half)      SIG_L   0.016 (double)
    HSD_T   home_adv_sd   0.01   (half)      HSD_L   0.04  (double)

`home_adv_sd` is the strongest candidate: its posterior sits EXACTLY on its prior scale
(0.0198 vs 0.02) and jumps 80% when freed, so the data carries little information about
per-team home-advantage variation and the prior is simply setting the answer. That is an
unmade choice, not a measurement.

`stakes_effort_sd` is dropped: stakes_effort is worth only -0.00030 in total, so tuning
its prior cannot yield much, and a 4-arm sweep is already 3.2 hours.

PREDICTION ON RECORD. I expect the tighter arms to beat the looser ones (the screen's
direction), with HSD_T the most likely single winner, and the whole spread to be small
-- maybe -0.0003 at best. The honest risk: the screen's one-window dRPS values are all
within noise, so "looser is worse" could be an artefact of a single window, in which
case all four arms come back null. I am NOT dropping any arm on screen evidence here,
because unlike WP032's w1.75 (+0.016, 13x MDE) these differences are ~1x MDE.

    python work_products/wp034_prior_audit/run_cv.py [--workers 6]
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
    "SIG_T": {**CHAMPION, "sigma_att": 0.004, "sigma_def": 0.004},
    "SIG_L": {**CHAMPION, "sigma_att": 0.016, "sigma_def": 0.016},
    "HSD_T": {**CHAMPION, "home_adv_sd": 0.01},
    "HSD_L": {**CHAMPION, "home_adv_sd": 0.04},
    # FOLLOW-UP after the first four arms. home_adv_sd was the ONLY parameter that
    # moved, and LOOSER won (0.04 at -0.00005, 10/10 LOSO) against my prediction that
    # tighter would. -0.00005 is 1/20th of the MDE and below the project's inclusion
    # bar, so the question is whether there is a real GRADIENT here or just noise that
    # happened to line up across folds. The screen showed home_sd's posterior still
    # climbing when the prior was freed (0.0198 -> 0.0357 at scale 0.1), so 0.08
    # discriminates: if the gain grows, it is real and worth banking; if it turns over,
    # 0.04 is the peak, the effect really is ~0.00005, and the audit is finished.
    "HSD_XL": {**CHAMPION, "home_adv_sd": 0.08},
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
