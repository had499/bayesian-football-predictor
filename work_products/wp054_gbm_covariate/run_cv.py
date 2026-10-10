"""WP054 — the GBM residual correction as an IN-MODEL covariate.

WP053's post-hoc wrapper measured -0.00062 [-0.00122,-0.00005] in the mature regime.
But a wrapper carries NO uncertainty: the Bayesian model says "lambda is 1.50 +- 0.14"
and the GBM says "multiply by 1.03" as a bare point estimate, so the corrected
prediction is over-confident. Entering it as

    theta += beta * z,   z = log(m)   (m = the GBM's multiplicative correction)

gives the correction a posterior, and tests something the wrapper cannot: whether the
model WANTS it. beta = 1 reproduces the wrapper exactly (test-guarded).

WHAT THE COEFFICIENT WILL TELL US. Five additive covariates on theta have failed here --
congestion, absence, lineups, keeper, intactness -- all fitting at ~2% of prior SD
because the latents absorb that direction. This one differs in being CONSTRUCTED to
predict the champion's out-of-sample residual. So:
  * beta near 1   -> the GBM found real structure the latents miss
  * beta near 0   -> the Bayesian machinery REJECTS what the wrapper exploited, and the
                     wrapper's gain came from something the model would not endorse

PREDICTION ON RECORD. I expect beta to fit WELL BELOW 1 -- maybe 0.3-0.6 -- and the RPS
gain to be smaller than the wrapper's -0.00062. Reason: z is built from the champion's
OWN lambda plus features the latents already see, so the latents can partly reproduce it,
and the fitted beta will shrink accordingly. The honest version of this result is likely
to be less impressive than the wrapper, not more.

CAVEAT ON COVERAGE: only 66% of panel matches carry a non-zero z (the earliest windows
have no training base, and pre-2016 matches are never test rows). beta is therefore
identified mainly from later windows.
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

DATA = HERE / "cv_shared_data_gbm.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
# The NEW champion: xG at 0.75 (WP032), deep 0.5, stakes effort.
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.75,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
# Weight points, because WP032 proved the 0.5 hedge was UNDER-set for xG and this
# channel has no reason to share xG's optimum. M050 is the direct analogue of the
# other channels' default; M100/M025 bracket it.
ARMS = {"GBMCOV": {**CHAMPION, "use_gbm_covariate": True}}


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
