"""WP027 — fixture congestion as a fitted covariate.

WP009 tested rest days and found nothing, but with a LEAGUE-ONLY measure: a Thursday
European trip followed by a Sunday league game looked identically rested to a clear
week. Rebuilding rest from the complete fixture calendar (league + FA Cup + EFL Cup +
CL/EL/UECL) changes the rest value for **27.9% of matches** and raises the count of
home sides on <=3 days' rest from 981 to 1,444. So the measure genuinely was broken.

Cup and European fixtures are absent from Understat entirely, so this is information
the latent walk could never have absorbed -- unlike most covariates tried here.

TWO ARMS, SEPARATELY. rest and load correlate at -0.689, and WP011's crux test looked
significant purely because two features at r=0.76 split into cancelling coefficients.
BACKLOG caution 1: test correlated features alone.

  G_rest   champion + beta * standardised rest days (all competitions)
  H_load   champion + beta * standardised fixtures in the prior 14 days

EVIDENCE GOING IN -- discouraging on both counts:
  * the residual-correlation premise is null: all-competitions rest diff gives
    r = -0.0202 [-0.0468, +0.0092], and the point estimate is the WRONG SIGN for
    fatigue (more home rest associating with worse-than-predicted home performance).
  * a 150-draw smoke fit put beta_rest at 0.003 against a prior SD of 0.1 -- 3% of
    one prior SD, the same "declined to use it" signature as the original xG
    covariate (beta_xG fitted at ~zero because the latents already explained it).

Run anyway because a residual correlation is a weaker instrument than a fitted arm --
the model could extract signal the raw correlation misses. That distinction has been
right before: WP011's correlational null did not settle WP012's covariate test.

PREDICTION ON RECORD: null, and this time I expect the arm to agree with the premise.

    python work_products/wp027_congestion/run_cv.py [--workers 4]
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

DATA = HERE / "cv_shared_data_congestion.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.5,
            "use_deep_channel": True, "deep_channel_weight": 0.5}
ARMS = {
    "G_rest": {**CHAMPION, "use_congestion": True, "congestion_mode": "rest"},
    "H_load": {**CHAMPION, "use_congestion": True, "congestion_mode": "load"},
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
