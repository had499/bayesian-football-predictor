"""WP029 — congestion in the MULTIPLICATIVE effort term, not as an additive covariate.

THE HYPOTHESIS BEING TESTED IS ARCHITECTURAL. Four additive covariates all fitted at
~2% of a prior SD and died: congestion (WP027), absence (WP012), lineups (WP011),
keeper. WP028 then showed non-additive forms get USED -- the stakes effort term fitted
at 91% of its prior SD and the dispersion term at 60%. So the question is whether those
nulls were about the VARIABLES or about the FORM.

"A tired team plays below itself" is also a better mechanism than "a tired team scores
0.05 fewer goals", so congestion is the natural first retest.

  K_rest_eff   effort = 1 + g_stakes*stakes + g_rest*rest   (rest z-scored)
  L_load_eff   effort = 1 + g_stakes*stakes + g_load*load

Contributions sum INSIDE one multiplier so each stays identifiable;
corr(stakes, rest) = +0.049, so they are not fighting each other. Tested as separate
arms because corr(rest, load) = -0.689 and WP011's crux test looked significant purely
because correlated features split into cancelling coefficients.

BASELINE is WP028's `I_effort`, which IS the current champion (L0 + xG + deep + stakes
effort). The extra rest/load columns are inert when the flag is off, so those
checkpoints are a valid paired baseline.

EVIDENCE GOING IN -- discouraging, and recorded before the run:
  * WP027's additive arms: rest -0.00000 [-0.00004,+0.00003], load -0.00002
    [-0.00006,+0.00001] -- the TIGHTEST intervals in the project.
  * the residual-correlation premise was null and the WRONG SIGN for fatigue
    (all-competitions rest diff r = -0.0202).
  * a 150-draw smoke fit of THIS arm put rest_effort at 0.0088 (3% of its prior SD)
    and load_effort at -0.0009 (0.3%), while stakes_effort in the same multiplier
    fitted at -0.0995. So the multiplicative form does not appear to rescue it.

PREDICTION ON RECORD: null. I expect this to show that WP027's null was about the
VARIABLE, not the form -- which is worth establishing, because it bounds how far
WP028's architectural lesson generalises.

    python work_products/wp029_congestion_effort/run_cv.py [--workers 4]
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

DATA = HERE / "cv_shared_data_both.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.5,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True}
ARMS = {
    "K_rest_eff": {**CHAMPION, "use_congestion_effort": True, "congestion_effort_mode": "rest"},
    "L_load_eff": {**CHAMPION, "use_congestion_effort": True, "congestion_effort_mode": "load"},
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
