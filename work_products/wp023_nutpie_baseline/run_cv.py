"""WP023 — refit the champion and its reference arm with the nutpie sampler.

WHY THIS EXISTS. nutpie (a Rust NUTS backend for PyMC) is ~1.7x faster than
numpyro on this model, rising to 2.04x on the largest windows where nearly all
the time goes. It targets the IDENTICAL posterior with identical latent
parameters -- same algorithm, faster implementation -- so unlike coarsening the
latent grid it introduces no model change and no bias.

Measured on three windows (45/70/110), full harness, idle machine:

    window   numpyro   nutpie  speedup     dRPS
        45       89s      60s    1.47x  -0.00032
        70      131s      83s    1.59x  +0.00001
       110      242s     118s    2.04x  +0.00007

THE HAZARD THIS RUN EXISTS TO REMOVE. The mean sampler shift across those windows
is -0.00008, which is 10% of the deep-channel effect (0.00083). That shift cancels
only when BOTH arms use the same sampler. Comparing a nutpie arm against the
stored numpyro baseline would manufacture a fake 10% effect. So the baseline must
be refit before nutpie is used for anything.

Extra Monte Carlo noise is not a concern: it inflates the window-level SD of a
paired arm difference from 0.00511 to 0.00513 (+0.4%), moving the MDE 0.00093 ->
0.00094. And nutpie's lower ESS (332 vs 550 on the largest window) is headroom we
do not spend -- every arm runs `--predict-mode plugin`, which uses posterior MEANS,
and ESS 332 gives a mean with MC standard error sd/18.

Two arms, both on WP021's 115-window grid:
  S3_xgchannel_only   goals + xG            the "without deep" reference
  H_deep_added        goals + xG + deep     THE CHAMPION (ModelConfig.champion())

This also verifies the 1.7x claim at full scale rather than on three windows:
analyse.py checks nutpie's RPS against the stored numpyro checkpoints across all
115 windows.

    python work_products/wp023_nutpie_baseline/run_cv.py [--workers 4]
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
sys.path.insert(0, str(HERE.parent / "wp007_structural_fixes"))
sys.path.insert(0, str(HERE.parent))
from notify import done as _notify_done  # noqa: E402
from arms import SAMPLER_LEGACY as SAMPLER  # noqa: E402   full precision -- see feedback-keep-full-precision

DATA = HERE.parent / "wp021_power_confirmation" / "cv_shared_data_big.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"

ARMS = {
    "S3_xgchannel_only": {"use_xg_channel": True, "xg_channel_weight": 0.5},
    "H_deep_added":      {"use_xg_channel": True, "xg_channel_weight": 0.5,
                          "use_deep_channel": True, "deep_channel_weight": 0.5},
}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS[arm]), "--predict-mode", "plugin",
           "--draws", str(SAMPLER["draws"]), "--tune", str(SAMPLER["tune"]),
           "--chains", str(SAMPLER["chains"]), "--target-accept", str(SAMPLER["target_accept"]),
           "--nuts-sampler", "nutpie"]
    env = dict(os.environ)
    env["PYTENSOR_FLAGS"] = "cxx="
    # nutpie does not use XLA devices the way numpyro's pmap does; the thread-pinning
    # flags were measured to make no difference even for numpyro's parallel mode.
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
