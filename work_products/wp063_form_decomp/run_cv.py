"""WP063 — the TWO-TIMESCALE latent: slow "ability" + fast "form".

`use_form_decomposition` is fully implemented in model.py, defaults to False, and NO arm in
any work product has ever enabled it. Same category as `match_effect_prior`: written, wired,
never tested. I had written the structural space off as exhausted on the strength of WP057's
re-screen -- but that re-screened everything that had been RUN, and this was never run.

WHY IT IS THE STRONGEST REMAINING LEAD. WP049 established that Elo's advantage over this
model was in-season TRACKING SPEED, not entry information. Tracking speed is exactly what
the innovation process governs. A single Gaussian AR(1) must trade off adapting to real
shifts against being whipped around by noise; splitting into a persistent ability walk
(rho~0.98, sigma 0.003) and a mean-reverting form walk (rho~0.89, sigma 0.015) lets it do
both. WP031's finding that home advantage moved as a STEP is further evidence this system
has discrete shifts a single smooth walk smears out.

Scales are roughly comparable to the champion by construction: stationary SD ~0.036 here
against the champion's ~0.033, with the slow ability drift added on top. All four sigmas are
PRIORS (via ar1_hyperpriors), so the data can move them.

RISKS, both checked by smoke test before the full sweep: four AR(1) scans instead of two
(~2x latents, ~58,000), and ability-vs-form is identified ONLY by the difference in their
persistence priors, so rhat could be poor.
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

DATA = HERE.parent / "wp055_horizon1" / "cv_shared_data_h1.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
# The NEW champion: xG at 0.75 (WP032), deep 0.5, stakes effort.
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.75,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
# Weight points, because WP032 proved the 0.5 hedge was UNDER-set for xG and this
# channel has no reason to share xG's optimum. M050 is the direct analogue of the
# other channels' default; M100/M025 bracket it.
ARMS = {
    # the two-timescale latent: slow ability + fast form
    "FORM": {**CHAMPION, "use_form_decomposition": True},
    # per-team innovation SD, partially pooled. Implemented for WP006, which predates the
    # work_products structure -- so it was last judged against a FAR older champion (no xG
    # channel, no deep, no stakes effort) on a smaller grid, and it defaults to False. It is
    # also exactly the "promoted-team innovation variance" idea WP049 flagged as next: a
    # promoted side or a team mid-managerial-change plausibly needs a bigger sigma than a
    # stable top-six squad, which one shared value cannot represent.
    "PTSIG": {**CHAMPION, "use_per_team_sigma": True},
}


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
