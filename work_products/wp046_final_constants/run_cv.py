"""WP046 — the last two unaudited constants.

`clip_theta` = 5.0 is PROVABLY INERT and needs no sweep: theta = log(lambda) spans
-0.969 to +1.372 across all 7,560 predicted rates, so the soft clip at 5.0 never
activates (0.000000 of rates even reach 4.0). Audited for free.

`season_start_window` = 5 is the one genuinely untested knob left. It was a DEAD config
field until WP035 wired it (declared on ModelConfig, never read, so the earlier screen
measured exactly +0.00000 for every value). It sets how many rounds at each season start
get the inflated innovation variance that `season_start_sigma_mult` applies -- and WP035
found the MULTIPLIER is flat anywhere in 2.0-3.0, so the WINDOW is the remaining degree
of freedom in that mechanism.

PREDICTION ON RECORD: null. WP035 showed the multiplier is flat, which suggests the model
is simply insensitive to how season-boundary uncertainty is parameterised. Running it to
CLOSE the audit rather than from expectation -- the category is 1-for-9 and the one win
(xG weight) came from a constant nobody had measured at all.
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
    "SSW_3": {**CHAMPION, "season_start_window": 3},
    "SSW_8": {**CHAMPION, "season_start_window": 8},
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
