"""WP025 — Elo-scaled entry level for promoted teams.

ONE new arm against WP023's existing nutpie champion baseline (reusable as-is: the
sampler standard is nutpie 600/600, which is exactly what WP023 was fitted with).

  F_elo_anchor   champion + use_elo_anchor  (Elo anchors EVERY team, EVERY round)

WHAT IT DOES. The AR(1) walk currently reverts toward 0 -- "the average team". This
gives it an Elo-implied reversion target instead:

    x_t = mu_t + rho*(x_{t-1} - mu_t) + sigma*z_t,    mu_t = beta * elo_z[t, team]

At the fitted rho ~ 0.993 the pull is 0.7% per round, ~21% of the gap over a season:
an anchor, not an override. Match results still drive short-term movement, while Elo
says where a team's strength belongs.

WHY THIS AND NOT THE PROMOTED-ONLY VARIANT. An entry-only shift fires once per team
per promotion, so it touches ~30% of matches and only in a side's first season. The
anchor acts on every round for every team, and it also carries information arriving
MID-season -- clubelo rates European and cup matches, which this pipeline never sees.
It subsumes the entry case: a promoted team simply starts at its Elo-implied level.

THE DEFECT IT ALSO FIXES. `entry_shift` was a single global scalar, fitted at -0.06 in
WP007 against a real promotion gap of -0.37. Pre-debut Elo across 33 promotions runs
1475 (Huddersfield 2017) to 1726 (Burnley 2023) -- a spread one scalar cannot express.
The champion has NO promotion effect at all (WP007's global version measured WORSE at
+0.00285), so the anchor is the first informed treatment of team entry.

WHY THIS SHAPE, not a channel or a covariate: a channel needs E[obs] = f(lambda) and
Elo is a rating, not a rate. An additive covariate on theta competes with
attack - defence, and Elo correlates 0.936 with the model's own implied strength --
the documented failure mode of the original rolling-mean xG covariate (fitted at ~zero
weight). So Elo enters as a PRIOR on the latent: it scales where a team's walk starts.

EVIDENCE GOING IN. A 150-draw smoke fit learned elo_anchor_att = +0.151,
elo_anchor_def = +0.052 -- both the expected sign (higher Elo pulls the latent up).

PREDICTION ON RECORD: genuinely uncertain, wider than usual. Elo correlates 0.936 with
the model's own implied strength, so most of what it knows the latents already have --
that redundancy is what killed every shot-feed decomposition in WP019. Against that,
an output-level blend of model and Elo gave -0.00018, and the same blend for xG gave
-0.00017 while the real xG CHANNEL delivered -0.00258, a 15x amplification. If that
factor carries, this lands near -0.0027; if the redundancy dominates, it lands at zero.
The grid's MDE is +-0.00095, so unlike most arms here this one CAN resolve either way.

    python work_products/wp025_elo_channel/run_cv.py [--workers 4]
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
from arms import SAMPLER, sampler_args  # noqa: E402  nutpie 600/600, the new standard

DATA = HERE / "cv_shared_data_elo.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"

CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.5,
            "use_deep_channel": True, "deep_channel_weight": 0.5}
# Only the GENERAL form is run. The promoted-teams-only variant (use_elo_entry)
# acts once per entry and so touches ~30% of matches; the anchor acts on every round
# for every team and subsumes it, including mid-season information the EPL-only
# pipeline cannot see (a deep European run lifts a side's Elo). 24 partial fits of
# the entry-only arm are parked under checkpoints/_partial_E_elo_entry_24of115.
ARMS = {"F_elo_anchor": {**CHAMPION, "use_elo_anchor": True}}


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
    print(f"{len(tasks)} tasks = {len(ARMS)} arm x {n} windows, {args.workers} at a time", flush=True)
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
