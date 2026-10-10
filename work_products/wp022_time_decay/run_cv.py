"""WP022 — Dixon-Coles TIME DECAY, the half of that paper never implemented here.

WP002/WP007 tested only the tau correction. Decay is a genuinely different
mechanism from the AR(1) walk: a walk lets team strengths DRIFT, decay changes
how much each observation COUNTS. Dixon & Coles needed it because their strengths
were static; ours are not, so this is an open question rather than a formality.

PREMISE (9,120 team-matches, predicting each team's next match from a
decay-weighted mean of its past xG):

      half-life      MSE      vs no decay
        30 days   1.50356        +2.19%
        90 days   1.46323        -0.55%
       180 days   1.45663        -1.00%
       270 days   1.45639        -1.01%   <- optimum
       545 days   1.45895        -0.84%
       no decay   1.47130         0.00%

PREDICTION ON RECORD: RPS is minimised at the 270-day arm, with 90 worse than 270
and 810 between them. That is a falsifiable SHAPE, not just a sign — if the three
arms come out monotone or flat, the premise did not transfer and decay is doing
something other than what the weighted-mean check measured.

Confidence is low. The premise was measured on a STATIC weighted mean with no walk
at all, so -1.01% MSE is an upper bound on what a model that already drifts can
gain. My honest expectation is null.

PREDICTION UPDATED BEFORE RUNNING (recorded as an update, not a retrofit): measuring
the weights on a real training window (rounds 1-253, n=2,780, spanning 7.3 years) shows
normalisation preserves total likelihood MASS but not INFORMATION --

      half-life   effective N   share of data retained
         90 days      250          9%
        270 days      800         29%
        810 days     1970         71%
       no decay      2780        100%

The 270-day arm therefore discards ~71% of the effective sample. The premise check
faced that same tradeoff and still preferred 270, but it was optimising a static
weighted mean with no latent walk to lose precision in. Given the model already
extracts time structure via the walk, I now expect the ordering to be dominated by
data loss rather than recency: 810 best of the three, and plausibly all three worse
than no decay. The shape prediction above is retained so both can be scored.

All arms are the CHAMPION config (L0 + xG + deep) plus decay, against WP021's
existing champion-minus-decay baseline on the same 115-window grid.

  N_decay_090   half-life  90 days   (premise says WORSE than 270)
  N_decay_270   half-life 270 days   (premise optimum)
  N_decay_810   half-life 810 days   (weak decay, near no-decay)

    python work_products/wp022_time_decay/run_cv.py [--workers 4]
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
from arms import SAMPLER_LEGACY as SAMPLER  # noqa: E402

DATA = HERE / "cv_shared_data_decay.pkl"
CKPT = HERE / "checkpoints"
SCRIPT = REPO / "scripts" / "run_cv_window.py"

CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.5,
            "use_deep_channel": True, "deep_channel_weight": 0.5}
ARMS_WP022 = {
    f"N_decay_{hl:03d}": {**CHAMPION, "use_time_decay": True,
                          "decay_half_life_days": float(hl), "decay_normalise": True}
    for hl in (90, 270, 810)
}


def run_task(arm, w):
    ck = CKPT / arm / f"w{w:03d}.pkl"
    if ck.exists():
        return arm, w, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(w), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps(ARMS_WP022[arm]), "--predict-mode", "plugin",
           "--draws", str(SAMPLER["draws"]), "--tune", str(SAMPLER["tune"]),
           "--chains", str(SAMPLER["chains"]), "--target-accept", str(SAMPLER["target_accept"])]
    env = dict(os.environ)
    env["PYTENSOR_FLAGS"] = "cxx="
    env.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={SAMPLER['chains']} "
                                "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
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
    ap.add_argument("--arms", default=",".join(ARMS_WP022))
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    n = len(pickle.loads(DATA.read_bytes())["windows"])
    tasks = [(a, w) for w in range(1, n + 1) for a in arms]
    print(f"{len(tasks)} tasks = {len(arms)} arms x {n} windows, {args.workers} at a time", flush=True)
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
