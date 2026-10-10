"""WP039b screen — which fixed shape fixes the geometry?

WP039's market channel is real (-0.00079, 10/10 LOSO) but `max rhat` reached 1.412
because the learned `mkt_shape` chose 58-120 -- believing the market's implied rate to
+-9%, which makes an extremely stiff likelihood.

SELECTING ON RHAT FROM ONE WINDOW IS LEGITIMATE, unlike selecting on RPS. rhat is a
property of the posterior GEOMETRY, which is set by the model and the training data, not
by the 74-match test set. Single-window RPS magnitude is unreliable (WP035; and WP039's
own smoke overstated the effect 12x) so the RPS column below is printed for shape only
and must NOT pick the winner.
"""
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
from arms import SAMPLER, sampler_args  # noqa: E402

DATA = HERE / "cv_shared_data_market.pkl"
CKPT = HERE / "shape_screen"
WINDOW = int(os.environ.get("SCREEN_WINDOW", "70"))
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.75,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
BASE = {**CHAMPION, "use_market_channel": True, "market_channel_weight": 1.0}
TESTS = {
    "learned":  {},                                   # WP039 as shipped: rhat 1.412
    "tightprior": {"market_channel_shape_alpha": 8.0, "market_channel_shape_beta": 2.0},
    "fix3":  {"market_channel_shape_fixed": 3.0},     # CV 58%, as loose as xG
    "fix6":  {"market_channel_shape_fixed": 6.0},     # CV 41%
    "fix12": {"market_channel_shape_fixed": 12.0},    # CV 29%
    "fix25": {"market_channel_shape_fixed": 25.0},    # CV 20%
}


def run(tag):
    ck = CKPT / tag / f"w{WINDOW:03d}.pkl"
    if ck.exists():
        return tag, "cached"
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO / "scripts" / "run_cv_window.py"),
           "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(WINDOW), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps({**BASE, **TESTS[tag]}),
           "--predict-mode", "plugin", *sampler_args()]
    env = dict(os.environ); env["PYTENSOR_FLAGS"] = "cxx="
    env.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={SAMPLER['chains']}")
    with open(ck.parent / f"w{WINDOW:03d}.log", "w") as lf:
        subprocess.run(cmd, check=True, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return tag, "ok"


def main():
    with ThreadPoolExecutor(max_workers=3) as ex:
        for fu in as_completed([ex.submit(run, t) for t in TESTS]):
            print("  ", *fu.result(), flush=True)
    champ = pickle.loads((HERE.parent / "wp032_xg_weight/checkpoints/W075" /
                          f"w{WINDOW:03d}.pkl").read_bytes())["results"]
    champ = champ[0] if isinstance(champ, list) else champ
    print(f"\nchampion window {WINDOW}: rps {champ['rps']:.5f}, max rhat {champ['max_rhat']:.3f}\n")
    print(f"  {'arm':12s} {'CV':>6s} {'max rhat':>9s} {'div':>5s} {'mkt_shape':>10s} "
          f"{'mkt_kappa':>10s} {'rps (shape only)':>18s}")
    for tag in TESTS:
        r = pickle.loads((CKPT / tag / f"w{WINDOW:03d}.pkl").read_bytes())["results"]
        r = r[0] if isinstance(r, list) else r
        pm_ = r["param_means"]
        sh = pm_.get("mkt_shape")
        fx = TESTS[tag].get("market_channel_shape_fixed")
        eff = sh if sh is not None else fx
        cv = f"{100/ (eff ** 0.5):.0f}%" if eff else "—"
        print(f"  {tag:12s} {cv:>6s} {r['max_rhat']:9.3f} {r['n_diverging']:5d} "
              f"{(sh if sh is not None else fx):10.2f} "
              f"{pm_.get('mkt_kappa', float('nan')):10.3f} {r['rps']:18.5f}")
    print("\n  DECIDE ON RHAT. The champion sits at ~1.02-1.10; anything above ~1.1 is")
    print("  not reliably sampled and should not ship whatever its RPS says.")


if __name__ == "__main__":
    main()
