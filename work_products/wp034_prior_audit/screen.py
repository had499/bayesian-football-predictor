"""WP034 screen — WHICH priors actually bind? One window, six fits, ~15 minutes.

BACKLOG caution 21 (learned the hard way in WP031): a posterior sitting far from its
prior centre does NOT mean the prior is distorting the fit. `home_mu` sits 2.8 prior SDs
from its centre and widening it moved RPS by -0.00005 -- the likelihood simply dominated.

So this screens by the only test that counts: WIDEN the prior and see whether the
POSTERIOR MOVES. Only parameters that move earn a 2-3 hour 115-window sweep.

Champion posteriors vs their priors (115 windows), i.e. the candidates:
    sigma_att  0.0196  vs HalfNormal(0.008)   -- 2.45x the prior SCALE, top candidate
    sigma_def  0.0199  vs HalfNormal(0.008)   -- 2.49x
    rho_att    0.9893  vs Beta(29,1) mean .967 -- wants MORE persistence
    rho_def    0.9842  vs Beta(29,1)
    home_sd    0.0203  vs HalfNormal(0.02)    -- sitting exactly ON the scale
    xg_shape   3.2088  vs Gamma(2,0.5) mean 4  -- comfortable, not a candidate

`home_mu` is the CONTROL: known from WP031 to be non-binding despite looking extreme.
If the screen flags it as binding, the screen itself is wrong.
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

DATA = HERE.parent / "wp030_squad_intactness" / "cv_shared_data_intact.pkl"
CKPT = HERE / "screen"
SCRIPT = REPO / "scripts" / "run_cv_window.py"
WINDOW = 70
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.75,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}
# Each arm widens ONE prior by ~5-6x so a binding prior has room to show it.
TESTS = {
    "base":     {},
    "sigma6x":  {"sigma_att": 0.05, "sigma_def": 0.05},
    "rho_wide": {"rho_att_alpha": 5.0, "rho_def_alpha": 5.0},   # Beta(5,1) mean 0.833
    "homesd5x": {"home_adv_sd": 0.10},
    "stakes3x": {"stakes_effort_sd": 1.0},
    "homemu5x": {"home_sd": 0.15},                              # the CONTROL (WP031)
    # --- SCREEN 2: the constants still unaudited after WP034's first four arms ---
    # These are NOT prior scales but STRUCTURAL constants, so "does the posterior
    # move" is the wrong test for some of them -- for those the RPS column is the
    # signal. Flagged in the output.
    "ssmult_1":  {"season_start_sigma_mult": 1.0},   # no season-start inflation at all
    "ssmult_6":  {"season_start_sigma_mult": 6.0},   # double the current 3.0
    "sswin_3":   {"season_start_window": 3},         # 3 rounds instead of 5
    "sswin_8":   {"season_start_window": 8},
    "init_05":   {"init_scale": 0.05},               # t=0 team-strength spread
    "init_50":   {"init_scale": 0.50},
    "xgshape_t": {"xg_channel_shape_alpha": 8.0, "xg_channel_shape_beta": 2.0},
}
# Arms whose point is the RPS column, not a posterior shift (structural, not a prior).
STRUCTURAL = {"ssmult_1", "ssmult_6", "sswin_3", "sswin_8", "init_05", "init_50"}
WATCH = ["sigma_att", "sigma_def", "rho_att", "rho_def", "home_sd", "home_mu",
         "xg_shape", "xg_kappa", "stakes_effort"]


def run(name):
    ck = CKPT / name / f"w{WINDOW:03d}.pkl"
    if ck.exists():
        return name, "cached", 0.0
    ck.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT), "--data-path", str(DATA), "--checkpoint-path", str(ck),
           "--window-index", str(WINDOW), "--use-xg", "false", "--use-dc", "false",
           "--config-json", json.dumps({**CHAMPION, **TESTS[name]}),
           "--predict-mode", "plugin", *sampler_args()]
    env = dict(os.environ); env["PYTENSOR_FLAGS"] = "cxx="
    env.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={SAMPLER['chains']}")
    t0 = time.time()
    try:
        with open(ck.parent / f"w{WINDOW:03d}.log", "w") as lf:
            subprocess.run(cmd, check=True, timeout=45 * 60, env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
        return name, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        ck.unlink(missing_ok=True)
        return name, f"FAILED {type(e).__name__}", time.time() - t0


def load(name):
    r = pickle.loads((CKPT / name / f"w{WINDOW:03d}.pkl").read_bytes())["results"]
    r = r[0] if isinstance(r, list) else r
    return r["param_means"], r["rps"]


def main():
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fu in as_completed([ex.submit(run, n) for n in TESTS]):
            n, st, s = fu.result()
            print(f"  {n:10s} {st} ({s:.0f}s)", flush=True)

    bp, brps = load("base")
    print(f"\nwindow {WINDOW}, base RPS {brps:.5f}\n")
    print(f"  {'arm':10s} {'widened':24s} " + " ".join(f"{w[:9]:>9s}" for w in WATCH) + "    dRPS")
    print(f"  {'base':10s} {'-':24s} " + " ".join(f"{bp.get(w, float('nan')):9.4f}" for w in WATCH)
          + "        -")
    verdicts = {}
    for name in TESTS:
        if name == "base":
            continue
        p, rps = load(name)
        cells, moved = [], []
        for w in WATCH:
            a, b = bp.get(w), p.get(w)
            if a is None or b is None:
                cells.append(f"{'-':>9s}"); continue
            cells.append(f"{b:9.4f}")
            if a != 0 and abs(b - a) / abs(a) > 0.10:    # >10% shift = it moved
                moved.append(f"{w} {a:.4f}->{b:.4f}")
        verdicts[name] = moved
        print(f"  {name:10s} {str(TESTS[name])[:24]:24s} " + " ".join(cells)
              + f"   {rps - brps:+.5f}")

    print("\n=== WHICH PRIORS BIND? (>10% posterior shift on any watched scalar) ===")
    for name, moved in verdicts.items():
        tag = "BINDS" if moved else "no — prior is not constraining the fit"
        print(f"  {name:10s} {tag}")
        for m in moved:
            print(f"               {m}")
    print("\n  homemu5x is the CONTROL: WP031 proved it is non-binding. If it shows")
    print("  as BINDS here, distrust this screen rather than the WP031 result.")
    print("\n=== STRUCTURAL arms — judge these on dRPS, not on posterior movement ===")
    print("    (season_start_sigma_mult / window and init_scale are not prior scales;")
    print("     changing them changes the MODEL, so a still posterior proves nothing.)")
    for name in TESTS:
        if name in STRUCTURAL:
            _, rps = load(name)
            print(f"  {name:10s} {str(TESTS[name])[:34]:34s} dRPS {rps - brps:+.5f}")
    print("\n  one window only -- this RANKS candidates for a sweep, it does not decide")
    print("  anything. WP032's partial read and WP033's smoke both mis-ranked neighbours.")


if __name__ == "__main__":
    main()
