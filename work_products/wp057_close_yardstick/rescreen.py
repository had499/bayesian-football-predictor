"""WP057b -- re-screen every past sweep against the CLOSING line (4-5x power).

WITHIN-SWEEP CONTRASTS ONLY. Sweeps ran at different times against different
champions-of-the-day, so an arm is compared only against another arm from the SAME
checkpoint directory. No cross-sweep, no cross-grid comparison.

EXCLUDED: wp039/wp051 (market channel) -- grading a market-fed arm against the market is
circular, and WP057 measured the inflation (close-gain/outcome-gain 0.77 vs xG's 0.32).

Reported: paired dRPS vs OUTCOME (the old metric) beside paired dRPS vs CLOSE (Shin
de-vigged Pinnacle), both cluster-bootstrapped by window. Negative = the TEST arm is
better than the baseline. The close metric is a COMPASS: direction and significance only,
magnitudes do not transfer to outcome-RPS.

MULTIPLICITY: ~40 contrasts at 95% means ~2 expected false positives. A survivor needs
the outcome-RPS point estimate to agree in sign before it means anything.
"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
from analysis import attach_meta, load_checkpoints, paired_diff   # noqa: E402
from market_analysis import join_market                           # noqa: E402

BOOKS = {"pin_open": ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"],
         "pin_close": ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close",
                       "pinnacle_1x2_away_close"]}
SHC = [f"pin_close_shin_{x}" for x in "hda"]
KEY = ["window", "round", "home", "away"]
PCOL = ["p_home", "p_draw", "p_away"]
ODR = pd.read_pickle(WP / "wp008_market_benchmark" / "odds_raw.pkl")
BIG = pickle.loads((WP / "wp021_power_confirmation/cv_shared_data_big.pkl").read_bytes())["df_cv"]
SML = pickle.loads((WP / "wp007_structural_fixes/cv_shared_data.pkl").read_bytes())["df_cv"]

# (directory, baseline arm, [test arms], human label)
SCREENS = [
 # ---- CONSTANTS: values we picked, re-examined at 5x power ----
 ("wp033_deep_weight", "D025", ["D000", "D075", "D100"], "deep channel weight (champion=0.50, absent here)"),
 ("wp032_xg_weight",   "W075", ["W025", "W100"],         "xG channel weight (champion=0.75=baseline)"),
 ("wp046_final_constants", "SSW_3", ["SSW_8"],           "season_start_window (champion=3=baseline)"),
 ("wp034_prior_audit", "HSD_T", ["HSD_L", "SIG_T", "SIG_L"], "prior widths: HFA sd / sigma, tight vs loose"),
 ("wp035_season_boundary", "SSM_1", ["SSM_2", "INIT_50", "INIT_100"], "season-boundary constants"),
 # ---- THE HEADLINE NULL: deep channel confirmation at 115 windows ----
 ("wp021_power_confirmation", "S3_xgchannel_only", ["H_deep_added", "K_deep_shots"],
  "DEEP CONFIRMATION (WP021 verdict: NOT confirmed)"),
 # ---- the original ladder: six arms declared null against L0 ----
 ("wp007_structural_fixes", "L0_baseline",
  ["L1_posterior", "L2_level", "L3_randomwalk", "L4_promoted", "L5_xgchannel",
   "L6_dixoncoles", "S1_xgweight1", "S2_dc_only_fixed"], "WP007 original ladder vs L0"),
 # ---- monotone families and paired arms ----
 ("wp017_outcome_likelihood", "J_outcome_05", ["J_outcome_10", "J_outcome_30"], "outcome-level likelihood weight"),
 ("wp022_time_decay", "N_decay_810", ["N_decay_270", "N_decay_090"], "time decay (verdict: WORSE)"),
 ("wp015_time_varying_level", "G_both", ["E_hfa_only", "F_level_only"], "time-varying HFA / league level"),
 ("wp019_shot_channel", "F_shots_only", ["E_shots_added", "G_shots_heavy"], "shot-count channel"),
 ("wp020_deep_channel", "I_deep_only", ["H_deep_added", "J_deep_heavy"], "deep channel (WP020 grid)"),
 ("wp014_npxg_decomposition", "B_npxg", ["C_kappa_season", "D_shotxg"], "npxG / per-season kappa / shot-xG"),
 ("wp016_split_defence", "H_split", ["H_split_loose"], "split defence (verdict: WORSE)"),
 ("wp027_congestion", "G_rest", ["H_load"], "congestion rest vs load"),
 ("wp029_congestion_effort", "K_rest_eff", ["L_load_eff"], "congestion x effort"),
 ("wp028_stakes", "I_effort", ["J_disp"], "stakes effort vs dispersion"),
 ("wp030_squad_intactness", "M_intact", ["N_intact_centred"], "squad intactness (verdict: WORSE)"),
 ("wp031_hfa_step", "A_crowd", ["B_hfa_wide"], "crowd regime / wide HFA prior"),
]

def rps_vs_close(p, c):
    return ((p.cumsum(1) - c.cumsum(1))[:, :2] ** 2).sum(1) / 2

def run(d, base, tests, label):
    m, _ = load_checkpoints(WP / d / "checkpoints")
    n0 = (m.arm == base).sum()
    df = BIG if n0 > 2000 else SML
    j, _, _ = join_market(attach_meta(m[m.arm == base].copy(), df), ODR, books=BOOKS)
    j = j.dropna(subset=SHC)
    pc = j[SHC].to_numpy(); pc = pc / pc.sum(1, keepdims=True)
    cp = j[KEY].copy(); cp[["c_h", "c_d", "c_a"]] = pc
    def arm(nm):
        x = m[m.arm == nm][KEY + PCOL + ["rps"]].merge(cp, on=KEY)
        return x.sort_values(KEY).reset_index(drop=True)
    A = arm(base)
    if len(A) == 0:
        print(f"  {label}: no market join, skipped"); return
    C = A[["c_h", "c_d", "c_a"]].to_numpy()
    rcA = rps_vs_close(A[PCOL].to_numpy(), C)
    print(f"\n  {label}   [baseline {base}, n={len(A)}]")
    for t in tests:
        B = arm(t)
        if len(B) != len(A) or not (B[KEY].values == A[KEY].values).all():
            print(f"    {t:20s} row mismatch ({len(B)} vs {len(A)}), skipped"); continue
        rcB = rps_vs_close(B[PCOL].to_numpy(), C)
        g = A.window.to_numpy()
        o, olo, ohi = paired_diff(B.rps, A.rps, groups=g)
        c, clo, chi = paired_diff(pd.Series(rcB), pd.Series(rcA), groups=g)
        sig = "BETTER" if chi < 0 else ("worse " if clo > 0 else "  --  ")
        agree = "" if np.sign(o) == np.sign(c) else "  [signs differ]"
        new = ""
        if (chi < 0 or clo > 0) and (olo < 0 < ohi):
            new = "  <== NEW: close resolves what outcome could not"
        print(f"    {t:20s} out {o:+9.5f}[{olo:+.5f},{ohi:+.5f}]  "
              f"close {c:+9.5f}[{clo:+.5f},{chi:+.5f}] {sig}{new}{agree}")

print("=" * 118)
print("WP057b  RE-SCREEN AGAINST THE CLOSING LINE   (negative = TEST arm better than baseline)")
print("=" * 118)
for d, b, ts, lab in SCREENS:
    try: run(d, b, ts, lab)
    except Exception as e: print(f"\n  {lab}: FAILED {type(e).__name__}: {e}")
print("\n" + "=" * 118)
print("'BETTER'/'worse' = the CLOSE metric's CI excludes zero. Compass only: magnitudes do")
print("not transfer to outcome-RPS. ~40 contrasts at 95% => ~2 expected false positives,")
print("so a survivor also needs the outcome point estimate to agree in sign.")
