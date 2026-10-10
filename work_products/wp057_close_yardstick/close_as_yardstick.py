"""Is the CLOSING LINE a higher-power yardstick than the realised outcome?

We score arms with RPS against the OUTCOME. One match = one coin flip, so the metric
carries irreducible noise that does NOT cancel in a paired comparison (it cancels only
to the extent two arms agree). MDE is +-0.00095 at 115 windows, and ~25 arms came back
"null" against that bar.

The closing line is a far lower-variance estimate of the same truth. Define the SAME
ordinal RPS but against the close's cumulative probabilities instead of the 0/1 outcome:

    RPS_close(model) = mean_i sum_{k<3} ( cumsum(p_model)_k - cumsum(p_close)_k )^2 / 2

If p_close ~= truth, then E[RPS_outcome] = RPS_close + irreducible(p_close), and the
irreducible term does not depend on the model -- so comparing arms on RPS_close estimates
the same quantity with the outcome noise removed.

TEST: on pairs whose verdict is already ESTABLISHED, does RPS_close agree in DIRECTION,
and how much tighter is the CI? The CI-width ratio is the power gain.
"""
import pickle, sys, numpy as np, pandas as pd
from pathlib import Path
WP = Path('work_products')
sys.path.insert(0,'work_products/wp007_structural_fixes'); sys.path.insert(0,'work_products/wp008_market_benchmark')
from analysis import attach_meta, load_checkpoints, paired_diff
from market_analysis import join_market
BOOKS = {"pin_open":["pinnacle_1x2_home","pinnacle_1x2_draw","pinnacle_1x2_away"],
         "pin_close":["pinnacle_1x2_home_close","pinnacle_1x2_draw_close","pinnacle_1x2_away_close"]}
odr = pd.read_pickle(WP/'wp008_market_benchmark/odds_raw.pkl')
SHC=[f"pin_close_shin_{x}" for x in "hda"]
KEY=['window','round','home','away']

def close_probs(m, df):
    j,_,_ = join_market(attach_meta(m.copy(), df), odr, books=BOOKS)
    j = j.dropna(subset=SHC).copy()
    pc = j[SHC].to_numpy(); pc = pc/pc.sum(1,keepdims=True)
    out = j[KEY].copy()
    out[['c_h','c_d','c_a']] = pc
    return out

def rps_vs_close(p, c):
    """same ordinal RPS form, soft target = the close."""
    return ((p.cumsum(1)-c.cumsum(1))[:, :2]**2).sum(1)/2

PAIRS = [
 ("wp033_deep_weight", "D075", "D000", "deep channel ON vs OFF  [ESTABLISHED: ON better, 0/10 LOSO]"),
 ("wp032_xg_weight",   "W075", "W025", "xG weight .75 vs .25   [ESTABLISHED: .75 better]"),
 ("wp032_xg_weight",   "W075", "W100", "xG weight .75 vs 1.00  [known: .75 optimal, small]"),
 ("wp033_deep_weight", "D025", "D100", "deep weight .25 vs 1.0 [known: flat/small]"),
 ("wp039_market_channel","M100","M025","market channel 1.0 vs .25"),
 ("wp030_squad_intactness","M_intact","N_intact_centred","intactness variants [both WORSE than champ]"),
]
shared = {}
def get_shared(d):
    for cand in ('cv_shared_data_market.pkl','cv_shared_data.pkl','cv_shared_data_intact.pkl',
                 'cv_shared_data_crowd.pkl','cv_shared_data_both.pkl'):
        for loc in (WP/d/cand, WP/'wp007_structural_fixes'/cand, WP/'wp030_squad_intactness'/cand,
                    WP/'wp039_market_channel'/cand):
            if loc.exists(): return pickle.loads(loc.read_bytes())['df_cv']
    raise FileNotFoundError(d)

print(f"{'pair':58s} {'dRPS vs OUTCOME':>26s} {'dRPS vs CLOSE':>26s} {'power':>7s}")
for d, a, b, note in PAIRS:
    m,_ = load_checkpoints(WP/d/'checkpoints')
    df = get_shared(d)
    cp = close_probs(m[m.arm==a], df)
    def arm(nm):
        x = m[m.arm==nm][KEY+['p_home','p_draw','p_away','rps']].merge(cp, on=KEY)
        return x.sort_values(KEY).reset_index(drop=True)
    A, B = arm(a), arm(b)
    assert (A[KEY].values == B[KEY].values).all(), "arm rows misaligned"
    C = A[['c_h','c_d','c_a']].to_numpy()
    rcA = rps_vs_close(A[['p_home','p_draw','p_away']].to_numpy(), C)
    rcB = rps_vs_close(B[['p_home','p_draw','p_away']].to_numpy(), C)
    g = A.window.to_numpy()
    o_d, o_lo, o_hi = paired_diff(A.rps, B.rps, groups=g)
    c_d, c_lo, c_hi = paired_diff(pd.Series(rcA), pd.Series(rcB), groups=g)
    gain = ((o_hi-o_lo)/2) / max((c_hi-c_lo)/2, 1e-12)
    agree = "OK" if np.sign(o_d)==np.sign(c_d) else "DISAGREES"
    print(f"{note:58s} {o_d:+9.5f}[{o_lo:+.5f},{o_hi:+.5f}] {c_d:+9.5f}[{c_lo:+.5f},{c_hi:+.5f}] {gain:6.1f}x {agree}")
print(f"\n  n={len(A)} matches. 'power' = outcome-CI half-width / close-CI half-width.")
print("  Negative dRPS = the FIRST arm is better, on both metrics.")
