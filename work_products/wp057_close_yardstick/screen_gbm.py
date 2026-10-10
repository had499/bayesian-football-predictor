"""WP057d — is the GBM WRAPPER real? Screen it against the closing line at ~5x power.

The wrapper is the only live candidate left. On the deployment-accurate horizon-1 grid it
measures −0.00040 [−0.00083,+0.00005] at beta=0.5 — the CI INCLUDES ZERO, so on
outcome-RPS it is unresolved. WP057 established the close is a 4-5x tighter yardstick for
RANKING, and the GBM is NOT market-fed, so the screen is legal here (unlike the market
channel, which is circular against the close).

Reported together: dRPS vs OUTCOME (the unresolved number) and dRPS vs CLOSE, plus the
LEVEL/DISCRIMINATION split that WP057c established as the guard — if a close-metric gain
is carried by LEVEL, the arm is drifting toward the market's average rather than
predicting matches, and the gain should be distrusted.
"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
sys.path.insert(0, str(WP / "wp053_features"))
from analysis import attach_meta, load_checkpoints, paired_diff   # noqa: E402
from market_analysis import join_market                           # noqa: E402
from gbm import walk_forward                                      # noqa: E402

MAXG = 12
CFG = dict(max_depth=2, max_iter=150, learning_rate=0.03, l2_regularization=10.0)
BOOKS = {"pin_open": ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"],
         "pin_close": ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close",
                       "pinnacle_1x2_away_close"]}
SHC = [f"pin_close_shin_{x}" for x in "hda"]


def probs_from_lam(lh, la):
    k = np.arange(MAXG + 1)
    ph = poisson.pmf(k[None, :], lh[:, None]); pa = poisson.pmf(k[None, :], la[:, None])
    g = ph[:, :, None] * pa[:, None, :]
    iu = np.triu_indices(MAXG + 1, 1); il = np.tril_indices(MAXG + 1, -1)
    p = np.stack([g[:, il[0], il[1]].sum(1), np.einsum("ikk->i", g),
                  g[:, iu[0], iu[1]].sum(1)], 1)
    return p / p.sum(1, keepdims=True)


def rps_outcome(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return ((p.cumsum(1) - o.cumsum(1))[:, :2] ** 2).sum(1) / 2


def rps_close(p, c):
    return ((p.cumsum(1) - c.cumsum(1))[:, :2] ** 2).sum(1) / 2


def split(p, c):
    """WP057c's exact LEVEL / DISCRIMINATION decomposition of RPS_close."""
    d = p.cumsum(1)[:, :2] - c.cumsum(1)[:, :2]
    md = d.mean(0)
    return (md ** 2).sum() / 2, ((d - md) ** 2).sum(1).mean() / 2


def main():
    S = pickle.load((WP / "wp056_player_features/features_v2.pkl").open("rb"))
    D, OLD = S["D"], S["old"]
    d = D.reset_index(drop=True)
    wins = np.sort(d.window.unique())
    eval_w = [w for w in wins if w > 200]
    print(f"recomputing the wrapper on eval windows {eval_w[0]}-{eval_w[-1]} "
          f"({len(eval_w)}), {len(OLD)} features", flush=True)
    m_corr = walk_forward(CFG, eval_w, d[OLD].to_numpy(float), d)
    d["m"] = m_corr

    # pair the two sides of each match by the stored keys
    H = d[d.k_side == "h"].set_index(["window", "k_round", "k_home", "k_away"])
    A = d[d.k_side == "a"].set_index(["window", "k_round", "k_home", "k_away"])
    P = H[["lam", "m", "resid"]].join(A[["lam", "m", "resid"]], lsuffix="_h", rsuffix="_a",
                                      how="inner").dropna().reset_index()
    gh = (P.resid_h + P.lam_h).round().astype(int).to_numpy()
    ga = (P.resid_a + P.lam_a).round().astype(int).to_numpy()
    y = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
    print(f"paired {len(P)} matches with a finite correction", flush=True)

    # closing line, joined on the same keys
    ck, _ = load_checkpoints(WP / "wp055_horizon1/checkpoints")
    base = pickle.loads((WP / "wp055_horizon1/cv_shared_data_h1.pkl").read_bytes())["df_cv"]
    j, _, _ = join_market(attach_meta(ck[ck.arm == "H1_CHAMP"].copy(), base),
                          pd.read_pickle(WP / "wp008_market_benchmark/odds_raw.pkl"),
                          books=BOOKS)
    j = j.dropna(subset=SHC)
    pc = j[SHC].to_numpy(); pc = pc / pc.sum(1, keepdims=True)
    cj = j[["window", "round", "home", "away"]].copy()
    cj.columns = ["window", "k_round", "k_home", "k_away"]
    cj[["c_h", "c_d", "c_a"]] = pc
    Q = P.merge(cj, on=["window", "k_round", "k_home", "k_away"], how="inner")
    print(f"of those, {len(Q)} have a Pinnacle closing line\n", flush=True)

    yq = y[P.index.isin(Q.index)] if len(Q) == len(P) else None
    ghq = (Q.resid_h + Q.lam_h).round().astype(int).to_numpy()
    gaq = (Q.resid_a + Q.lam_a).round().astype(int).to_numpy()
    yq = np.where(ghq > gaq, 0, np.where(ghq == gaq, 1, 2))
    C = Q[["c_h", "c_d", "c_a"]].to_numpy()
    g = Q.window.to_numpy()
    p0 = probs_from_lam(Q.lam_h.to_numpy(), Q.lam_a.to_numpy())
    r0o, r0c = rps_outcome(p0, yq), rps_close(p0, C)
    l0, d0 = split(p0, C)
    print(f"  {'beta':>5s} {'dRPS vs OUTCOME':>27s} {'dRPS vs CLOSE':>27s} {'LEVEL':>9s} {'DISCRIM':>9s}")
    print(f"  {0.0:5.2f} {'(baseline)':>27s} {'(baseline)':>27s} {l0:9.6f} {d0:9.6f}")
    for beta in (0.25, 0.5, 1.0):
        lh = Q.lam_h.to_numpy() * np.clip(Q.m_h.to_numpy(), .3, 3.0) ** beta
        la = Q.lam_a.to_numpy() * np.clip(Q.m_a.to_numpy(), .3, 3.0) ** beta
        p = probs_from_lam(lh, la)
        o, ol, oh = paired_diff(pd.Series(rps_outcome(p, yq)), pd.Series(r0o), groups=g)
        c, cl, ch = paired_diff(pd.Series(rps_close(p, C)), pd.Series(r0c), groups=g)
        lv, ds = split(p, C)
        tag = " BETTER" if ch < 0 else ("  worse" if cl > 0 else "   --")
        print(f"  {beta:5.2f} {o:+9.5f}[{ol:+.5f},{oh:+.5f}] {c:+9.5f}[{cl:+.5f},{ch:+.5f}] "
              f"{lv:9.6f} {ds:9.6f}{tag}")
    print("\n  If the CLOSE column excludes zero while OUTCOME does not, the wrapper is real")
    print("  and outcome-RPS simply lacked the power. Check the LEVEL column has not grown:")
    print("  a gain carried by LEVEL is drift toward the market's average, not prediction.")


if __name__ == "__main__":
    main()
