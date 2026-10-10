"""WP060 — do the two confirmed post-hoc gains COMPOSE?

Both are post-hoc (so neither is subject to the latent-crushing failure mode) and both are
now confirmed on the closing-line metric, but they have never been applied together:

  GBM wrapper   lambda * m^beta, beta=0.5, config locked on windows 70-200 (WP053/WP057d)
  injuries      lambda * (1 - k*out_own) * (1 + k*out_opp), k=0.20 on `out_tot`,
                k locked on windows <=200 (WP059)

Both coefficients were fixed on windows <=200, so windows 201-343 is a clean held-out
range for the combination — no parameter here is chosen on the evaluation data.

THE QUESTION. If the two signals are independent the gains add (~ -0.0001 + -0.00011
= -0.00021 on the close). If they OVERLAP they will not: the GBM's 253 form features may
already partially reflect an absent player through the team's recent output. The size of the
shortfall against the independent sum IS the overlap, and it decides whether the champion
should carry one correction or two.

The GBM correction is saved to disk this time -- WP057d recomputed it and threw it away.
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

MAXG, BETA, K = 12, 0.5, 0.20
CFG = dict(max_depth=2, max_iter=150, learning_rate=0.03, l2_regularization=10.0)
BOOKS = {"pin_open": ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"],
         "pin_close": ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close",
                       "pinnacle_1x2_away_close"]}
SHC = [f"pin_close_shin_{x}" for x in "hda"]
MCACHE = HERE / "gbm_correction.pkl"


def probs(lh, la):
    k = np.arange(MAXG + 1)
    ph = poisson.pmf(k[None, :], lh[:, None]); pa = poisson.pmf(k[None, :], la[:, None])
    g = ph[:, :, None] * pa[:, None, :]
    iu = np.triu_indices(MAXG + 1, 1); il = np.tril_indices(MAXG + 1, -1)
    p = np.stack([g[:, il[0], il[1]].sum(1), np.einsum("ikk->i", g),
                  g[:, iu[0], iu[1]].sum(1)], 1)
    return p / p.sum(1, keepdims=True)


def rps_o(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return ((p.cumsum(1) - o.cumsum(1))[:, :2] ** 2).sum(1) / 2


def rps_c(p, c):
    return ((p.cumsum(1) - c.cumsum(1))[:, :2] ** 2).sum(1) / 2


def lvl(p, c):
    d = p.cumsum(1)[:, :2] - c.cumsum(1)[:, :2]
    return (d.mean(0) ** 2).sum() / 2


def main():
    S = pickle.load((WP / "wp056_player_features/features_v2.pkl").open("rb"))
    d = S["D"].reset_index(drop=True)
    eval_w = [w for w in np.sort(d.window.unique()) if w > 200]
    if MCACHE.exists():
        d["m"] = pd.read_pickle(MCACHE)
        print("GBM correction loaded from cache")
    else:
        print(f"computing GBM wrapper on windows {eval_w[0]}-{eval_w[-1]} "
              f"({len(eval_w)}) -- ~35 min", flush=True)
        d["m"] = walk_forward(CFG, eval_w, d[S["old"]].to_numpy(float), d)
        pd.Series(d["m"]).to_pickle(MCACHE)
        print("saved gbm_correction.pkl", flush=True)

    H = d[d.k_side == "h"].set_index(["window", "k_round", "k_home", "k_away"])
    A = d[d.k_side == "a"].set_index(["window", "k_round", "k_home", "k_away"])
    P = (H[["lam", "m", "resid"]].join(A[["lam", "m", "resid"]], lsuffix="_h", rsuffix="_a",
                                       how="inner").dropna().reset_index())
    P.columns = ["window", "round", "home", "away", "lam_h", "m_h", "res_h",
                 "lam_a", "m_a", "res_a"]

    av = pd.read_pickle(HERE / "availability.pkl")
    av["out_tot"] = av.out_w_new + av.out_w_old
    base = pickle.loads((WP / "wp055_horizon1/cv_shared_data_h1.pkl").read_bytes())["df_cv"]
    ck, _ = load_checkpoints(WP / "wp055_horizon1/checkpoints")
    mm = attach_meta(ck[ck.arm == "H1_CHAMP"].copy(), base)
    mm["d"] = pd.to_datetime(mm.date).dt.normalize()
    ai = av.set_index(["team", "d"]).out_tot
    for s in ("home", "away"):
        mm[f"out_{s}"] = ai.reindex(pd.MultiIndex.from_arrays([mm[s], mm.d])).to_numpy()
    j, _, _ = join_market(mm, pd.read_pickle(WP / "wp008_market_benchmark/odds_raw.pkl"),
                          books=BOOKS)
    j = j.dropna(subset=SHC)
    pc = j[SHC].to_numpy(); pc = pc / pc.sum(1, keepdims=True)
    cj = j[["window", "round", "home", "away"]].copy()
    cj[["c_h", "c_d", "c_a"]] = pc
    Q = (P.merge(mm[["window", "round", "home", "away", "out_home", "out_away"]],
                 on=["window", "round", "home", "away"], how="inner")
          .merge(cj, on=["window", "round", "home", "away"], how="inner"))
    Q = Q[Q.window > 200]
    print(f"\nHELD-OUT windows 201-343: n={len(Q)} matches with GBM correction, "
          f"availability and a Pinnacle close")
    print(f"availability present on both sides: {Q[['out_home','out_away']].notna().all(1).mean():.1%}")
    oh = np.nan_to_num(Q.out_home.to_numpy()); oa = np.nan_to_num(Q.out_away.to_numpy())
    mh = np.clip(Q.m_h.to_numpy(), .3, 3.0); ma = np.clip(Q.m_a.to_numpy(), .3, 3.0)
    lh0, la0 = Q.lam_h.to_numpy(), Q.lam_a.to_numpy()
    gh = (Q.res_h + lh0).round().astype(int).to_numpy()
    ga = (Q.res_a + la0).round().astype(int).to_numpy()
    y = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
    C = Q[["c_h", "c_d", "c_a"]].to_numpy(); g = Q.window.to_numpy()

    ARMS = {
        "baseline":       (np.ones_like(lh0), np.ones_like(la0)),
        "GBM only":       (mh ** BETA, ma ** BETA),
        "injuries only":  ((1 - K * oh) * (1 + K * oa), (1 - K * oa) * (1 + K * oh)),
        "BOTH":           (mh ** BETA * (1 - K * oh) * (1 + K * oa),
                           ma ** BETA * (1 - K * oa) * (1 + K * oh)),
    }
    res = {}
    p0 = probs(lh0, la0); b_o, b_c = rps_o(p0, y), rps_c(p0, C)
    print(f"\n  {'arm':15s} {'RPS':>9s} {'dRPS vs OUTCOME':>27s} {'dRPS vs CLOSE':>29s} {'LEVEL':>9s}")
    for nm, (fh, fa) in ARMS.items():
        p = probs(np.clip(lh0 * fh, .05, 8), np.clip(la0 * fa, .05, 8))
        ro, rc = rps_o(p, y), rps_c(p, C)
        res[nm] = (ro, rc)
        if nm == "baseline":
            print(f"  {nm:15s} {ro.mean():9.5f} {'--':>27s} {'--':>29s} {lvl(p,C):9.6f}")
            continue
        o, ol, ohi = paired_diff(pd.Series(ro), pd.Series(b_o), groups=g)
        c, cl, chi = paired_diff(pd.Series(rc), pd.Series(b_c), groups=g)
        t = " BETTER" if chi < 0 else ("  worse" if cl > 0 else "   --")
        print(f"  {nm:15s} {ro.mean():9.5f} {o:+9.5f}[{ol:+.5f},{ohi:+.5f}] "
              f"{c:+10.6f}[{cl:+.6f},{chi:+.6f}] {lvl(p,C):9.6f}{t}")

    # --- the overlap question ---
    def dd(a, b, i):
        return paired_diff(pd.Series(res[a][i]), pd.Series(res[b][i]), groups=g)
    print("\n=== DO THEY COMPOSE? ===")
    for i, lab in ((1, "CLOSE"), (0, "OUTCOME")):
        g1 = res["GBM only"][i].mean() - res["baseline"][i].mean()
        g2 = res["injuries only"][i].mean() - res["baseline"][i].mean()
        gb = res["BOTH"][i].mean() - res["baseline"][i].mean()
        indep = g1 + g2
        print(f"\n  {lab}: GBM {g1:+.6f} + injuries {g2:+.6f} = {indep:+.6f} if INDEPENDENT")
        print(f"         BOTH actually {gb:+.6f}  -> captured {100*gb/indep:.0f}% of the sum"
              if indep != 0 else "")
        a, al, ah = dd("BOTH", "GBM only", i)
        print(f"         BOTH vs GBM only      {a:+.6f} [{al:+.6f},{ah:+.6f}]"
              f"{'  <- injuries ADD over the GBM' if ah<0 else '  <- no added value'}")
        b, bl, bh = dd("BOTH", "injuries only", i)
        print(f"         BOTH vs injuries only {b:+.6f} [{bl:+.6f},{bh:+.6f}]"
              f"{'  <- GBM ADDS over injuries' if bh<0 else '  <- no added value'}")


if __name__ == "__main__":
    main()
