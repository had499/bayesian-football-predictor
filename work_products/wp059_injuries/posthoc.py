"""WP059b — the FREE decisive test: a post-hoc availability multiplier, nothing fitted.

Four arms have died by handing the Bayesian model a covariate the latents then crushed
(additive terms land at ~2% of prior SD; WP054's in-model GBM covariate fitted at 5% and
was null while the SAME signal as a post-hoc wrapper gave -0.00062). The failure mode is
that ~29,000 latents explain the training data without help, so the coefficient shrinks to
nothing and there is nothing left to use at prediction time.

A multiplier with a HAND-SET coefficient has no coefficient to crush -- nothing is
estimated. So it is both the cheapest test (no MCMC, just reprocessing checkpoints) and the
one immune to the documented failure mode. If no k helps, the direction dies for free.

    lambda_home' = lambda_home * (1 - k*out_home) * (1 + k*out_away)
    lambda_away' = lambda_away * (1 - k*out_away) * (1 + k*out_home)

i.e. a team's own absences depress its scoring AND lift its opponent's.

FALSIFICATION BUILT IN. `out_w_old` (absent >14 days) is run through the identical grid. The
mechanism claim is that only UNPRICED change can help, because the latents have already
tracked a long-term absence. If out_w_old scores like out_w_new, that claim is false and
nothing here should be believed.

Scored on outcome-RPS and on the closing line (5x power, legal -- not market-fed), with the
LEVEL/DISCRIM guard.
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
from analysis import attach_meta, load_checkpoints, paired_diff   # noqa: E402
from market_analysis import join_market                           # noqa: E402

MAXG = 12
BOOKS = {"pin_open": ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"],
         "pin_close": ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close",
                       "pinnacle_1x2_away_close"]}
SHC = [f"pin_close_shin_{x}" for x in "hda"]


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


def lvl_dis(p, c):
    d = p.cumsum(1)[:, :2] - c.cumsum(1)[:, :2]
    md = d.mean(0)
    return (md ** 2).sum() / 2, ((d - md) ** 2).sum(1).mean() / 2


def main():
    base = pickle.loads((WP / "wp055_horizon1/cv_shared_data_h1.pkl").read_bytes())["df_cv"]
    ck, _ = load_checkpoints(WP / "wp055_horizon1/checkpoints")
    m = attach_meta(ck[ck.arm == "H1_CHAMP"].copy(), base)
    m["d"] = pd.to_datetime(m.date).dt.normalize()
    A = pd.read_pickle(HERE / "availability.pkl")
    av = A.set_index(["team", "d"])

    for side in ("home", "away"):
        for c in ("out_w_new", "out_w_old", "ret_w_new", "out_gk_new"):
            m[f"{c}_{side}"] = av[c].reindex(
                pd.MultiIndex.from_arrays([m[side], m.d])).to_numpy()
    cov = m[["out_w_new_home", "out_w_new_away"]].notna().all(1)
    print(f"matches: {len(m)}, with availability on BOTH sides: {cov.sum()} ({cov.mean():.1%})")
    m = m[cov].copy()

    j, _, _ = join_market(m, pd.read_pickle(WP / "wp008_market_benchmark/odds_raw.pkl"),
                          books=BOOKS)
    j = j.dropna(subset=SHC)
    pc = j[SHC].to_numpy(); pc = pc / pc.sum(1, keepdims=True)
    cj = j[["window", "round", "home", "away"]].copy()
    cj[["c_h", "c_d", "c_a"]] = pc
    Q = m.merge(cj, on=["window", "round", "home", "away"], how="inner")
    print(f"of those, {len(Q)} have a Pinnacle close\n")

    y = np.where(Q.goals_home > Q.goals_away, 0, np.where(Q.goals_home == Q.goals_away, 1, 2))
    C = Q[["c_h", "c_d", "c_a"]].to_numpy()
    g = Q.window.to_numpy()
    lh0, la0 = Q.lambda_home.to_numpy(), Q.lambda_away.to_numpy()
    p0 = probs(lh0, la0)
    b_o, b_c = rps_o(p0, y), rps_c(p0, C)
    print(f"baseline RPS {b_o.mean():.5f}   RPS_close {b_c.mean():.6f}")

    # premise check, free: does newly-unavailable share move the residual at all?
    print("\n=== PREMISE CHECK: correlation with the champion's residual (goals - lambda) ===")
    print("  (premise checks in this project have over-promised 3/3 -- upper bound only)")
    for c in ("out_w_new", "out_w_old", "ret_w_new"):
        rh = Q.goals_home.to_numpy() - lh0
        ra = Q.goals_away.to_numpy() - la0
        x = np.concatenate([Q[f"{c}_home"].to_numpy(), Q[f"{c}_away"].to_numpy()])
        r = np.concatenate([rh, ra])
        ok = np.isfinite(x) & np.isfinite(r)
        cc = np.corrcoef(x[ok], r[ok])[0, 1]
        print(f"    {c:12s} corr(own residual) {cc:+.4f}   "
              f"expect NEGATIVE for out_*, POSITIVE for ret_*")

    print("\n=== POST-HOC MULTIPLIER GRID (nothing fitted) ===")
    print(f"  {'signal':12s} {'k':>5s} {'dRPS vs OUTCOME':>27s} {'dRPS vs CLOSE':>27s} {'LEVEL':>9s}")
    for sig in ("out_w_new", "out_w_old", "ret_w_new"):
        oh = np.nan_to_num(Q[f"{sig}_home"].to_numpy())
        oa = np.nan_to_num(Q[f"{sig}_away"].to_numpy())
        if sig.startswith("ret"):
            oh, oa = -oh, -oa              # a return is the opposite sign to an absence
        for k in (0.25, 0.5, 1.0, 2.0):
            lh = lh0 * (1 - k * oh) * (1 + k * oa)
            la = la0 * (1 - k * oa) * (1 + k * oh)
            lh, la = np.clip(lh, .05, 8), np.clip(la, .05, 8)
            p = probs(lh, la)
            do, ol, ohi = paired_diff(pd.Series(rps_o(p, y)), pd.Series(b_o), groups=g)
            dc, cl, chi = paired_diff(pd.Series(rps_c(p, C)), pd.Series(b_c), groups=g)
            lv, _ = lvl_dis(p, C)
            tag = " BETTER" if chi < 0 else ("  worse" if cl > 0 else "   --")
            print(f"  {sig:12s} {k:5.2f} {do:+9.5f}[{ol:+.5f},{ohi:+.5f}] "
                  f"{dc:+9.6f}[{cl:+.6f},{chi:+.6f}] {lv:9.6f}{tag}")
        print()
    print("  out_w_old is the FALSIFICATION arm: if it scores like out_w_new, the")
    print("  'only unpriced change can help' mechanism is wrong and this is all noise.")


if __name__ == "__main__":
    main()
