"""WP062b — do position / goalkeeper / severity beat the crude total?

Every arm's coefficients are chosen on windows <=200 and evaluated on 201-343. Nothing is
tuned on the evaluation range. `out_tot` is the incumbent (WP059) and the pre-registered
question is whether any refinement beats IT -- not whether each beats the champion.

MULTIPLICITY, stated up front: five arms are reported. Picking whichever scores best on the
EVAL range would itself be selection, so the decision rule is set in advance -- a refinement
is adopted only if it beats `out_tot` on the eval range by more than `out_tot` beats the
champion, i.e. the refinement must add at least as much as the original signal did.
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
COLS = ["out_tot", "out_att", "out_mid", "out_def", "out_long", "out_gk"]


def probs(lh, la):
    k = np.arange(MAXG + 1)
    ph = poisson.pmf(k[None, :], lh[:, None]); pa = poisson.pmf(k[None, :], la[:, None])
    g = ph[:, :, None] * pa[:, None, :]
    iu = np.triu_indices(MAXG + 1, 1); il = np.tril_indices(MAXG + 1, -1)
    p = np.stack([g[:, il[0], il[1]].sum(1), np.einsum("ikk->i", g),
                  g[:, iu[0], iu[1]].sum(1)], 1)
    return p / p.sum(1, keepdims=True)


def rps_c(p, c):
    return ((p.cumsum(1) - c.cumsum(1))[:, :2] ** 2).sum(1) / 2


def rps_o(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return ((p.cumsum(1) - o.cumsum(1))[:, :2] ** 2).sum(1) / 2


def load():
    base = pickle.loads((WP / "wp055_horizon1/cv_shared_data_h1.pkl").read_bytes())["df_cv"]
    ck, _ = load_checkpoints(WP / "wp055_horizon1/checkpoints")
    m = attach_meta(ck[ck.arm == "H1_CHAMP"].copy(), base)
    m["d"] = pd.to_datetime(m.date).dt.normalize()
    A = pd.read_pickle(HERE / "availability_rich.pkl").set_index(["team", "d"])
    for s in ("home", "away"):
        for c in COLS:
            m[f"{c}_{s}"] = A[c].reindex(
                pd.MultiIndex.from_arrays([m[s], m.d])).to_numpy()
    m = m[m[[f"out_tot_{s}" for s in ("home", "away")]].notna().all(1)].copy()
    j, _, _ = join_market(m, pd.read_pickle(WP / "wp008_market_benchmark/odds_raw.pkl"),
                          books=BOOKS)
    j = j.dropna(subset=SHC)
    pc = j[SHC].to_numpy(); pc = pc / pc.sum(1, keepdims=True)
    cj = j[["window", "round", "home", "away"]].copy()
    cj[["c_h", "c_d", "c_a"]] = pc
    Q = m.merge(cj, on=["window", "round", "home", "away"], how="inner")
    for c in COLS:
        for s in ("home", "away"):
            Q[f"{c}_{s}"] = Q[f"{c}_{s}"].fillna(0.0)
    return Q


def ev(sub, fh, fa):
    # the arm builders operate on DataFrame columns, so they hand back Series --
    # coerce before any numpy broadcasting
    fh = np.asarray(fh, dtype=float)
    fa = np.asarray(fa, dtype=float)
    lh0, la0 = sub.lambda_home.to_numpy(), sub.lambda_away.to_numpy()
    y = np.where(sub.goals_home > sub.goals_away, 0,
                 np.where(sub.goals_home == sub.goals_away, 1, 2))
    C = sub[["c_h", "c_d", "c_a"]].to_numpy(); g = sub.window.to_numpy()
    p0 = probs(lh0, la0)
    p = probs(np.clip(lh0 * fh, .05, 8), np.clip(la0 * fa, .05, 8))
    return (paired_diff(pd.Series(rps_c(p, C)), pd.Series(rps_c(p0, C)), groups=g),
            paired_diff(pd.Series(rps_o(p, y)), pd.Series(rps_o(p0, y)), groups=g))


def main():
    Q = load()
    SEL, EV = Q[Q.window <= 200], Q[Q.window > 200]
    print(f"SELECT n={len(SEL)}   HELD-OUT n={len(EV)}\n")
    G1 = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    G2 = (0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60)

    def arm_tot(k):
        return (lambda d: (1 - k * d.out_tot_home) * (1 + k * d.out_tot_away),
                lambda d: (1 - k * d.out_tot_away) * (1 + k * d.out_tot_home))

    def arm_pos(ka, kd):
        return (lambda d: (1 - ka * d.out_att_home - ka * 0.5 * d.out_mid_home)
                          * (1 + kd * d.out_def_away + kd * 0.5 * d.out_mid_away),
                lambda d: (1 - ka * d.out_att_away - ka * 0.5 * d.out_mid_away)
                          * (1 + kd * d.out_def_home + kd * 0.5 * d.out_mid_home))

    def arm_gk(k, kg):
        return (lambda d: (1 - k * d.out_tot_home) * (1 + k * d.out_tot_away)
                          * (1 + kg * d.out_gk_away),
                lambda d: (1 - k * d.out_tot_away) * (1 + k * d.out_tot_home)
                          * (1 + kg * d.out_gk_home))

    def arm_sev(k, ks):
        return (lambda d: (1 - k * (d.out_tot_home + ks * d.out_long_home))
                          * (1 + k * (d.out_tot_away + ks * d.out_long_away)),
                lambda d: (1 - k * (d.out_tot_away + ks * d.out_long_away))
                          * (1 + k * (d.out_tot_home + ks * d.out_long_home)))

    def arm_long(k):
        return (lambda d: (1 - k * d.out_long_home) * (1 + k * d.out_long_away),
                lambda d: (1 - k * d.out_long_away) * (1 + k * d.out_long_home))

    ARMS = [("out_tot (incumbent)", arm_tot, [G1]),
            ("POSITION (att/def)", arm_pos, [G1, G1]),
            ("out_tot + GOALKEEPER", arm_gk, [G1, G2]),
            ("out_tot + SEVERITY", arm_sev, [G1, G2]),
            ("out_long ONLY", arm_long, [G1])]
    print(f"  {'arm':24s} {'params':>14s} {'HELD-OUT vs CLOSE':>32s} {'vs OUTCOME':>24s}")
    out = {}
    for name, fn, grids in ARMS:
        best, bv = None, np.inf
        if len(grids) == 1:
            cand = [(k,) for k in grids[0]]
        else:
            cand = [(x, y) for x in grids[0] for y in grids[1]]
        for p in cand:
            fh, fa = fn(*p)
            (c, _, _), _ = ev(SEL, fh(SEL), fa(SEL))
            if c < bv:
                bv, best = c, p
        fh, fa = fn(*best)
        (c, cl, ch), (o, ol, oh) = ev(EV, fh(EV), fa(EV))
        out[name] = c
        ps = ", ".join(f"{x:.2f}" for x in best)
        print(f"  {name:24s} {ps:>14s} {c:+10.6f}[{cl:+.6f},{ch:+.6f}] "
              f"{o:+8.5f}[{ol:+.5f},{oh:+.5f}]")
    base = out["out_tot (incumbent)"]
    print(f"\n  DECISION RULE: adopt a refinement only if it beats out_tot ({base:+.6f}) by")
    print(f"  at least what out_tot itself is worth. Threshold: {2*base:+.6f} or better.")
    for k, v in out.items():
        if k == "out_tot (incumbent)":
            continue
        verdict = "ADOPT" if v <= 2 * base else "no"
        print(f"    {k:24s} {v:+.6f}  -> {verdict}")


if __name__ == "__main__":
    main()
