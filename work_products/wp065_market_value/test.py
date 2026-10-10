"""WP065c — does value-weighted availability beat minutes-weighted? And does squad value
itself carry anything? Plus the LEAK AUDIT.

Coefficients chosen on windows <=200, evaluated on 201-343. Nothing tuned on the eval range.
`out_min` is the incumbent (WP059) and the question is whether anything beats IT.

LEAK AUDIT (the reason this file is careful). Transfermarkt historical values might be
START-of-season snapshots (legal) or REVISED end-of-season values (look-ahead: the value
would be partly determined by the results being predicted). The audit splits the gain into
EARLY vs LATE season. A start-of-season snapshot should help EARLY at least as much as late
-- indeed more, since the latents take over as results accumulate. If the gain is
concentrated LATE, the value embeds information from those very results and must be
discarded. Same shape as WP059c's borderline/unambiguous split, which is what rescued the
injury result.
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
    A = pd.read_pickle(HERE / "value_features.pkl")
    ix = A.set_index(["team", "d"])
    for s in ("home", "away"):
        for c in ("out_min", "out_val", "sq_value_m"):
            m[f"{c}_{s}"] = ix[c].reindex(
                pd.MultiIndex.from_arrays([m[s], m.d])).to_numpy()
    m = m[m[["out_min_home", "out_min_away", "sq_value_m_home",
             "sq_value_m_away"]].notna().all(1)].copy()
    # relative squad value, standardised log ratio
    z = np.log(m.sq_value_m_home / m.sq_value_m_away)
    m["relval"] = (z - z.mean()) / z.std()
    # round within season, for the early/late leak audit
    first = m.groupby("season")["round"].transform("min")
    m["sr"] = m["round"] - first + 1
    j, _, _ = join_market(m, pd.read_pickle(WP / "wp008_market_benchmark/odds_raw.pkl"),
                          books=BOOKS)
    j = j.dropna(subset=SHC)
    pc = j[SHC].to_numpy(); pc = pc / pc.sum(1, keepdims=True)
    cj = j[["window", "round", "home", "away"]].copy()
    cj[["c_h", "c_d", "c_a"]] = pc
    Q = m.merge(cj, on=["window", "round", "home", "away"], how="inner")
    for c in ("out_min", "out_val"):
        for s in ("home", "away"):
            Q[f"{c}_{s}"] = Q[f"{c}_{s}"].fillna(0.0)
    return Q


def ev(sub, fh, fa):
    fh = np.asarray(fh, float); fa = np.asarray(fa, float)
    lh0, la0 = sub.lambda_home.to_numpy(), sub.lambda_away.to_numpy()
    y = np.where(sub.goals_home > sub.goals_away, 0,
                 np.where(sub.goals_home == sub.goals_away, 1, 2))
    C = sub[["c_h", "c_d", "c_a"]].to_numpy(); g = sub.window.to_numpy()
    p0 = probs(lh0, la0)
    p = probs(np.clip(lh0 * fh, .05, 8), np.clip(la0 * fa, .05, 8))
    return (paired_diff(pd.Series(rps_c(p, C)), pd.Series(rps_c(p0, C)), groups=g),
            paired_diff(pd.Series(rps_o(p, y)), pd.Series(rps_o(p0, y)), groups=g))


def absn(col, k):
    return (lambda d: (1 - k * d[f"{col}_home"]) * (1 + k * d[f"{col}_away"]),
            lambda d: (1 - k * d[f"{col}_away"]) * (1 + k * d[f"{col}_home"]))


def main():
    Q = load()
    SEL, EV = Q[Q.window <= 200], Q[Q.window > 200]
    print(f"SELECT n={len(SEL)}   HELD-OUT n={len(EV)}\n")
    G = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    GV = (0.0, 0.02, 0.05, 0.10, 0.20)
    ARMS = [
        ("out_min (incumbent)", lambda k: absn("out_min", k), [G]),
        ("out_val (value-wtd)", lambda k: absn("out_val", k), [G]),
        ("out_min + out_val", lambda k, k2: (
            lambda d: (1 - k * d.out_min_home - k2 * d.out_val_home)
                      * (1 + k * d.out_min_away + k2 * d.out_val_away),
            lambda d: (1 - k * d.out_min_away - k2 * d.out_val_away)
                      * (1 + k * d.out_min_home + k2 * d.out_val_home)), [G, G]),
        ("relval (squad value)", lambda k: (
            lambda d: np.exp(k * d.relval / 2), lambda d: np.exp(-k * d.relval / 2)), [GV]),
    ]
    print(f"  {'arm':22s} {'params':>12s} {'HELD-OUT vs CLOSE':>32s} {'vs OUTCOME':>24s}")
    best_params = {}
    for name, fn, grids in ARMS:
        cand = [(x,) for x in grids[0]] if len(grids) == 1 else \
               [(x, y) for x in grids[0] for y in grids[1]]
        bp, bv = None, np.inf
        for p in cand:
            fh, fa = fn(*p)
            (c, _, _), _ = ev(SEL, fh(SEL), fa(SEL))
            if c < bv:
                bv, bp = c, p
        fh, fa = fn(*bp)
        (c, cl, ch), (o, ol, oh) = ev(EV, fh(EV), fa(EV))
        best_params[name] = (fn, bp)
        ps = ", ".join(f"{x:.2f}" for x in bp)
        print(f"  {name:22s} {ps:>12s} {c:+10.6f}[{cl:+.6f},{ch:+.6f}] "
              f"{o:+8.5f}[{ol:+.5f},{oh:+.5f}]")

    print("\n=== LEAK AUDIT: where does the squad-value gain sit, EARLY or LATE season? ===")
    print("  start-of-season snapshot -> helps EARLY at least as much as late (legal)")
    print("  end-of-season revision   -> gain concentrated LATE (look-ahead, discard)\n")
    for name in ("relval (squad value)", "out_val (value-wtd)"):
        fn, bp = best_params[name]
        fh, fa = fn(*bp)
        print(f"  {name}:")
        for lab, mask in (("EARLY (rounds 1-12)", EV.sr <= 12),
                          ("MID   (13-26)", (EV.sr > 12) & (EV.sr <= 26)),
                          ("LATE  (27+)", EV.sr > 26)):
            s = EV[mask.to_numpy()]
            if len(s) < 100:
                print(f"    {lab:22s} n={len(s)} too few"); continue
            (c, cl, ch), _ = ev(s, fh(s), fa(s))
            print(f"    {lab:22s} n={len(s):4d}  {c:+.6f} [{cl:+.6f},{ch:+.6f}]")


if __name__ == "__main__":
    main()
