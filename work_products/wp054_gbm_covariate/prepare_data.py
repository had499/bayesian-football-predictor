"""WP054 — the GBM correction as an IN-MODEL covariate, not a post-hoc wrapper.

WP053 measured a two-stage GBM correction at -0.00062 [-0.00122,-0.00005] (mature
regime). But that wrapper has no uncertainty: the Bayesian model says "lambda is
1.50 +- 0.14" and the GBM says "multiply by 1.03" as a bare point estimate. Feeding it in
as a covariate with a FITTED coefficient fixes that, and tests something the wrapper
cannot: whether the model WANTS the correction.

    theta += beta * z,   z = log(m),   m = the GBM's multiplicative correction

At beta = 1 this reproduces the wrapper exactly. If beta fits near 1 the GBM is finding
real structure; if it collapses to ~2% of prior SD like the five failed additive
covariates (congestion, absence, lineups, keeper, intactness), the wrapper's gain was
exploiting something the Bayesian machinery rejects.

LEAK SAFETY. The composite for every match is computed from STRICTLY EARLIER CV windows.
Test sets are disjoint (STEP == TEST_LEN == 3), so each match belongs to exactly one
window's test set; giving it the value computed there means it is leak-free both as a
test row AND later, when that same match serves as a training row for a later window.
Matches before window 26 get z = 0 (no training base), which makes the covariate
contribute nothing rather than guess.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import attach_meta, load_checkpoints  # noqa: E402

SRC = WP / "wp039_market_channel" / "cv_shared_data_market.pkl"
OUT = HERE / "cv_shared_data_gbm.pkl"
CFG = dict(max_depth=2, max_iter=150, learning_rate=0.03, l2_regularization=10.0)
MIN_TRAIN = 1500


def main():
    S = pickle.load((WP / "wp053_features" / "features.pkl").open("rb"))
    D, FE = S["D"], S["feats"]
    D = D.reset_index(drop=True)
    X = D[FE].to_numpy(float)
    lam = D.lam.to_numpy()
    goals = D.resid.to_numpy() + lam
    y = np.clip(goals / lam, 0, 20)          # offset trick: Poisson on the ratio
    w = lam                                   # ... with lambda as the weight

    m_pred = np.full(len(D), np.nan)
    for win in np.sort(D.window.unique()):
        tr = (D.window < win).to_numpy()
        te = (D.window == win).to_numpy()
        if tr.sum() < MIN_TRAIN or te.sum() == 0:
            continue
        Xtr, Xte = X[tr], X[te]
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        mdl = HistGradientBoostingRegressor(loss="poisson", random_state=0,
                                            early_stopping=False, **CFG)
        mdl.fit(Xtr, y[tr], sample_weight=w[tr])
        m_pred[te] = mdl.predict(Xte)
    got = np.isfinite(m_pred)
    print(f"composite computed for {got.sum()}/{len(D)} rows "
          f"({D.window[got].nunique()} of {D.window.nunique()} windows)")
    # z on the LOG-RATE scale, which is where theta lives
    z = np.where(got, np.log(np.clip(m_pred, 0.3, 3.0)), 0.0)
    print(f"z: mean {z[got].mean():+.4f}, sd {z[got].std():.4f}, "
          f"range {z[got].min():+.3f}..{z[got].max():+.3f}")

    # --- map back to match rows: one z for the home side, one for the away side ---
    d = pickle.loads(SRC.read_bytes())
    df = d["df_cv"].copy()
    mm, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    mm = attach_meta(mm[mm.arm == "W075"], df)
    mm["d"] = pd.to_datetime(mm.date).dt.normalize()
    # D was built in the same row order as mm, two rows per match (home then away)
    assert len(D) == 2 * len(mm), f"row-order assumption broken: {len(D)} vs {2*len(mm)}"
    H = D.is_home.to_numpy() == 1
    assert H[0::2].all() and (~H[1::2]).all(), "rows are not strictly home,away interleaved"
    key = pd.DataFrame(dict(d=mm.d.to_numpy(), home=mm.home.to_numpy(),
                            away=mm.away.to_numpy(),
                            z_home=z[0::2], z_away=z[1::2]))
    df["d"] = pd.to_datetime(df.datetime).dt.normalize()
    hk = np.where(df.is_home == 1, df.team, df.opp_team)
    ak = np.where(df.is_home == 1, df.opp_team, df.team)
    idx = pd.MultiIndex.from_arrays([df.d, hk, ak])
    ki = key.set_index(["d", "home", "away"])
    assert not ki.index.duplicated().any(), "(date, home, away) not unique"
    got2 = ki.reindex(idx)
    df["gbm_z_home"] = np.nan_to_num(got2.z_home.to_numpy(), nan=0.0)
    df["gbm_z_away"] = np.nan_to_num(got2.z_away.to_numpy(), nan=0.0)
    nz = (df[df.is_home == 1].gbm_z_home != 0).mean()
    print(f"attached to the panel: {nz:.1%} of matches carry a non-zero z")
    d["df_cv"] = df.drop(columns="d")
    OUT.write_bytes(pickle.dumps(d))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
