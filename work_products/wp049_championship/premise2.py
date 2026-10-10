"""WP049 premise 2 — an OPPONENT-ADJUSTED, RECENCY-WEIGHTED Championship rating.

premise1 tested Championship points per game and goal difference per game: both null
(ppg -0.028, gdpg +0.037, CIs spanning zero). But those are the wrong measures for two
reasons the user identified:

  1. NOT OPPONENT-ADJUSTED. 2.0 ppg against a strong Championship is not 2.0 against a
     weak one. A fitted model separates team strength from schedule.
  2. NOT TIME-AWARE. Season-average ppg blurs a side that IMPROVED through the campaign,
     which promoted teams often do (January signings, momentum). What the EPL model
     needs is strength at the END of the Championship season, not its average.

This computes a closed-form stand-in for the real thing: a ridge-regularised
least-squares attack/defence rating per team (a Massey-style model on log goal rates),
optionally with exponential RECENCY weighting so late-season form dominates. Seconds
rather than MCMC, and it tests whether the signal exists before 10 Bayesian fits.

LEAK SAFETY: ratings for EPL season Y use ONLY Championship matches from season Y-1, so
nothing after a promoted club's final Championship round enters its entry value.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import attach_meta, load_checkpoints  # noqa: E402


def boot_corr(x, y, g, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(g); idx = {k: np.where(g == k)[0] for k in uq}
    out = np.empty(n)
    for i in range(n):
        sel = np.concatenate([idx[k] for k in rng.choice(uq, len(uq), replace=True)])
        out[i] = np.corrcoef(x[sel], y[sel])[0, 1]
    return np.corrcoef(x, y)[0, 1], *np.percentile(out, [2.5, 97.5])


def ratings(season_df, half_life=None, ridge=1.0):
    """Ridge least-squares attack/defence on goals, opponent-adjusted.
       goals_ij ~ mu + att_i - def_j (+ hfa for the home side).
       half_life in DAYS gives exponential recency weighting; None = flat."""
    d = season_df.sort_values("date")
    teams = sorted(set(d.home) | set(d.away))
    ti = {t: i for i, t in enumerate(teams)}
    n, T = len(d) * 2, len(teams)
    X = np.zeros((n, 2 * T + 2)); y = np.zeros(n); w = np.ones(n)
    if half_life:
        last = d.date.max()
        age = (last - d.date).dt.days.to_numpy()
        decay = 0.5 ** (age / half_life)
    else:
        decay = np.ones(len(d))
    for k, (_, r) in enumerate(d.iterrows()):
        for off, (sc, a, b, is_home) in enumerate((
                (r.hg, r.home, r.away, 1.0), (r.ag, r.away, r.home, 0.0))):
            row = 2 * k + off
            X[row, ti[a]] = 1.0            # attack of the scoring side
            X[row, T + ti[b]] = -1.0       # defence of the conceding side
            X[row, 2 * T] = 1.0            # intercept
            X[row, 2 * T + 1] = is_home    # home advantage
            y[row] = sc
            w[row] = decay[k]
    sw = np.sqrt(w)[:, None]
    A = X * sw; bvec = y * np.sqrt(w)
    P = np.eye(X.shape[1]) * ridge; P[2 * T:, 2 * T:] = 0.0   # don't penalise mu/hfa
    beta = np.linalg.solve(A.T @ A + P, A.T @ bvec)
    return pd.DataFrame({"ch_team": teams, "att": beta[:T], "dfn": beta[T:2 * T]})


def main():
    low = pd.read_pickle(HERE / "lower_divisions.pkl")
    ch = low[low.division == "Championship"].copy()
    epl = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                       .read_bytes())["df_cv"]
    seasons = sorted(epl.season.unique())
    tbs = {s: set(epl[epl.season == s].team) | set(epl[epl.season == s].opp_team)
           for s in seasons}
    promo = {s: (tbs[s] - tbs[seasons[i - 1]]) if i else set()
             for i, s in enumerate(seasons)}
    longname = dict(zip(epl.team, epl.team_long))
    import difflib

    variants = {"flat (season-long)": None, "recency hl=60d": 60, "recency hl=30d": 30}
    built = {k: [] for k in variants}
    for s in seasons:
        if not promo[s]:
            continue
        prev = f"{int(s)-1}-{int(s) % 100:02d}"
        sdf = ch[ch.season == prev]
        if not len(sdf):
            continue
        for lab, hl in variants.items():
            R = ratings(sdf, half_life=hl)
            R["z_att"] = (R.att - R.att.mean()) / R.att.std()
            R["z_dfn"] = (R.dfn - R.dfn.mean()) / R.dfn.std()
            avail = list(R.ch_team)
            for code in sorted(promo[s]):
                ln = longname.get(code, code)
                c = difflib.get_close_matches(ln, avail, n=1, cutoff=0.45) or \
                    difflib.get_close_matches(ln + " FC", avail, n=1, cutoff=0.4)
                if not c:
                    continue
                row = R[R.ch_team == c[0]].iloc[0]
                built[lab].append(dict(season=s, code=code, z_att=row.z_att,
                                       z_dfn=row.z_dfn, z_tot=row.z_att + row.z_dfn))

    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = attach_meta(m[m.arm == "W075"], epl)
    recs = []
    for _, r in m.iterrows():
        recs.append(dict(window=r.window, season=r.season, team=r.home,
                         resid=r.goals_home - r.lambda_home,
                         resid_def=r.goals_away - r.lambda_away))
        recs.append(dict(window=r.window, season=r.season, team=r.away,
                         resid=r.goals_away - r.lambda_away,
                         resid_def=r.goals_home - r.lambda_home))
    base = pd.DataFrame(recs)

    print("=== does an OPPONENT-ADJUSTED Championship rating predict promoted-team error? ===")
    print("    resid > 0 = the club outscored what the champion expected (model under-rated it)")
    print("    premise1's season-average ppg: -0.028 [-0.083,+0.026]  (null)\n")
    print(f"  {'rating variant':22s} {'measure':10s} {'n':>5s} {'corr':>9s} {'CI':>22s}")
    for lab, rows in built.items():
        P = pd.DataFrame(rows)
        if not len(P):
            continue
        L = base.merge(P, left_on=["season", "team"], right_on=["season", "code"],
                       how="inner")
        g = L.window.to_numpy()
        for col, mlab in (("z_att", "attack"), ("z_dfn", "defence"), ("z_tot", "combined")):
            tgt = L.resid.to_numpy() if col != "z_dfn" else -L.resid_def.to_numpy()
            c, lo, hi = boot_corr(L[col].to_numpy(), tgt, g)
            flag = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
            print(f"  {lab:22s} {mlab:10s} {len(L):5d} {c:+9.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]{flag}")
    print("\n  for scale: intactness premise +0.1469, Elo +0.047, stakes +0.040")
    print("  if recency-weighted ratings beat flat ones, END-OF-SEASON form is the signal")
    print("  and season-average ppg was simply the wrong measure.")


if __name__ == "__main__":
    main()
