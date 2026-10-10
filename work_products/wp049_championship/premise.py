"""WP049 premise — does Championship form predict how promoted teams do in the EPL?

If yes, we can replace the external Elo dependency with something we derive ourselves.
WP025's Elo anchor is worth -0.00049 on promoted-team matches (CI excluding zero) vs
-0.00010 on established teams, and the only reason Elo knows anything about a newly
promoted club is that it watched them play a Championship season. We now have those
results (openfootball, 12 seasons, complete).

THE TEST. For each promoted club, take its Championship season strength and ask whether
that predicts the champion's out-of-sample ERROR on its EPL matches. The champion
currently starts every promoted side at a prior meaning "average team", while WP007
measured real promoted sides ~0.37 below average in attack — so if Championship form
separates the good promotions from the bad ones, it is information the model lacks.

NAME MATCHING IS HAND-VERIFIED. openfootball uses "Leeds United FC", our panel uses
Understat short names. There are only ~30 promoted club-seasons, so every mapping is
printed for inspection rather than trusted to difflib — which earlier today mapped three
different clubs onto "Krasnodar" and silently dropped a quarter of the RFPL data.
"""
import difflib
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


def champ_table(df):
    """per team-season Championship strength, regular season only (playoffs excluded:
    they are a 5-match lottery and only some teams play them)."""
    ch = df[df.division == "Championship"].copy()
    rows = []
    for (s, t), g in pd.concat([
        ch.assign(team=ch.home, gf=ch.hg, ga=ch.ag),
        ch.assign(team=ch.away, gf=ch.ag, ga=ch.hg)]).groupby(["season", "team"]):
        n = len(g)
        pts = ((g.gf > g.ga) * 3 + (g.gf == g.ga)).sum()
        rows.append(dict(season=s, ch_team=t, n=n, ppg=pts / n,
                         gdpg=(g.gf - g.ga).mean(), gfpg=g.gf.mean(), gapg=g.ga.mean()))
    return pd.DataFrame(rows)


def main():
    low = pd.read_pickle(HERE / "lower_divisions.pkl")
    tab = champ_table(low)
    print(f"Championship table: {len(tab)} club-seasons, "
          f"{tab.season.nunique()} seasons")

    epl = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                       .read_bytes())["df_cv"]
    seasons = sorted(epl.season.unique())
    tbs = {s: set(epl[epl.season == s].team) | set(epl[epl.season == s].opp_team)
           for s in seasons}
    promo = {s: (tbs[s] - tbs[seasons[i - 1]]) if i else set()
             for i, s in enumerate(seasons)}
    longname = dict(zip(epl.team, epl.team_long))

    # map Understat -> openfootball for promoted clubs only, and PRINT every mapping
    ch_names = sorted(tab.ch_team.unique())
    print("\nname mapping for promoted clubs (verify by eye):")
    rows = []
    for s in seasons:
        if not promo[s]:
            continue
        prev = f"{int(s)-1}-{int(s) % 100:02d}"
        avail = sorted(tab[tab.season == prev].ch_team.unique())
        if not avail:
            continue
        for code in sorted(promo[s]):
            ln = longname.get(code, code)
            cand = difflib.get_close_matches(ln, avail, n=1, cutoff=0.45)
            if not cand:
                cand = difflib.get_close_matches(ln + " FC", avail, n=1, cutoff=0.4)
            hit = cand[0] if cand else None
            flag = "" if hit else "   <-- UNMATCHED"
            print(f"  EPL {s}  {code:4s} {ln:22s} -> {str(hit):28s}{flag}")
            if hit:
                r = tab[(tab.season == prev) & (tab.ch_team == hit)].iloc[0]
                rows.append(dict(season=s, code=code, ch_team=hit, ppg=r.ppg,
                                 gdpg=r.gdpg, gfpg=r.gfpg, gapg=r.gapg))
    P = pd.DataFrame(rows)
    print(f"\nmatched {len(P)} promoted club-seasons")

    # the champion's out-of-sample error on those clubs' matches
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = attach_meta(m[m.arm == "W075"], epl)
    recs = []
    for _, r in m.iterrows():
        for side, team, gl, lam in (("h", r.home, r.goals_home, r.lambda_home),
                                    ("a", r.away, r.goals_away, r.lambda_away)):
            recs.append(dict(window=r.window, season=r.season, team=team,
                             resid=gl - lam))
    L = pd.DataFrame(recs).merge(P, left_on=["season", "team"],
                                 right_on=["season", "code"], how="inner")
    print(f"promoted-team rows in the test set: {len(L)}")

    print("\n=== does Championship form predict the model's error on promoted clubs? ===")
    print("    resid > 0 = the club scored MORE than the champion expected, i.e. the")
    print("    model UNDER-rated them. A positive correlation with Championship")
    print("    strength means strong promotions are under-rated and weak ones over-rated")
    print("    -- exactly the information a generic 'average team' prior cannot carry.\n")
    g = L.window.to_numpy()
    for col, lab in (("ppg", "points per game"), ("gdpg", "goal diff per game"),
                     ("gfpg", "goals for per game"), ("gapg", "goals against p.g.")):
        c, lo, hi = boot_corr(L[col].to_numpy(), L.resid.to_numpy(), g)
        print(f"  {lab:22s} corr {c:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"\n  [for scale: intactness premise +0.1469, Elo +0.047, stakes +0.040]")

    print("\n=== by Championship strength tercile ===")
    L["q"] = pd.qcut(L.ppg, 3, labels=["weak promotion", "mid", "strong promotion"])
    for q, k in L.groupby("q", observed=True):
        rng = np.random.default_rng(0)
        gg = k.window.to_numpy(); uq = np.unique(gg)
        idx = {x: np.where(gg == x)[0] for x in uq}
        v = k.resid.to_numpy()
        bs = np.array([v[np.concatenate([idx[x] for x in rng.choice(uq, len(uq), True)])].mean()
                       for _ in range(3000)])
        print(f"  {str(q):18s} n={len(k):4d}  ppg {k.ppg.mean():.2f}  "
              f"mean resid {v.mean():+.4f} "
              f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")


if __name__ == "__main__":
    main()
