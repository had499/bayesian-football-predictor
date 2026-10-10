"""WP049 — Championship-derived ENTRY STRENGTH for newly promoted EPL teams.

REPLACES THE EXTERNAL ELO DEPENDENCY. WP025's Elo entry/anchor works, and the gain is
concentrated exactly where our latents are blind: −0.00049 on promoted-team matches
[CI excluding zero] vs −0.00010 on established teams only. The only reason clubelo knows
anything about a newly promoted club is that it watched a Championship season. We can
watch it too -- openfootball has every Championship result back to 2000, free, and
clubelo's API is now DEACTIVATED AT SOURCE (not proxy-blocked: /Fixtures returns
"Fixtures API deactivated"), so the cached Elo CSV is a frozen asset that can never be
refreshed. A Championship-derived version has no such ceiling.

THE MODEL FOR CHAMPIONSHIP GAMES. There is no xG below the top five leagues, so this is
goals-only: a ridge-regularised, opponent-adjusted attack/defence fit with a home-
advantage term,
    goals_ij ~ mu + att_i - def_j + hfa*is_home
solved in closed form per season. Three justifications for goals-only:
  * clubelo has no xG or shots either, so we are not handicapped against what we replace;
  * our form is richer than an Elo -- separate attack and defence, fitted home advantage,
    proper opponent adjustment -- on the same data;
  * the entry COEFFICIENT is fitted, so a noisy input ATTENUATES the effect rather than
    biasing it. A weak signal costs benefit, it cannot inject error.
46-game Championship seasons give ~21% more observations per team than the EPL's 38,
which partly offsets the loss of xG's smoothing (WP007: xG is ~2.2x smoother than goals).

FLAT, NOT RECENCY-WEIGHTED. premise2 tested half-lives of 60 and 30 days and both were
WORSE than season-long (combined corr +0.0277/+0.0268 vs +0.0419), so the
"promoted teams improve late" story is wrong and season-average strength is the signal.

LEAK SAFETY. A club promoted into EPL season Y is rated from Championship season Y-1
ONLY, so nothing after its final Championship match enters its entry value.
"""
import difflib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
SRC = WP / "wp030_squad_intactness" / "cv_shared_data_intact.pkl"
OUT = HERE / "cv_shared_data_chentry.pkl"
RIDGE = 1.0


def ratings(sdf, ridge=RIDGE):
    """Opponent-adjusted attack/defence per team, closed form, season-long weighting."""
    d = sdf.sort_values("date")
    teams = sorted(set(d.home) | set(d.away))
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    X = np.zeros((2 * len(d), 2 * T + 2)); y = np.zeros(2 * len(d))
    for k, (_, r) in enumerate(d.iterrows()):
        for off, (sc, a, b, ih) in enumerate(((r.hg, r.home, r.away, 1.0),
                                              (r.ag, r.away, r.home, 0.0))):
            row = 2 * k + off
            X[row, ti[a]] = 1.0
            X[row, T + ti[b]] = -1.0
            X[row, 2 * T] = 1.0
            X[row, 2 * T + 1] = ih
            y[row] = sc
    P = np.eye(X.shape[1]) * ridge
    P[2 * T:, 2 * T:] = 0.0                      # never penalise intercept / hfa
    beta = np.linalg.solve(X.T @ X + P, X.T @ y)
    R = pd.DataFrame({"ch_team": teams, "att": beta[:T], "dfn": beta[T:2 * T]})
    # standardise WITHIN the Championship cohort: "how good relative to this division".
    # The fitted entry coefficient maps that onto the EPL scale, so no cup cross-links
    # are needed to identify the league-level offset.
    R["z"] = (((R.att - R.att.mean()) / R.att.std()) +
              ((R.dfn - R.dfn.mean()) / R.dfn.std())) / 2.0
    return R


def main():
    shared = pickle.loads(SRC.read_bytes())
    df = shared["df_cv"]
    low = pd.read_pickle(HERE / "lower_divisions.pkl")
    ch = low[low.division == "Championship"]

    dfd = df.assign(dt=pd.to_datetime(df.datetime))
    round_date = dfd.groupby("cum_round").dt.min().sort_index()
    teams = sorted(set(df.team) | set(df.opp_team))
    longname = dict(zip(df.team, df.team_long))

    seasons = sorted(df.season.unique())
    tb = {s: set(df[df.season == s].team) | set(df[df.season == s].opp_team)
          for s in seasons}

    grid = pd.DataFrame(0.0, index=round_date.index, columns=teams)
    found, missing = [], []
    print("entry strength for each promotion (Championship season -> z):")
    for i in range(1, len(seasons)):
        s = seasons[i]
        promoted = tb[s] - tb[seasons[i - 1]]
        if not promoted:
            continue
        prev = f"{int(s)-1}-{int(s) % 100:02d}"
        sdf = ch[ch.season == prev]
        if not len(sdf):
            print(f"  EPL {s}: no Championship data for {prev}"); continue
        R = ratings(sdf)
        avail = list(R.ch_team)
        for code in sorted(promoted):
            ln = longname.get(code, code)
            cand = (difflib.get_close_matches(ln, avail, n=1, cutoff=0.45) or
                    difflib.get_close_matches(ln + " FC", avail, n=1, cutoff=0.4))
            # entry round = this club's first EPL round of that season
            er = dfd[(dfd.season == s) &
                     ((dfd.team == code) | (dfd.opp_team == code))].cum_round.min()
            if not cand:
                missing.append((s, code)); print(f"  EPL {s} {code:4s} {ln:20s} UNMATCHED")
                continue
            z = float(R[R.ch_team == cand[0]].iloc[0].z)
            grid.loc[er, code] = z
            found.append((s, code, cand[0], z))
            print(f"  EPL {s} {code:4s} {ln:20s} <- {cand[0]:26s} z={z:+.2f} (round {er})")

    # CENTRE ACROSS PROMOTIONS. Raw z is positive for every promotion (mean +1.38) --
    # unavoidable, since promoted sides are by construction the BEST Championship teams.
    # Left raw, the coefficient would fit a near-constant promoted-level shift PLUS the
    # variation, and WP007 already measured a global promoted offset as WORSE (+0.00285).
    # The premise correlation (+0.0419) is location-invariant, so the evidence we have is
    # for the VARIATION only -- which promotions are stronger than a typical one. Centring
    # isolates exactly that and leaves the level to the model's own prior, as now.
    zs_raw = np.array([z for *_, z in found])
    zmean = zs_raw.mean()
    for (s_, code, _, z) in found:
        er = dfd[(dfd.season == s_) &
                 ((dfd.team == code) | (dfd.opp_team == code))].cum_round.min()
        grid.loc[er, code] = z - zmean
    print(f"\ncentred entry z by {zmean:+.3f} (promotions are all above the Championship")
    print("  mean by construction; only the SPREAD carries evidence)")
    found = [(a, b, c, z - zmean) for (a, b, c, z) in found]

    print(f"\nmatched {len(found)} promotions, {len(missing)} unmatched")
    assert len(missing) == 0, f"unmatched promotions would silently get z=0: {missing}"
    nz = (grid.to_numpy() != 0).sum()
    print(f"grid {grid.shape}, {nz} non-zero cells (one per promotion)")
    print(f"z range {grid.to_numpy().min():+.2f} .. {grid.to_numpy().max():+.2f}")
    # sanity: z should spread promotions apart, not give them all the same value
    zs = np.array([z for *_, z in found])
    print(f"z across promotions: mean {zs.mean():+.3f}, sd {zs.std():.3f}, "
          f"min {zs.min():+.2f}, max {zs.max():+.2f}")
    assert zs.std() > 0.2, "entry z barely varies — it would carry no information"

    shared["elo_z_grid"] = grid
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
