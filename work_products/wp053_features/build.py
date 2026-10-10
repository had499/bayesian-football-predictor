"""WP053 — one comprehensive, leak-safe feature set for the residual-composite work.

WP052b showed the champion's residual IS predictable from our own features (+0.0023 OOS
R^2, p=0.000 against a 40-shuffle null) once DEFENSIVE records were included -- my first
attempt omitted them and wrongly concluded the feature space was exhausted. So before
building a composite, assemble everything that the target mechanically depends on.

WHAT GOALS SCORED DEPENDS ON, and the feature for each:
  own attacking quality   goals/xG/npxG/open-play xG/set-piece xG/deep/shots/
                          shots-on-target/corners/big chances/shots-in-box, xG per shot
  opponent DEFENCE        the conceded mirror of every one of those
  finishing & luck        rolling (goals - xG), both for and against -- exactly what the
                          latents may miss if they are driven by the xG channel
  match context           is_home, stakes, rest, load, crowdless, round-in-season
  league standing         points per game and goal difference per game to date, position
  squad                   red card in the PREVIOUS match (the only pre-open-knowable
                          absence; WP047 showed it is null alone, included for interactions)
  model internals         own and opponent lambda, implied total and difference,
                          predicted outcome probabilities
  EXCLUDED                `intact` (needs the match's own team sheet -- not knowable at
                          the opening price), and the MARKET price, kept separate as the
                          ceiling

LEAK SAFETY. Every form feature is `shift(1)` then `.rolling()`, so strictly prior
matches only. League-standing features accumulate within season up to BEFORE this match.
Red cards come from the team's previous match. Nothing uses the current match's outcome.

WINDOWS 3 / 10 / 25 with mean, rolling sd, and a short-minus-long trend term, because a
single window cannot express form-versus-baseline, which is what "form" means.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
from analysis import attach_meta, load_checkpoints  # noqa: E402
from market_analysis import CODE_TO_FD  # noqa: E402

WINDOWS = (3, 10, 25)
SET_PIECE = {"FromCorner", "SetPiece", "DirectFreekick"}
OUT = HERE / "features.pkl"


def shot_detail():
    """per team-match attacking detail from the cached shot feed (WP014)."""
    rows = []
    for f in sorted((WP / "wp014_npxg_decomposition" / "shots").glob("*.pkl")):
        for rec in pickle.loads(f.read_bytes()).values():
            m = rec["meta"]
            d = pd.Timestamp(m["datetime"]).normalize()
            for side, team in (("h", m["home_short"]), ("a", m["away_short"])):
                sh = rec[side]
                npx = sum(s["xG"] for s in sh if s["situation"] != "Penalty")
                rows.append(dict(
                    t=team, d=d,
                    npxg=npx,
                    xg_op=sum(s["xG"] for s in sh if s["situation"] == "OpenPlay"),
                    xg_sp=sum(s["xG"] for s in sh if s["situation"] in SET_PIECE),
                    bigch=sum(1 for s in sh if s["xG"] > 0.3),
                    inbox=sum(1 for s in sh if s["X"] > 0.84),
                    nsh=len(sh),
                ))
    df = pd.DataFrame(rows)
    df["xg_per_shot"] = df.npxg / df.nsh.clip(lower=1)
    return df.set_index(["t", "d"])


def fd_detail():
    """shots on target, corners and reds from football-data (100% coverage)."""
    fd = pd.read_pickle(WP / "wp044_league_scan" / "odds_all_leagues.pkl")["EPL"].copy()
    fd["d"] = pd.to_datetime(fd.Date, dayfirst=True, errors="coerce").dt.normalize()
    rows = []
    for _, r in fd.iterrows():
        rows.append(dict(fd=r.HomeTeam, d=r.d, sot=r.HST, sot_a=r.AST,
                         corn=r.HC, corn_a=r.AC, reds=r.HR))
        rows.append(dict(fd=r.AwayTeam, d=r.d, sot=r.AST, sot_a=r.HST,
                         corn=r.AC, corn_a=r.HC, reds=r.AR))
    return pd.DataFrame(rows).dropna(subset=["fd", "d"]).set_index(["fd", "d"])


def main():
    base = pickle.loads((WP / "wp039_market_channel/cv_shared_data_market.pkl")
                        .read_bytes())["df_cv"]
    crowd = pickle.loads((WP / "wp031_hfa_step/cv_shared_data_crowd.pkl")
                         .read_bytes())["df_cv"][["datetime", "team", "crowdless"]]
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = attach_meta(m[m.arm == "W075"], base)
    h = base[base.is_home == 1].copy()
    h["d"] = pd.to_datetime(h.datetime).dt.normalize()

    # ---- long per-team-match panel: every quantity FOR and AGAINST ----
    def side(team_col, opp_col, suff):
        return pd.DataFrame(dict(
            d=h.d, season=h.season, rnd=h["round"], t=h[team_col], home=1.0 if suff == "h" else 0.0,
            gf=h.goals_home if suff == "h" else h.goals_away,
            ga=h.goals_away if suff == "h" else h.goals_home,
            xgf=h.xG_home if suff == "h" else h.xG_away,
            xga=h.xG_away if suff == "h" else h.xG_home,
            deepf=h.deep_home if suff == "h" else h.deep_away,
            deepa=h.deep_away if suff == "h" else h.deep_home,
            shf=h.shots_home if suff == "h" else h.shots_away,
            sha=h.shots_away if suff == "h" else h.shots_home,
        ))
    long = pd.concat([side("team", "opp_team", "h"), side("opp_team", "team", "a")])
    long = long.sort_values("d").reset_index(drop=True)

    # attach shot-level and football-data detail (own side only; the opponent's
    # version is picked up later by looking up the OPPONENT's row)
    sd = shot_detail()
    long = long.join(sd, on=["t", "d"])
    long["fd"] = long.t.map(CODE_TO_FD)
    long = long.join(fd_detail(), on=["fd", "d"])
    long = long.join(crowd.assign(d=pd.to_datetime(crowd.datetime).dt.normalize())
                     .set_index(["team", "d"]).crowdless, on=["t", "d"])

    # derived: finishing / luck, and shot quality
    long["fin"] = long.gf - long.xgf            # over/under-performing xG, attacking
    long["fin_a"] = long.ga - long.xga          # same defensively (shot-stopping/luck)
    long["sot_rate"] = long.sot / long.shf.clip(lower=1)

    QTY = ["gf", "ga", "xgf", "xga", "deepf", "deepa", "shf", "sha", "npxg",
           "xg_op", "xg_sp", "bigch", "inbox", "xg_per_shot", "sot", "sot_a",
           "corn", "corn_a", "fin", "fin_a", "sot_rate"]
    g = long.groupby("t")
    cols = []
    for c in QTY:
        for K in WINDOWS:
            long[f"{c}_m{K}"] = g[c].transform(
                lambda x, K=K: x.shift(1).rolling(K, min_periods=max(3, K // 3)).mean())
            cols.append(f"{c}_m{K}")
        long[f"{c}_sd"] = g[c].transform(
            lambda x: x.shift(1).rolling(10, min_periods=5).std())
        long[f"{c}_tr"] = long[f"{c}_m3"] - long[f"{c}_m25"]
        cols += [f"{c}_sd", f"{c}_tr"]
    # home/away-split form for the headline quantities
    for c in ("gf", "ga", "xgf", "xga"):
        for hv, tag in ((1.0, "H"), (0.0, "A")):
            long[f"{c}_{tag}"] = (long[c].where(long.home == hv).groupby(long.t)
                                  .transform(lambda x: x.shift(1).rolling(8, min_periods=3).mean()))
            cols.append(f"{c}_{tag}")
    # league standing SO FAR this season, strictly before this match
    long["pts"] = np.where(long.gf > long.ga, 3.0, np.where(long.gf == long.ga, 1.0, 0.0))
    gs = long.groupby(["t", "season"])
    for c, nm in (("pts", "ppg_todate"), ("gf", "gfpg_todate"), ("ga", "gapg_todate")):
        long[nm] = gs[c].transform(lambda x: x.shift(1).expanding().mean())
        cols.append(nm)
    long["gdpg_todate"] = long.gfpg_todate - long.gapg_todate
    long["reds_prev"] = g.reds.transform(lambda x: x.shift(1))
    cols += ["gdpg_todate", "reds_prev", "crowdless"]

    form = long.set_index(["t", "d"])[cols]
    print(f"per-team form features: {len(cols)}")

    # ---- assemble one row per (match, side) ----
    mt = h[["d", "team", "opp_team", "stakes_home", "stakes_away", "rest_home",
            "rest_away", "load_home", "load_away", "market_lh", "market_la",
            "market_obs_w", "round", "season"]].set_index(["team", "opp_team", "d"])
    first = base.groupby("season")["round"].min()
    m["d"] = pd.to_datetime(m.date).dt.normalize()
    rows, miss = [], 0
    for _, r in m.iterrows():
        try:
            c = mt.loc[(r.home, r.away, r.d)]
        except KeyError:
            miss += 1; continue
        for s_ in ("h", "a"):
            me, opp = (r.home, r.away) if s_ == "h" else (r.away, r.home)
            gl = r.goals_home if s_ == "h" else r.goals_away
            lam = r.lambda_home if s_ == "h" else r.lambda_away
            lam_o = r.lambda_away if s_ == "h" else r.lambda_home
            try:
                fm, fo = form.loc[(me, r.d)], form.loc[(opp, r.d)]
            except KeyError:
                miss += 1; continue
            rec = dict(window=r.window, season=r.season, resid=gl - lam,
                       is_home=1.0 if s_ == "h" else 0.0,
                       lam=lam, lam_opp=lam_o, lam_tot=lam + lam_o, lam_diff=lam - lam_o,
                       p_home=r.p_home, p_draw=r.p_draw, p_away=r.p_away,
                       sr=r["round"] - first[r.season] + 1,
                       stakes_own=c.stakes_home if s_ == "h" else c.stakes_away,
                       stakes_opp=c.stakes_away if s_ == "h" else c.stakes_home,
                       rest_own=c.rest_home if s_ == "h" else c.rest_away,
                       rest_opp=c.rest_away if s_ == "h" else c.rest_home,
                       load_own=c.load_home if s_ == "h" else c.load_away,
                       load_opp=c.load_away if s_ == "h" else c.load_home,
                       mkt_own=(c.market_lh if s_ == "h" else c.market_la)
                               if c.market_obs_w > 0 else np.nan,
                       mkt_opp=(c.market_la if s_ == "h" else c.market_lh)
                               if c.market_obs_w > 0 else np.nan)
            for k in cols:
                rec[f"own_{k}"] = fm[k]
                rec[f"opp_{k}"] = fo[k]
            rows.append(rec)
    D = pd.DataFrame(rows)
    feats = [c for c in D.columns if c not in ("window", "season", "resid",
                                               "mkt_own", "mkt_opp")]
    print(f"rows {len(D)} ({miss} unmatched), TOTAL FEATURES {len(feats)}")
    cov = D[feats].notna().mean()
    print(f"feature coverage: median {cov.median():.1%}, "
          f"{(cov < 0.9).sum()} features below 90%")
    print(f"  worst: {cov.nsmallest(5).round(3).to_dict()}")
    pickle.dump({"D": D, "feats": feats}, OUT.open("wb"))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
