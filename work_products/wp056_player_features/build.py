"""WP056 -- feature families that no work product has ever used, for the GBM.

WP053's 253 features are ALL team-match aggregates (goals, xG, shots, deep, corners,
cards, rest/load, standing, model internals). Five families sitting on disk were never
touched by any arm, channel or feature set:

  A  PLAYER / LINEUP HISTORY   wp030's player_minutes.pkl (207k rows, pos + importance).
     Used once, as a model covariate for the CURRENT match's team sheet (WP030), which
     is illegal at the opening price. Built from ALREADY-PLAYED matches it is legal, and
     it has never been a feature. Includes GK continuity -- WP016 established that the
     absence nulls never tested goalkeepers at all.
  C  SCORE-STATE EXPOSURE      the shot feed carries `minute`, so the full scoreline
     timeline is reconstructable. WP018 proved score-state dependence is REAL. Rolling
     xG is biased by it: a team that leads often takes fewer, better chances. These
     features measure that bias directly, which no aggregate can.
  D  SHOT RESULT / Y LOCATION  `result` (Saved/Blocked/Post) and `Y` were never read.
     Faced-shot SAVE rate is a direct goalkeeper shot-stopping measure -- the quantity
     WP016 failed to isolate by splitting the defence latent.
  E  REFEREE                   100% coverage, 51 refs, with cards. Never used anywhere.
  F  TOTALS / ASIAN HANDICAP   wp041's O/U 2.5 and AH opening lines. The market's direct
     estimate of total goals and goal difference. CIRCULAR for any model-vs-market
     comparison, so these are tagged separately and NEVER mixed into the honest arm.

LEAK SAFETY. Every per-team quantity is shift(1) then rolling/expanding, so strictly
prior matches. H2H and referee use shift(1).expanding(). Asserted at the bottom.

GRID. Built against wp055's 343-window TEST_LEN=1 checkpoints, because that is the
deployment-accurate protocol and the GBM has only ever been measured on TEST_LEN=3.
"""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
sys.path.insert(0, str(WP / "wp053_features"))
from analysis import attach_meta, load_checkpoints          # noqa: E402
from market_analysis import CODE_TO_FD                      # noqa: E402
from build import shot_detail, fd_detail, SET_PIECE         # noqa: E402

WINDOWS = (3, 10, 25)
OUT = HERE / "features_v2.pkl"


# ----------------------------------------------------------------- A: players
def player_features():
    """per team-match lineup-history quantities. RAW values for match d; the rolling /
    shift machinery downstream is what makes them leak-safe."""
    pm = pd.read_pickle(WP / "wp030_squad_intactness" / "player_minutes.pkl")
    pm = pm[pm.minutes > 0].copy()
    pm["date"] = pd.to_datetime(pm.date).dt.normalize()
    # STARTERS = top 11 by minutes in that team-match (the `pos` label is sparse:
    # non-Sub rows with minutes>0 range 3..14, so it cannot define the XI)
    pm = pm.sort_values(["team", "date", "minutes"], ascending=[True, True, False])
    pm["rk"] = pm.groupby(["team", "date"]).cumcount()
    pm["is_start"] = (pm.rk < 11).astype(int)
    rows = []
    for (t, d), g in pm.groupby(["team", "date"], sort=True):
        st = set(g.loc[g.is_start == 1, "player_id"])
        gkrow = g[g.pos == "GK"]
        gk = gkrow.player_id.iloc[0] if len(gkrow) else None   # max-minutes GK (sorted)
        mins = g.minutes.to_numpy(dtype=float)
        sh = mins / max(mins.sum(), 1.0)
        rows.append(dict(t=t, d=d, _xi=st, _gk=gk,
                         sq_used=len(g),                      # players used this match
                         mins_hhi=float((sh ** 2).sum()),     # minute concentration
                         imp_xi=float(g.loc[g.is_start == 1, "importance"].mean()),
                         imp_sq=float(g.importance.mean())))
    P = pd.DataFrame(rows).sort_values(["t", "d"]).reset_index(drop=True)

    # continuity against the team's OWN previous match
    P["_xi_prev"] = P.groupby("t")["_xi"].shift(1)
    P["_gk_prev"] = P.groupby("t")["_gk"].shift(1)
    def jac(r):
        a, b = r._xi, r._xi_prev
        if not isinstance(b, set) or not a or not b: return np.nan
        return len(a & b) / len(a | b)
    P["xi_jac"] = P.apply(jac, axis=1)
    P["gk_same"] = [np.nan if (p is None or (isinstance(p, float) and np.isnan(p))) or c is None
                    else float(c == p) for c, p in zip(P._gk, P._gk_prev)]
    # individual fatigue: minutes the CURRENT starters played in the prior 7 / 21 days
    mn = pm.set_index(["player_id", "date"]).minutes.groupby(level=[0, 1]).sum()
    mnd = {k: v for k, v in mn.items()}
    pdates = pm.groupby("player_id").date.apply(lambda s: np.sort(s.unique()))
    def load_days(xi, d, days):
        lo = d - pd.Timedelta(days=days)
        tot = 0.0
        for p in xi:
            ds = pdates.get(p)
            if ds is None: continue
            for dd in ds[(ds > np.datetime64(lo)) & (ds <= np.datetime64(d))]:
                tot += mnd.get((p, pd.Timestamp(dd)), 0.0)
        return tot / max(len(xi), 1)
    P["xi_load7"] = [load_days(x, d, 7) for x, d in zip(P._xi, P.d)]
    P["xi_load21"] = [load_days(x, d, 21) for x, d in zip(P._xi, P.d)]
    return P.drop(columns=["_xi", "_gk", "_xi_prev", "_gk_prev"]).set_index(["t", "d"])


# ------------------------------------------- C + D: score state, result, location
def state_features():
    """score-state exposure, late-game xG, shot result rates and central xG.
    Reconstructs the scoreline minute by minute from goal events in the shot feed."""
    rows = []
    for f in sorted((WP / "wp014_npxg_decomposition" / "shots").glob("*.pkl")):
        for rec in pickle.loads(f.read_bytes()).values():
            m = rec["meta"]
            d = pd.Timestamp(m["datetime"]).normalize()
            H, A = rec["h"], rec["a"]
            gh = sorted(s["minute"] for s in H if s["result"] == "Goal")
            ga = sorted(s["minute"] for s in A if s["result"] == "Goal")
            # goal-difference timeline from the HOME team's perspective
            ev = sorted([(x, +1) for x in gh] + [(x, -1) for x in ga])
            def gd_at(minute):
                g = 0
                for mm, s in ev:
                    if mm <= minute: g += s
                    else: break
                return g
            # minutes spent in each state (home perspective)
            bounds = [0] + [mm for mm, _ in ev] + [90]
            lead = level = trail = 0.0
            for i in range(len(bounds) - 1):
                dur = max(bounds[i + 1] - bounds[i], 0)
                g = gd_at(bounds[i])
                if g > 0: lead += dur
                elif g < 0: trail += dur
                else: level += dur
            for side, team, mine, theirs, sgn in (("h", m["home_short"], H, A, +1),
                                                  ("a", m["away_short"], A, H, -1)):
                L = lead if sgn > 0 else trail
                T = trail if sgn > 0 else lead
                def xg_in_state(shots, want):
                    return sum(s["xG"] for s in shots
                               if np.sign(gd_at(s["minute"]) * sgn) == want)
                n_f, n_a = max(len(mine), 1), max(len(theirs), 1)
                rows.append(dict(
                    t=team, d=d,
                    st_lead=L / 90.0, st_trail=T / 90.0, st_level=level / 90.0,
                    xgf_level=xg_in_state(mine, 0),
                    xgf_trail=xg_in_state(mine, -1),
                    xga_lead=xg_in_state(theirs, 1),   # conceded while WE lead
                    late_xgf=sum(s["xG"] for s in mine if s["minute"] >= 75),
                    late_xga=sum(s["xG"] for s in theirs if s["minute"] >= 75),
                    blk_f=sum(1 for s in mine if s["result"] == "BlockedShot") / n_f,
                    blk_a=sum(1 for s in theirs if s["result"] == "BlockedShot") / n_a,
                    sav_a=sum(1 for s in theirs if s["result"] == "SavedShot") / n_a,
                    post_f=sum(1 for s in mine if s["result"] == "ShotOnPost"),
                    cen_xgf=sum(s["xG"] for s in mine if abs(s["Y"] - 0.5) < 0.12),
                    cen_xga=sum(s["xG"] for s in theirs if abs(s["Y"] - 0.5) < 0.12),
                ))
    return pd.DataFrame(rows).set_index(["t", "d"])


# ------------------------------------------------------------------- E: referee
def referee_features():
    """referee tendencies from PRIOR matches only, keyed by (fd home team, date)."""
    fd = pd.read_pickle(WP / "wp044_league_scan" / "odds_all_leagues.pkl")["EPL"].copy()
    fd["d"] = pd.to_datetime(fd.Date, dayfirst=True, errors="coerce").dt.normalize()
    fd = fd.dropna(subset=["d", "Referee"]).sort_values("d").reset_index(drop=True)
    fd["cards"] = fd.HY.fillna(0) + fd.AY.fillna(0) + 3 * (fd.HR.fillna(0) + fd.AR.fillna(0))
    fd["hpts"] = np.where(fd.FTHG > fd.FTAG, 3.0, np.where(fd.FTHG == fd.FTAG, 1.0, 0.0))
    fd["hgd"] = fd.FTHG - fd.FTAG
    g = fd.groupby("Referee")
    for src, nm in (("cards", "ref_cards"), ("hpts", "ref_hppg"), ("hgd", "ref_hgd")):
        fd[nm] = g[src].transform(lambda x: x.shift(1).expanding().mean())
    fd["ref_n"] = g.cumcount()
    return fd.set_index(["HomeTeam", "d"])[["ref_cards", "ref_hppg", "ref_hgd", "ref_n"]]


# ----------------------------------------------------------------------- B: H2H
def h2h_features(h):
    """prior head-to-head, own perspective, strictly earlier meetings."""
    rows = []
    for _, r in h.iterrows():
        rows.append(dict(me=r.team, op=r.opp_team, d=r.d, home=1,
                         gd=r.goals_home - r.goals_away, xgd=r.xG_home - r.xG_away))
        rows.append(dict(me=r.opp_team, op=r.team, d=r.d, home=0,
                         gd=r.goals_away - r.goals_home, xgd=r.xG_away - r.xG_home))
    L = pd.DataFrame(rows).sort_values("d").reset_index(drop=True)
    g = L.groupby(["me", "op"])
    L["h2h_gd"] = g.gd.transform(lambda x: x.shift(1).expanding().mean())
    L["h2h_xgd"] = g.xgd.transform(lambda x: x.shift(1).expanding().mean())
    L["h2h_n"] = g.cumcount()
    gv = L.groupby(["me", "op", "home"])
    L["h2h_gd_venue"] = gv.gd.transform(lambda x: x.shift(1).expanding().mean())
    return L.set_index(["me", "op", "d"])[["h2h_gd", "h2h_xgd", "h2h_n", "h2h_gd_venue"]]


# ------------------------------------------------------------- F: totals / AH
def market_extra():
    o = pd.read_pickle(WP / "wp041_other_markets" / "odds_other.pkl").copy()
    o["d"] = pd.to_datetime(o.date).dt.normalize()
    po, pu = 1.0 / o.pinnacle_over25, 1.0 / o.pinnacle_under25
    o["mkt_p_over25"] = po / (po + pu)                 # de-vigged P(total > 2.5)
    o["mkt_ah_line"] = o.ah_line
    return o.dropna(subset=["d"]).set_index(["home_team", "away_team", "d"])[
        ["mkt_p_over25", "mkt_ah_line"]]


def main():
    base = pickle.loads((WP / "wp055_horizon1/cv_shared_data_h1.pkl").read_bytes())["df_cv"]
    crowd = pickle.loads((WP / "wp031_hfa_step/cv_shared_data_crowd.pkl")
                         .read_bytes())["df_cv"][["datetime", "team", "crowdless"]]
    m, _ = load_checkpoints(WP / "wp055_horizon1/checkpoints")
    m = attach_meta(m[m.arm == "H1_CHAMP"].copy(), base)
    h = base[base.is_home == 1].copy()
    h["d"] = pd.to_datetime(h.datetime).dt.normalize()
    print(f"grid: {len(m)} predictions, {m.window.nunique()} windows")

    def side(tc, oc, suff):
        return pd.DataFrame(dict(
            d=h.d, season=h.season, rnd=h["round"], t=h[tc],
            home=1.0 if suff == "h" else 0.0,
            gf=h.goals_home if suff == "h" else h.goals_away,
            ga=h.goals_away if suff == "h" else h.goals_home,
            xgf=h.xG_home if suff == "h" else h.xG_away,
            xga=h.xG_away if suff == "h" else h.xG_home,
            deepf=h.deep_home if suff == "h" else h.deep_away,
            deepa=h.deep_away if suff == "h" else h.deep_home,
            shf=h.shots_home if suff == "h" else h.shots_away,
            sha=h.shots_away if suff == "h" else h.shots_home))
    long = pd.concat([side("team", "opp_team", "h"), side("opp_team", "team", "a")])
    long = long.sort_values("d").reset_index(drop=True)

    sd = shot_detail(); long = long.join(sd, on=["t", "d"])
    long["fd"] = long.t.map(CODE_TO_FD)
    long = long.join(fd_detail(), on=["fd", "d"])
    long = long.join(crowd.assign(d=pd.to_datetime(crowd.datetime).dt.normalize())
                     .set_index(["team", "d"]).crowdless, on=["t", "d"])
    long["fin"] = long.gf - long.xgf
    long["fin_a"] = long.ga - long.xga
    long["sot_rate"] = long.sot / long.shf.clip(lower=1)

    # ---- NEW families on to the same panel ----
    PF, SF = player_features(), state_features()
    long = long.join(PF, on=["t", "d"]).join(SF, on=["t", "d"])
    print(f"player-feature coverage {long.xi_jac.notna().mean():.1%}, "
          f"state {long.st_lead.notna().mean():.1%}, gk {long.gk_same.notna().mean():.1%}")

    OLD_QTY = ["gf", "ga", "xgf", "xga", "deepf", "deepa", "shf", "sha", "npxg",
               "xg_op", "xg_sp", "bigch", "inbox", "xg_per_shot", "sot", "sot_a",
               "corn", "corn_a", "fin", "fin_a", "sot_rate"]
    NEW_QTY = ["xi_jac", "gk_same", "sq_used", "mins_hhi", "imp_xi", "imp_sq",
               "xi_load7", "xi_load21",
               "st_lead", "st_trail", "st_level", "xgf_level", "xgf_trail", "xga_lead",
               "late_xgf", "late_xga", "blk_f", "blk_a", "sav_a", "post_f",
               "cen_xgf", "cen_xga"]
    g = long.groupby("t")
    cols, new_cols = [], []
    for q in OLD_QTY + NEW_QTY:
        tgt = new_cols if q in NEW_QTY else cols
        for w in WINDOWS:
            nm = f"{q}_m{w}"
            long[nm] = g[q].transform(lambda x: x.shift(1).rolling(w, min_periods=2).mean())
            tgt.append(nm)
        nm = f"{q}_sd"
        long[nm] = g[q].transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
        tgt.append(nm)
        long[f"{q}_tr"] = long[f"{q}_m3"] - long[f"{q}_m25"]
        tgt.append(f"{q}_tr")
    # home/away-split form for the headline quantities (WP053 parity)
    for q in ("gf", "ga", "xgf", "xga"):
        for hv, tag in ((1.0, "H"), (0.0, "A")):
            long[f"{q}_{tag}"] = (long[q].where(long.home == hv).groupby(long.t)
                                  .transform(lambda x: x.shift(1).rolling(8, min_periods=3).mean()))
            cols.append(f"{q}_{tag}")
    long["pts"] = np.where(long.gf > long.ga, 3.0, np.where(long.gf == long.ga, 1.0, 0.0))
    gs = long.sort_values("d").groupby(["t", "season"])
    for c, nm in (("pts", "ppg_todate"), ("gf", "gfpg_todate"), ("ga", "gapg_todate")):
        long[nm] = gs[c].transform(lambda x: x.shift(1).expanding().mean()); cols.append(nm)
    long["gdpg_todate"] = long.gfpg_todate - long.gapg_todate
    long["reds_prev"] = g.reds.transform(lambda x: x.shift(1))
    cols += ["gdpg_todate", "reds_prev", "crowdless"]
    form = long.set_index(["t", "d"])[cols + new_cols]
    print(f"per-team form features: {len(cols)} old + {len(new_cols)} NEW")

    REF, H2H, MX = referee_features(), h2h_features(h), market_extra()
    mt = h[["d", "team", "opp_team", "stakes_home", "stakes_away", "rest_home",
            "rest_away", "load_home", "load_away", "market_lh", "market_la",
            "market_obs_w", "round", "season", "team_long", "opp_team_long"]
           ].set_index(["team", "opp_team", "d"])
    first = base.groupby("season")["round"].min()
    m["d"] = pd.to_datetime(m.date).dt.normalize()
    rows, miss = [], 0
    for _, r in m.iterrows():
        try: c = mt.loc[(r.home, r.away, r.d)]
        except KeyError: miss += 1; continue
        rf = REF.loc[(CODE_TO_FD.get(r.home), r.d)] if (CODE_TO_FD.get(r.home), r.d) in REF.index else None
        try: mx = MX.loc[(c.team_long, c.opp_team_long, r.d)]
        except KeyError: mx = None
        for s_ in ("h", "a"):
            me, opp = (r.home, r.away) if s_ == "h" else (r.away, r.home)
            gl = r.goals_home if s_ == "h" else r.goals_away
            lam = r.lambda_home if s_ == "h" else r.lambda_away
            lam_o = r.lambda_away if s_ == "h" else r.lambda_home
            try: fm, fo = form.loc[(me, r.d)], form.loc[(opp, r.d)]
            except KeyError: miss += 1; continue
            try: hh = H2H.loc[(me, opp, r.d)]
            except KeyError: hh = None
            # match keys are carried so downstream work can JOIN external series
            # (e.g. WP057's closing-line metric). Without them D is unjoinable and
            # the only recourse is replaying this loop's row order, which is fragile.
            rec = dict(window=r.window, season=r.season, resid=gl - lam,
                       k_round=r['round'], k_home=r.home, k_away=r.away, k_side=s_,
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
            for k in cols: rec[f"own_{k}"] = fm[k]; rec[f"opp_{k}"] = fo[k]
            for k in new_cols: rec[f"N_own_{k}"] = fm[k]; rec[f"N_opp_{k}"] = fo[k]
            for k in ("h2h_gd", "h2h_xgd", "h2h_n", "h2h_gd_venue"):
                v = np.nan if hh is None else hh[k]
                rec[f"N_{k}"] = v * (1 if s_ == "h" else -1) if k in ("h2h_gd", "h2h_xgd", "h2h_gd_venue") else v
            for k in ("ref_cards", "ref_hppg", "ref_hgd", "ref_n"):
                rec[f"N_{k}"] = np.nan if rf is None else rf[k]
            # F: market extras, kept OUT of both honest sets (prefix M_)
            rec["M_mkt_p_over25"] = np.nan if mx is None else mx.mkt_p_over25
            rec["M_mkt_ah_line"] = np.nan if mx is None else mx.mkt_ah_line
            rows.append(rec)
    D = pd.DataFrame(rows)
    META = ("window", "season", "resid", "mkt_own", "mkt_opp",
            "k_round", "k_home", "k_away", "k_side")
    old = [c for c in D.columns if c not in META and not c.startswith(("N_", "M_"))]
    new = [c for c in D.columns if c.startswith("N_")]
    mkt = [c for c in D.columns if c.startswith("M_")]
    print(f"rows {len(D)} ({miss} unmatched) | OLD {len(old)}  NEW {len(new)}  MARKET {len(mkt)}")
    cov = D[old + new].notna().mean()
    print(f"coverage: median {cov.median():.1%}; worst {cov.nsmallest(4).round(3).to_dict()}")
    print(f"  NEW-only coverage median {D[new].notna().mean().median():.1%}")

    # ---- LEAK ASSERTIONS ----
    # a feature built only from prior matches must be missing for a team's FIRST match
    f0 = D[D.window == D.window.min()]
    assert D.N_h2h_n.min() == 0, "h2h_n should start at 0 for a first meeting"
    for c in ("N_own_xi_jac_m3", "N_own_st_lead_m3", "N_ref_cards"):
        assert D[c].isna().any(), f"{c} is never missing -- suspicious for a lagged feature"
    assert abs(np.corrcoef(D.resid, D.N_own_st_lead_m3.fillna(D.N_own_st_lead_m3.mean()))[0,1]) < 0.3
    pickle.dump({"D": D, "old": old, "new": new, "mkt": mkt}, OUT.open("wb"))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
