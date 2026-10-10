"""WP062 — refine the crude availability signal: position, goalkeeper, severity.

WP059's feature is one coefficient on TOTAL unavailable minutes. Three things it ignores:

  POSITION   losing a striker should depress your own scoring; losing a centre-back should
             lift your opponent's. WP059 tested an attack-only and a defence-only form and
             found them nearly redundant -- but that test did not know WHO was missing, so
             it could not distinguish the two mechanisms at all.
  GOALKEEPER a missing first-choice keeper is plausibly worth more than 1/11th of a squad.
             WP016 flagged goalkeeping as never properly tested; WP056 killed GK CONTINUITY
             as a GBM feature, which is a different quantity from a GK-absence multiplier.
  SEVERITY   a knock and a cruciate tear are currently identical.

SEVERITY IS MEASURED LEAK-FREE. The obvious proxy, total spell length, depends on the
`until` date -- the look-ahead WP059c had to audit. Instead this uses DAYS ELAPSED since the
spell began as at the match date, which is known at prediction time: a player 60 days into
an injury is more seriously hurt than one 5 days in, and nothing about the future is used.

POSITION comes from each player's MODAL position across his PRIOR appearances only.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp059_injuries"))
from clubmap import TM_TO_US, match_players   # noqa: E402

LEGAL_LAG_DAYS = 2
FORM_MATCHES = 5
LONG_DAYS = 30
OUT = HERE / "availability_rich.pkl"

DEF_POS = {"GK", "DC", "DR", "DL", "DML", "DMR", "DMC"}
MID_POS = {"MC", "MR", "ML"}
ATT_POS = {"AMC", "AMR", "AML", "FW", "FWL", "FWR"}


def main():
    inj = json.loads((WP / "wp059_injuries" / "injuries.json").read_text())
    spells = {}
    for pid, ss in inj.items():
        rows = []
        for s in ss:
            a = pd.to_datetime(s.get("from"), format="%d/%m/%Y", errors="coerce")
            b = pd.to_datetime(s.get("until"), format="%d/%m/%Y", errors="coerce")
            if pd.isna(a):
                continue
            rows.append((a, a + pd.Timedelta(days=365) if pd.isna(b) else b))
        spells[pid] = rows
    squads = json.loads((WP / "wp059_injuries" / "squads.json").read_text())
    pm = pd.read_pickle(WP / "wp030_squad_intactness" / "player_minutes.pkl")
    pm["date"] = pd.to_datetime(pm.date).dt.normalize()
    pm["season"] = pm.season.astype(int)
    pm = pm[pm.minutes > 0]

    name2pid = {}
    for yr, rows in squads.items():
        y = int(yr); byclub = {}
        for r in rows:
            byclub.setdefault(r["club"], []).append(r)
        for club, players in byclub.items():
            us = TM_TO_US[club]
            s2p = {p["slug"]: p["pid"] for p in players}
            mins = pm[(pm.team == us) & (pm.season == y)].groupby("player").minutes.sum()
            if not len(mins):
                continue
            for nm, sl in match_players(list(mins.index), list(s2p)).items():
                name2pid[(us, y, nm)] = s2p[sl]

    recs = []
    for (team, season), g in pm.sort_values("date").groupby(["team", "season"]):
        dates = np.sort(g.date.unique())
        for i, d in enumerate(dates):
            if i < 2:
                continue
            D = pd.Timestamp(d)
            hist = g[g.date.isin(dates[max(0, i - FORM_MATCHES):i])]
            w = hist.groupby("player").minutes.sum()
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            # modal position from PRIOR appearances only
            pos = (hist[hist.pos != "Sub"].groupby("player").pos
                   .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "MC"))
            gk_h = hist[hist.pos == "GK"].groupby("player").minutes.sum()
            gk = gk_h.idxmax() if len(gk_h) else None
            cutoff = D - pd.Timedelta(days=LEGAL_LAG_DAYS)
            tot = att = mid = dfn = longw = 0.0
            gk_out = 0.0
            for player, wt in w.items():
                pid = name2pid.get((team, int(season), player))
                if pid is None:
                    continue
                for a, b in spells.get(pid, ()):
                    if a <= cutoff and b >= D:
                        tot += wt
                        p = pos.get(player, "MC")
                        if p in ATT_POS: att += wt
                        elif p in MID_POS: mid += wt
                        else: dfn += wt
                        if (D - a).days >= LONG_DAYS:   # leak-free severity proxy
                            longw += wt
                        if player == gk:
                            gk_out = 1.0
                        break
            recs.append(dict(team=team, d=D, out_tot=tot, out_att=att, out_mid=mid,
                             out_def=dfn, out_long=longw, out_gk=gk_out))
    A = pd.DataFrame(recs)
    print(f"team-matches: {len(A)}")
    for c in ("out_tot", "out_att", "out_mid", "out_def", "out_long", "out_gk"):
        print(f"  {c:9s} mean {A[c].mean():.4f}  nonzero {(A[c]>0).mean():5.1%}  max {A[c].max():.3f}")
    A.to_pickle(OUT)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
