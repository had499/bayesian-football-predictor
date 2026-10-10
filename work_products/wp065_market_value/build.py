"""WP065b — value-weighted availability, and relative squad value.

TWO FEATURES, for two different reasons.

`out_val` -- VALUE-weighted unavailability, the direct fix to a known blind spot. WP059's
working feature weights absent players by their SHARE OF RECENT MINUTES, so a new signing
injured early has ~0 weight however good he is. Weighting by market value instead sees him
from the day he signs. Same spell logic, same >=2-day legality rule; only the weights change.

`relval` -- the home team's share of the two sides' combined squad value, as a crude
forward-looking strength signal independent of results.

LEAK RISK TO AUDIT, NOT ASSUME. Transfermarkt historical squad pages might show a
START-of-season snapshot (legal) or a REVISED end-of-season value (a look-ahead leak: a
player who broke out mid-season would carry a value partly determined by the results we are
trying to predict). League totals rising monotonically 5.9bn -> 13.7bn suggests per-season
snapshots rather than current values, but that is suggestive, not proof. `test.py` audits it
by splitting the gain into EARLY vs LATE season: a start-of-season snapshot should predict
early matches at least as well as late ones, whereas end-of-season revision would show the
gain concentrated LATE. Same shape of audit as WP059c's borderline/unambiguous split.
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
OUT = HERE / "value_features.pkl"


def main():
    vals = json.loads((HERE / "values.json").read_text())
    inj = json.loads((WP / "wp059_injuries" / "injuries.json").read_text())
    squads = json.loads((WP / "wp059_injuries" / "squads.json").read_text())
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
    # (season, pid) -> value, and (season, US code) -> squad total
    val = {}
    club_tot = {}
    for yr, rows in vals.items():
        y = int(yr)
        for r in rows:
            val[(y, r["pid"])] = r["value_m"]
            us = TM_TO_US[r["club"]]
            club_tot[(y, us)] = club_tot.get((y, us), 0.0) + r["value_m"]

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
    miss_val = tot_val = 0
    for (team, season), g in pm.sort_values("date").groupby(["team", "season"]):
        dates = np.sort(g.date.unique())
        sq_total = club_tot.get((int(season), team))
        for i, d in enumerate(dates):
            if i < 2:
                continue
            D = pd.Timestamp(d)
            hist = g[g.date.isin(dates[max(0, i - FORM_MATCHES):i])]
            w = hist.groupby("player").minutes.sum()
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            cutoff = D - pd.Timedelta(days=LEGAL_LAG_DAYS)
            out_min = 0.0
            out_eur = 0.0
            for player, wt in w.items():
                pid = name2pid.get((team, int(season), player))
                if pid is None:
                    continue
                pv = val.get((int(season), pid))
                tot_val += 1
                if pv is None:
                    miss_val += 1
                for a, b in spells.get(pid, ()):
                    if a <= cutoff and b >= D:
                        out_min += wt
                        if pv is not None:
                            out_eur += pv
                        break
            recs.append(dict(team=team, d=D, season=int(season),
                             out_min=out_min,
                             out_val=(out_eur / sq_total) if sq_total else np.nan,
                             sq_value_m=sq_total))
    A = pd.DataFrame(recs)
    print(f"team-matches {len(A)};  players with no value: {miss_val}/{tot_val} "
          f"({miss_val/max(tot_val,1):.1%})")
    print(f"  out_min  mean {A.out_min.mean():.4f}  nonzero {(A.out_min>0).mean():.1%}")
    print(f"  out_val  mean {A.out_val.mean():.4f}  nonzero {(A.out_val>0).mean():.1%}")
    print(f"  corr(out_min, out_val) = {A[['out_min','out_val']].corr().iloc[0,1]:.3f}"
          f"   <- if ~1.0 the new weighting adds nothing")
    print(f"  squad value: median EUR {A.sq_value_m.median():,.0f}m, "
          f"range {A.sq_value_m.min():,.0f}-{A.sq_value_m.max():,.0f}m")
    A.to_pickle(OUT)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
