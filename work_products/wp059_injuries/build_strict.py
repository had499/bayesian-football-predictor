"""WP059c — a STRICTLY leak-free availability feature that never reads a spell's END date.

The user's point about Elo was that the FEED was unreliable, not that it was external. The
equivalent reliability question here is whether Transfermarkt back-fills a spell's `until`
from the player's comeback appearance. If it does, then "this spell covers match D" encodes
"he did not play in D" -- confirmed-lineup information, which is excluded by the open-price
rule. The diagnostic was reassuring but not clean: only 2.4% of `until` dates fall exactly
on the next appearance (a back-fill would spike hard at 0), yet 15.5% land within a day.

So rather than rely on that, this build DISCARDS the end date entirely. A player counts as
unavailable for a match on date D iff:
   (a) a spell STARTED on or before D - 2 days   (public before the opening line), AND
   (b) he has made NO appearance in any of his team's matches from that spell's start up to
       the last match BEFORE D.
Condition (b) uses only PAST team sheets, which are historical fact and fully available at
the open. Nothing about match D is consulted. The cost is that a player who quietly
recovered is still counted out until he actually reappears -- a conservative error that
makes the feature noisier, never optimistic.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(HERE))
from clubmap import TM_TO_US, match_players   # noqa: E402

LEGAL_LAG_DAYS = 2
FORM_MATCHES = 5
OUT = HERE / "availability_strict.pkl"


def main():
    inj = json.loads((HERE / "injuries.json").read_text())
    starts = {}
    for pid, spells in inj.items():
        ds = [pd.to_datetime(s.get("from"), format="%d/%m/%Y", errors="coerce")
              for s in spells]
        starts[pid] = np.array(sorted(d for d in ds if pd.notna(d)))
    squads = json.loads((HERE / "squads.json").read_text())
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
        appear = {p: np.sort(sub.date.unique()) for p, sub in g.groupby("player")}
        for i, d in enumerate(dates):
            if i < 2:
                continue
            D = pd.Timestamp(d)
            hist = g[g.date.isin(dates[max(0, i - FORM_MATCHES):i])]
            w = hist.groupby("player").minutes.sum()
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            gk_h = hist[hist.pos == "GK"].groupby("player").minutes.sum()
            gk = gk_h.idxmax() if len(gk_h) else None
            cutoff = D - pd.Timedelta(days=LEGAL_LAG_DAYS)
            prev_dates = dates[:i]                       # matches strictly BEFORE D
            out_w = 0.0; n_out = 0; gk_out = 0.0
            for player, wt in w.items():
                pid = name2pid.get((team, int(season), player))
                if pid is None:
                    continue
                st = starts.get(pid)
                if st is None or not len(st):
                    continue
                began = st[st <= np.datetime64(cutoff)]
                if not len(began):
                    continue
                last_start = pd.Timestamp(began[-1])
                ap = appear.get(player, np.array([], dtype="datetime64[ns]"))
                # has he played in any PRIOR match since that spell began?
                played_since = ((ap >= np.datetime64(last_start))
                                & (ap < np.datetime64(D))).any()
                if not played_since and (prev_dates >= np.datetime64(last_start)).any():
                    # a spell began, and at least one of his team's matches has been
                    # played since, in none of which he featured -> still out
                    out_w += wt; n_out += 1
                    if player == gk:
                        gk_out = 1.0
            recs.append(dict(team=team, d=D, out_strict=out_w, out_n_strict=n_out,
                             gk_out_strict=gk_out))
    A = pd.DataFrame(recs)
    print(f"team-matches: {len(A)}")
    for c in ("out_strict", "gk_out_strict"):
        print(f"  {c:14s} mean {A[c].mean():.4f}  nonzero {(A[c]>0).mean():.1%}  "
              f"max {A[c].max():.3f}")
    A.to_pickle(OUT)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
