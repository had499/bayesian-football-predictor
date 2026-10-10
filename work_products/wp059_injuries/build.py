"""WP059 — availability features from Transfermarkt's dated injury spells.

THE DESIGN POINT THAT DECIDES THIS. Total absence is the WRONG feature. A player who has
been out three months is ALREADY absorbed by the latents: the team's recent form was
produced without him, so the walk has tracked the weakened side. WP030 found exactly this
-- intactness made RPS WORSE, and the diagnosis was that depletion PERSISTS, so the
latents' "contamination" was itself a valid forecast. Four separate arms have now failed by
handing the model a nuisance the walk already carries
(feedback-latents-absorb-better-than-covariates).

So the quantity that can carry NEW information is CHANGE in availability:
  out_w_new   weight of players whose spell STARTED in the last 14 days -- the latents have
              not seen a match without them yet, so this is genuinely unpriced
  ret_w_new   weight of players who RETURNED in the last 14 days -- a returning key player
              is unpriced GOOD news, and no arm in this project has ever tested it
  out_w_old   weight of players out LONGER than 14 days -- should be INERT if the above
              reasoning is right. Included precisely as the falsification test: if
              out_w_old carries as much signal as out_w_new, the mechanism is not
              "unpriced change" and the whole story is wrong.
  out_gk      is the first-choice keeper newly out (WP016: goalkeeping never tested)

WEIGHTS are each player's share of his team's minutes over the team's LAST 5 MATCHES,
computed from matches STRICTLY BEFORE the one being predicted. So "how much of the side
that just played is missing", not "how many bodies".

LEGALITY. A spell counts for a match only if it STARTED >= 2 days before kickoff, so it was
public before the opening line posted. `LEGAL_LAG_DAYS` makes that explicit and testable.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(HERE))
from clubmap import TM_TO_US, match_players   # noqa: E402

LEGAL_LAG_DAYS = 2      # spell must have begun this long before kickoff to be public
NEW_WINDOW = 14         # "recent" change in availability
FORM_MATCHES = 5        # minutes share computed over the team's last N matches
OUT = HERE / "availability.pkl"


def load_spells():
    """pid -> list of (start, end) timestamps. `until` of '-' or 'unknown' = still out."""
    inj = json.loads((HERE / "injuries.json").read_text())
    out, bad = {}, 0
    for pid, spells in inj.items():
        rows = []
        for s in spells:
            a = pd.to_datetime(s.get("from"), format="%d/%m/%Y", errors="coerce")
            b = pd.to_datetime(s.get("until"), format="%d/%m/%Y", errors="coerce")
            if pd.isna(a):
                bad += 1; continue
            if pd.isna(b):
                b = a + pd.Timedelta(days=365)   # open-ended: treat as a long absence
            rows.append((a, b))
        out[pid] = rows
    print(f"  spells: {sum(len(v) for v in out.values())} parsed, {bad} unparseable dates")
    return out


def main():
    spells = load_spells()
    squads = json.loads((HERE / "squads.json").read_text())
    pm = pd.read_pickle(WP / "wp030_squad_intactness" / "player_minutes.pkl")
    pm["date"] = pd.to_datetime(pm.date).dt.normalize()
    pm["season"] = pm.season.astype(int)
    pm = pm[pm.minutes > 0]

    # ---- identity: (team, season, our player name) -> pid, matched WITHIN club-season ----
    name2pid, unmatched_min, total_min = {}, 0.0, 0.0
    for yr, rows in squads.items():
        y = int(yr)
        byclub = {}
        for r in rows:
            byclub.setdefault(r["club"], []).append(r)
        for club, players in byclub.items():
            us = TM_TO_US[club]
            slug2pid = {p["slug"]: p["pid"] for p in players}
            mins = pm[(pm.team == us) & (pm.season == y)].groupby("player").minutes.sum()
            if not len(mins):
                continue
            m = match_players(list(mins.index), list(slug2pid))
            for nm in mins.index:
                total_min += mins[nm]
                if nm in m:
                    name2pid[(us, y, nm)] = slug2pid[m[nm]]
                else:
                    unmatched_min += mins[nm]
    print(f"  identity: {len(name2pid)} (team,season,player) -> pid; "
          f"{unmatched_min/total_min:.2%} of minutes unmatched")

    # ---- per team-match availability ----
    pm = pm.sort_values(["team", "date"])
    recs = []
    for (team, season), g in pm.groupby(["team", "season"]):
        dates = np.sort(g.date.unique())
        for i, d in enumerate(dates):
            if i < 2:
                continue                       # need some history for weights
            hist = g[g.date.isin(dates[max(0, i - FORM_MATCHES):i])]
            w = hist.groupby("player").minutes.sum()
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            gk_hist = hist[hist.pos == "GK"].groupby("player").minutes.sum()
            gk = gk_hist.idxmax() if len(gk_hist) else None
            cutoff = pd.Timestamp(d) - pd.Timedelta(days=LEGAL_LAG_DAYS)
            new_lo = pd.Timestamp(d) - pd.Timedelta(days=NEW_WINDOW)
            o_new = o_old = r_new = 0.0
            n_out = 0
            gk_new_out = 0.0
            for player, wt in w.items():
                pid = name2pid.get((team, int(season), player))
                if pid is None:
                    continue
                for a, b in spells.get(pid, ()):
                    if a <= cutoff and b >= pd.Timestamp(d):        # out, and public
                        n_out += 1
                        if a >= new_lo:
                            o_new += wt
                            if player == gk:
                                gk_new_out = 1.0
                        else:
                            o_old += wt
                        break
                    if new_lo <= b < pd.Timestamp(d) and a < b:     # returned recently
                        r_new += wt
                        break
            recs.append(dict(team=team, d=pd.Timestamp(d), out_w_new=o_new,
                             out_w_old=o_old, ret_w_new=r_new, out_n=n_out,
                             out_gk_new=gk_new_out))
    A = pd.DataFrame(recs)
    print(f"\n  team-matches with availability computed: {len(A)}")
    for c in ("out_w_new", "out_w_old", "ret_w_new", "out_gk_new"):
        nz = (A[c] > 0).mean()
        print(f"    {c:12s} mean {A[c].mean():.4f}  nonzero {nz:.1%}  max {A[c].max():.3f}")
    A.to_pickle(OUT)
    print(f"  wrote {OUT.name}")


if __name__ == "__main__":
    main()
