"""WP030 step 1 — player importance by EWMA, with the half-life CHOSEN not guessed.

WHY NOT A FIXED WINDOW. WP011 used a hard bench-minutes threshold and it "fired 4
matches late"; a fixed last-N window also weights a match 10 games ago the same as the
last one and jumps as the window slides.

THE TRAP THIS FEATURE HAS TO AVOID. If the half-life is SHORT, an injured player's
importance decays away -- so by the time you measure "how depleted are we", the missing
player no longer counts as important and the feature reports full strength. The measure
destroys what it is meant to detect. WP011 hit this in another form: `absent_share`
ended up "measuring squad dispersion not absence". So the half-life must be long enough
that a typical injury spell does not erase an established role.

HOW THE HALF-LIFE IS CHOSEN. Not by intuition: by which value best predicts a player's
minutes in their team's NEXT match. That is leak-free (only prior matches are used) and
it is what the quantity is actually for.

MINUTES, NOT xG+xA. WP011/WP012 weighted players by xG+xA, which scores keepers at
0.003 against 0.193 for outfielders -- so those nulls only ever tested ATTACKING
absence, as WP016 pointed out. Minutes are the manager's own revealed assessment of
importance and they are position-neutral.
"""
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROSTERS = HERE.parent / "wp011_squad_instability" / "rosters"
OUT = HERE / "player_minutes.pkl"


def build_minutes():
    """-> one row per (team, match, player) with minutes, INCLUDING 0 for squad
    members who did not play. Players absent from the squad entirely are filled in
    as 0 later, per team-season, since 'not in the squad' is the signal we want."""
    rows = []
    for f in sorted(glob.glob(str(ROSTERS / "*.pkl"))):
        season = Path(f).stem
        for rec in pickle.loads(open(f, "rb").read()).values():
            m = rec["meta"]
            dt = pd.Timestamp(m["datetime"]).normalize()
            for side, team in (("h", m["home_short"]), ("a", m["away_short"])):
                for p in rec[side].values():
                    rows.append(dict(season=season, date=dt, team=team,
                                     player_id=p["player_id"], player=p["player"],
                                     pos=p.get("position", ""),
                                     minutes=float(p.get("time", 0) or 0)))
    d = pd.DataFrame(rows)
    # a player can appear for two clubs in a season (transfer); keep team-specific rows
    d = d.drop_duplicates(["date", "team", "player_id"])
    return d.sort_values("date")


def expand_squad(d):
    """Fill in 0 minutes for team-matches where a squad member does not appear at all.

    Essential: an injured player is simply ABSENT from that match's roster, not listed
    with 0. Treating 'missing row' as 'no observation' would make absence invisible --
    the exact failure mode that produced WP011's false positive.
    """
    out = []
    for (season, team), g in d.groupby(["season", "team"]):
        dates = np.sort(g.date.unique())
        squad = g[["player_id", "player", "pos"]].drop_duplicates("player_id")
        full = pd.MultiIndex.from_product([dates, squad.player_id],
                                          names=["date", "player_id"]).to_frame(index=False)
        full = full.merge(g[["date", "player_id", "minutes"]], on=["date", "player_id"], how="left")
        full["minutes"] = full.minutes.fillna(0.0)
        full["season"] = season
        full["team"] = team
        out.append(full.merge(squad, on="player_id", how="left"))
    return pd.concat(out, ignore_index=True).sort_values(["team", "date"])


def ewma_importance(d, half_life):
    """Leak-free EWMA of minutes/90 over a team's PRIOR matches, per player."""
    lam = 0.5 ** (1.0 / half_life)
    res = np.empty(len(d))
    res[:] = np.nan
    for (team, pid), g in d.groupby(["team", "player_id"], sort=False):
        idx = g.index.to_numpy()
        frac = (g.minutes.to_numpy() / 90.0)
        num = den = 0.0
        for i, v in enumerate(frac):
            res[idx[i]] = (num / den) if den > 0 else np.nan   # BEFORE this match
            num = num * lam + v
            den = den * lam + 1.0
    return res


def main():
    raw = build_minutes()
    print(f"{len(raw)} roster rows, {raw.player_id.nunique()} players, "
          f"seasons {raw.season.min()}-{raw.season.max()}")
    d = expand_squad(raw).reset_index(drop=True)
    print(f"{len(d)} team-match-player rows after filling non-appearances with 0")
    print(f"  share with 0 minutes: {(d.minutes == 0).mean()*100:.1f}%\n")

    print("Choosing the half-life by how well importance predicts the player's")
    print("minutes in THIS match (leak-free: EWMA uses prior matches only).\n")
    y = (d.minutes / 90.0).to_numpy()
    print(f"  {'half-life':>10s} {'corr':>8s} {'MAE':>8s}")
    best = (None, -1)
    for hl in (3, 5, 8, 12, 18, 25, 40):
        imp = ewma_importance(d, hl)
        ok = ~np.isnan(imp)
        c = np.corrcoef(imp[ok], y[ok])[0, 1]
        mae = np.abs(imp[ok] - y[ok]).mean()
        flag = ""
        if c > best[1]:
            best = (hl, c); flag = "  <- best so far"
        print(f"  {hl:10d} {c:8.4f} {mae:8.4f}{flag}")
    print(f"\n  chosen half-life: {best[0]} matches (corr {best[1]:.4f})")
    d["importance"] = ewma_importance(d, best[0])
    d.attrs["half_life"] = best[0]
    d.to_pickle(OUT)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
