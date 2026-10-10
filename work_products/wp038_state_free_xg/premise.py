"""WP038 premise — is STATE-FREE xG a better measurement of team quality than full xG?

THE IDEA. The xG channel is the project's only established win (-0.00258), and WP032
showed its weight was under-set, so the model clearly values xG as a second observation
of the latent scoring rate. But full-match xG is CONTAMINATED: WP018 established that
score-state dependence is real -- teams change behaviour once leading or trailing. A
side 2-0 up stops attacking; a side chasing throws bodies forward and inflates its xG.
Neither tells you much about its true quality.

So: measure xG only while the score was LEVEL (or only in the first half), and it should
be a cleaner read on the same latent. WP007's own premise noted first-half xG predicts
second-half GOALS better than first-half goals do, which is this effect already showing
in the data.

THE TEST. Build leak-free rolling means of each candidate measure over a team's previous
K matches, then correlate with the team's goals in its NEXT match. The measure that
predicts future goals best is the better measurement channel.

THE CONFOUND I HAVE TO WATCH. Level-state minutes are not random: strong teams score
early and spend less time level, so their level-state window is systematically the
EARLY part of matches, and they get fewer level minutes to average over. Normalising to
a per-90 rate handles the exposure but not the selection. Both are reported.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
SHOTS = WP / "wp014_npxg_decomposition" / "shots"


def match_measures(rec):
    """Per-side xG under three definitions, from one match's shot list."""
    gh = sorted(s["minute"] for s in rec["h"] if s["result"] == "Goal")
    ga = sorted(s["minute"] for s in rec["a"] if s["result"] == "Goal")

    def lead_at(m):
        """home-minus-away goals STRICTLY BEFORE minute m."""
        return sum(1 for x in gh if x < m) - sum(1 for x in ga if x < m)

    # minutes spent level: walk the merged goal timeline
    events = sorted(set([0] + gh + ga + [95]))
    level_minutes = 0.0
    for a, b in zip(events[:-1], events[1:]):
        if lead_at(a + 0.5) == 0:
            level_minutes += (b - a)
    out = {}
    for side, shots in (("h", rec["h"]), ("a", rec["a"])):
        full = sum(s["xG"] for s in shots)
        level = sum(s["xG"] for s in shots if lead_at(s["minute"]) == 0)
        h1 = sum(s["xG"] for s in shots if s["minute"] <= 45)
        out[side] = dict(full=full, level=level, h1=h1, level_min=level_minutes)
    return out


def main():
    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    recs = {}
    for f in sorted(SHOTS.glob("*.pkl")):
        recs.update(pickle.loads(f.read_bytes()))
    print(f"shot cache: {len(recs)} matches across {len(list(SHOTS.glob('*.pkl')))} seasons")

    rows = []
    miss = 0
    for mid, rec in recs.items():
        try:
            mm = match_measures(rec)
        except Exception:
            miss += 1
            continue
        for side, key in (("h", "home_short"), ("a", "away_short")):
            rows.append(dict(team=rec["meta"][key],
                             datetime=rec["meta"]["datetime"], **mm[side]))
    s = pd.DataFrame(rows)
    s["date"] = pd.to_datetime(s.datetime).dt.date
    print(f"built {len(s)} team-match rows ({miss} unparseable)")

    # Join the team's ACTUAL goals from the model's own panel on (team, date).
    # NOT on match_id: the panel's match_id is an internal int16 index (0,0,5,...),
    # not the Understat id, so that join silently kept 124 of 9,120 rows on the
    # first attempt. (team, date) is the key WP030 established as unique -- the
    # documented trap is (cum_round, team), which is NOT unique because of ISO-week
    # bucketing.
    g = df[["datetime", "team", "goals"]].copy()
    g["date"] = pd.to_datetime(g.datetime).dt.date
    g = g.drop(columns="datetime")
    assert not g.duplicated(["team", "date"]).any(), "(team,date) must be unique"
    before = len(s)
    s = s.merge(g, on=["team", "date"], how="inner")
    print(f"joined to goals: {len(s)}/{before} rows "
          f"({len(s)/before:.1%} — anything well below 100% means a broken key)")
    assert len(s) > 0.9 * before, f"join dropped too much: {len(s)}/{before}"
    s["datetime"] = pd.to_datetime(s.datetime)
    s = s.sort_values("datetime")

    print(f"\nlevel minutes per match: mean {s.level_min.mean():.1f}, "
          f"p10 {s.level_min.quantile(.1):.1f}, p90 {s.level_min.quantile(.9):.1f}")
    print(f"share of a team's xG taken while level: {(s.level.sum()/s.full.sum()):.1%}")
    print(f"share taken in the first half:          {(s.h1.sum()/s.full.sum()):.1%}")

    # per-90 rates so the three measures are on a comparable scale
    s["level_r"] = s.level / np.maximum(s.level_min, 10.0) * 90.0
    s["h1_r"] = s.h1 * 2.0
    s["full_r"] = s.full

    print("\n=== does each measure predict the team's NEXT match goals? ===")
    print("    leak-free: rolling mean over the team's PREVIOUS K matches only.")
    print("    the incumbent channel observes `full`; a better predictor here is the")
    print("    case for switching what the channel observes.\n")
    print(f"  {'K':>3s}  " + "".join(f"{c:>12s}" for c in
                                     ["goals(base)", "full xG", "level xG", "1st-half xG"]))
    for K in (5, 10, 20):
        cells = []
        for col in ("goals", "full_r", "level_r", "h1_r"):
            roll = (s.groupby("team")[col]
                     .transform(lambda x: x.shift(1).rolling(K, min_periods=max(3, K // 2)).mean()))
            k = roll.notna()
            cells.append(np.corrcoef(roll[k], s.goals[k])[0, 1])
        print(f"  {K:3d}  " + "".join(f"{c:12.4f}" for c in cells))

    print("\n=== robustness: predict the NEXT 5 matches' total goals instead ===")
    s["fwd5"] = (s.groupby("team").goals
                  .transform(lambda x: x.shift(-1).rolling(5, min_periods=3).mean()))
    print(f"  {'K':>3s}  " + "".join(f"{c:>12s}" for c in
                                     ["goals(base)", "full xG", "level xG", "1st-half xG"]))
    for K in (10, 20):
        cells = []
        for col in ("goals", "full_r", "level_r", "h1_r"):
            roll = (s.groupby("team")[col]
                     .transform(lambda x: x.shift(1).rolling(K, min_periods=max(3, K // 2)).mean()))
            k = roll.notna() & s.fwd5.notna()
            cells.append(np.corrcoef(roll[k], s.fwd5[k])[0, 1])
        print(f"  {K:3d}  " + "".join(f"{c:12.4f}" for c in cells))

    print("\n  VERDICT RULE: state-free xG must beat FULL xG, not just the goals")
    print("  baseline. If full xG wins, the contamination is not costing anything")
    print("  measurable and the channel should keep observing full-match xG.")




def incremental():
    """Replacing full xG with state-free xG is worse. Does state-free xG add anything
    ALONGSIDE it -- i.e. would a SECOND channel help, even though a substitute does not?

    Leave-one-season-out OOS R^2, because a nested in-sample R^2 gain has a bootstrap
    lower bound of >=0 by construction (BACKLOG caution 7 / WP007's vacuous premise).
    """
    import pickle
    import numpy as np
    import pandas as pd

    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    recs = {}
    for f in sorted(SHOTS.glob("*.pkl")):
        recs.update(pickle.loads(f.read_bytes()))
    rows = []
    for mid, rec in recs.items():
        mm = match_measures(rec)
        for side, key in (("h", "home_short"), ("a", "away_short")):
            rows.append(dict(team=rec["meta"][key], datetime=rec["meta"]["datetime"],
                             **mm[side]))
    s = pd.DataFrame(rows)
    s["date"] = pd.to_datetime(s.datetime).dt.date
    g = df[["datetime", "team", "goals", "season"]].copy()
    g["date"] = pd.to_datetime(g.datetime).dt.date
    s = s.merge(g.drop(columns="datetime"), on=["team", "date"], how="inner")
    s["datetime"] = pd.to_datetime(s.datetime)
    s = s.sort_values("datetime")
    s["level_r"] = s.level / np.maximum(s.level_min, 10.0) * 90.0

    K = 10
    for c in ("full", "level_r", "goals"):
        s[f"r_{c}"] = (s.groupby("team")[c]
                        .transform(lambda x: x.shift(1).rolling(K, min_periods=5).mean()))
    d = s.dropna(subset=["r_full", "r_level_r", "r_goals"]).copy()

    def loso_r2(cols):
        errs, base = [], []
        for season in sorted(d.season.unique()):
            tr, te = d[d.season != season], d[d.season == season]
            X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
            beta, *_ = np.linalg.lstsq(X, tr.goals.to_numpy(), rcond=None)
            Xe = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
            errs.append(((te.goals.to_numpy() - Xe @ beta) ** 2).sum())
            base.append(((te.goals.to_numpy() - tr.goals.mean()) ** 2).sum())
        return 1 - sum(errs) / sum(base)

    print("\n=== 2. does state-free xG add anything ALONGSIDE full xG? ===")
    print("    leave-one-season-out OOS R^2 (a nested IN-SAMPLE gain is vacuous)")
    for lab, cols in (("full xG only", ["r_full"]),
                      ("level xG only", ["r_level_r"]),
                      ("full + level", ["r_full", "r_level_r"]),
                      ("full + goals", ["r_full", "r_goals"])):
        print(f"  {lab:16s} OOS R^2 = {loso_r2(cols):+.5f}")
    print("\n  if 'full + level' does not beat 'full xG only', the split carries no")
    print("  information the full measure lacks, and a second channel is pointless.")


if __name__ == "__main__":
    main()
    incremental()
