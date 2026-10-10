"""WP018 step 1 — does the goal-scoring rate depend on the current SCORE?

WP017 established that the model's home/away outcome tilt cannot be fixed by
re-weighting the objective: given lambda_home and lambda_away, the three outcome
probabilities are fully DETERMINED by convolving two independent Poissons, so the
true H/D/A point may simply lie off the reachable surface. The obvious mechanism
that would put it there is state dependence — a leading side sits back, a
trailing side pushes — which induces dependence between the two goal counts that
independent Poisson cannot represent and Dixon-Coles' four-cell patch cannot fix.

This measures that directly from shot minutes, before any modelling.

    python work_products/wp018_score_state/diagnose_state.py
"""
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SHOTS = Path(__file__).resolve().parents[1] / "wp014_npxg_decomposition" / "shots"
MAX_MIN = 95


def goal_timelines():
    """-> list of sorted [(minute, 'h'|'a')] per match. Own goals are credited to
    the side that benefits, matching how the scoreline actually moved."""
    out = []
    for f in sorted(SHOTS.glob("*.pkl")):
        for rec in pickle.loads(f.read_bytes()).values():
            gs = []
            for side in "ha":
                other = "a" if side == "h" else "h"
                for s in rec[side]:
                    if s["result"] == "Goal":
                        gs.append((int(s["minute"]), side))
                    elif s["result"] == "OwnGoal":
                        gs.append((int(s["minute"]), other))
            out.append(sorted(gs))
    return out


def rates_by_state(timelines, clip=2):
    """Exposure-weighted scoring rate per 90 min, split by the home side's
    current lead. Walks each match event to event, attributing the interval's
    exposure to the state that held DURING it and each goal to the state that
    held just BEFORE it."""
    expo, gh, ga = defaultdict(float), defaultdict(int), defaultdict(int)
    for gs in timelines:
        h = a = 0
        prev = 0
        for mn, side in gs + [(MAX_MIN, None)]:
            mn = min(max(mn, prev), MAX_MIN)
            st = int(np.clip(h - a, -clip, clip))
            expo[st] += mn - prev
            if side == "h":
                gh[st] += 1; h += 1
            elif side == "a":
                ga[st] += 1; a += 1
            prev = mn
    rows = []
    for st in sorted(expo):
        if expo[st] < 2000:      # too little exposure to say anything
            continue
        rows.append({"home_lead": st, "exposure_min": round(expo[st]),
                     "home_rate": gh[st] / expo[st] * 90,
                     "away_rate": ga[st] / expo[st] * 90,
                     "n_home_goals": gh[st], "n_away_goals": ga[st]})
    return pd.DataFrame(rows)


def main():
    tl = goal_timelines()
    tot_goals = sum(len(x) for x in tl)
    print(f"{len(tl)} matches, {tot_goals} goals, {tot_goals/len(tl):.2f} per match")
    minutes = [m for gs in tl for m, _ in gs]
    print(f"goal minutes: min {min(minutes)} median {int(np.median(minutes))} max {max(minutes)}"
          f"  <- must NOT all be 0 (the first cache lacked the field)\n")

    df = rates_by_state(tl)
    df["ratio_h_over_a"] = df.home_rate / df.away_rate
    print("Scoring rate per 90 min, by the home side's current lead:")
    print(df.round(3).to_string(index=False))

    lvl = df[df.home_lead == 0].iloc[0]
    print(f"\nLevel (0-0 etc): home {lvl.home_rate:.3f} vs away {lvl.away_rate:.3f}"
          f"  -> ratio {lvl.ratio_h_over_a:.2f}")
    for st in (1, -1):
        if (df.home_lead == st).any():
            r = df[df.home_lead == st].iloc[0]
            who = "leading" if st > 0 else "trailing"
            print(f"Home {who} by 1: home {r.home_rate:.3f} vs away {r.away_rate:.3f}"
                  f"  -> ratio {r.ratio_h_over_a:.2f}")
    print("\nIndependent Poisson assumes these rates are CONSTANT across states.")
    print("A leading side's rate falling while its opponent's rises is the")
    print("dependence that no (lambda_home, lambda_away) pair can express.")


if __name__ == "__main__":
    main()
