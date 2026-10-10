"""WP038 premise 3 — do ANY of the shot-level decompositions add information?

State-free xG failed twice (worse as a substitute, no OOS gain as a complement). The
same cheap test settles the other shot-level candidates in one pass, so the whole
direction can be closed or opened on evidence instead of one idea at a time.

CANDIDATES, all from the cached shot feed (xG / situation / result / minute / X / Y):
  openplay / setpiece   -- `situation` splits OpenPlay, FromCorner, SetPiece,
                           DirectFreekick, Penalty. Set-piece threat is a distinct and
                           persistent skill; one lumped xG forces one kappa and one
                           noise level onto two different processes.
  bigchance             -- count of shots with xG > 0.3. Total xG constrains
                           shots x quality jointly; the COUNT of high-quality chances
                           is extra distributional information. WP019's raw shot count
                           was null, but raw counts are dominated by low-value shots,
                           which is plausibly why it failed.
  shots_in_box          -- count of shots from inside the box (X > 0.84), a
                           location-based quality proxy independent of Understat's
                           own xG model.

TEST: leave-one-season-out OOS R^2 predicting the team's next-match goals, each
candidate ADDED to full xG. The control is `full + goals`, which is known to add real
information (+0.0017 over full alone) -- it proves the test can detect a true signal, so
a null here means "no information", not "no power".
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
SHOTS = WP / "wp014_npxg_decomposition" / "shots"
SET_PIECE = {"FromCorner", "SetPiece", "DirectFreekick"}


def main():
    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    recs = {}
    for f in sorted(SHOTS.glob("*.pkl")):
        recs.update(pickle.loads(f.read_bytes()))

    rows = []
    for rec in recs.values():
        for side, key in (("h", "home_short"), ("a", "away_short")):
            sh = rec[side]
            rows.append(dict(
                team=rec["meta"][key], datetime=rec["meta"]["datetime"],
                full=sum(s["xG"] for s in sh),
                openplay=sum(s["xG"] for s in sh if s["situation"] == "OpenPlay"),
                setpiece=sum(s["xG"] for s in sh if s["situation"] in SET_PIECE),
                bigchance=sum(1 for s in sh if s["xG"] > 0.3),
                shots_in_box=sum(1 for s in sh if s["X"] > 0.84),
                nshots=len(sh),
            ))
    s = pd.DataFrame(rows)
    s["date"] = pd.to_datetime(s.datetime).dt.date
    g = df[["datetime", "team", "goals", "season"]].copy()
    g["date"] = pd.to_datetime(g.datetime).dt.date
    s = s.merge(g.drop(columns="datetime"), on=["team", "date"], how="inner")
    assert len(s) == 9120, f"join broke: {len(s)}"
    s["datetime"] = pd.to_datetime(s.datetime)
    s = s.sort_values("datetime")

    print(f"joined {len(s)} team-match rows")
    print(f"  set-piece share of xG: {s.setpiece.sum()/s.full.sum():.1%}")
    print(f"  big chances per match: {s.bigchance.mean():.2f}")
    print(f"  shots in box per match: {s.shots_in_box.mean():.2f} of {s.nshots.mean():.2f}\n")

    K = 10
    cols = ["full", "openplay", "setpiece", "bigchance", "shots_in_box", "nshots", "goals"]
    for c in cols:
        s[f"r_{c}"] = (s.groupby("team")[c]
                        .transform(lambda x: x.shift(1).rolling(K, min_periods=5).mean()))
    d = s.dropna(subset=[f"r_{c}" for c in cols]).copy()

    def loso_r2(use):
        errs, base = [], []
        for season in sorted(d.season.unique()):
            tr, te = d[d.season != season], d[d.season == season]
            X = np.column_stack([np.ones(len(tr))] + [tr[f"r_{c}"].to_numpy() for c in use])
            beta, *_ = np.linalg.lstsq(X, tr.goals.to_numpy(), rcond=None)
            Xe = np.column_stack([np.ones(len(te))] + [te[f"r_{c}"].to_numpy() for c in use])
            errs.append(((te.goals.to_numpy() - Xe @ beta) ** 2).sum())
            base.append(((te.goals.to_numpy() - tr.goals.mean()) ** 2).sum())
        return 1 - sum(errs) / sum(base)

    solo = loso_r2(["full"])
    print("=== OOS R^2 predicting next-match goals (leave-one-season-out) ===")
    print(f"  {'specification':34s} {'OOS R^2':>10s} {'vs full alone':>14s}")
    print(f"  {'full xG only (incumbent)':34s} {solo:+10.5f} {'—':>14s}")
    for lab, use in (("full + goals  [CONTROL, real]", ["full", "goals"]),
                     ("full + openplay/setpiece split", ["openplay", "setpiece"]),
                     ("full + setpiece", ["full", "setpiece"]),
                     ("full + big chances", ["full", "bigchance"]),
                     ("full + shots in box", ["full", "shots_in_box"]),
                     ("full + raw shot count", ["full", "nshots"]),
                     ("everything", cols)):
        r = loso_r2(use)
        print(f"  {lab:34s} {r:+10.5f} {r - solo:+14.5f}")
    print("\n  the CONTROL must be positive, or the test lacks power and no null below")
    print("  can be trusted. Anything that does not beat full xG alone carries no")
    print("  information the incumbent channel lacks.")


if __name__ == "__main__":
    main()
