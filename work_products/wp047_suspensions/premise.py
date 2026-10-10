"""WP047 premise — do SUSPENSIONS carry pre-open information the model lacks?

WHY THIS IS NOT JUST THE FIFTH ABSENCE TEST. WP011 (lineups), WP012 (absence covariate),
WP016 (keepers) and WP030 (squad intactness) all failed. But every one measured
availability from MINUTES PLAYED, which conflates three different things: long-term
injury, matchday fitness calls, and manager rotation. Two of those are not knowable when
the opening price is posted, so those tests could never have explained our deficit TO THE
OPEN.

A red card is different on every axis that matters:
  * EXOGENOUS -- it is not a manager's choice and not a fitness judgement.
  * PRE-OPEN KNOWABLE -- public the moment it happens, days before the next price.
  * DETERMINISTIC -- the ban is automatic, not probabilistic.
It is the only form of absence that is certain and public at open time. 11.8% of matches
produce one.

LIMITATION, stated up front: football-data gives TEAM-level cards, so we know a team will
be a player short but not WHICH player. Yellow-card accumulation (5 -> 1 match ban) needs
player-level data we do not have, so only reds are testable.

TWO COMPETING HYPOTHESES, which is what makes this worth measuring:
  (a) the team is missing a player now  -> they UNDERPERFORM.
  (b) last match was played a man down, the model saw that result and over-reacted
      -> it UNDER-RATES them -> they OVERPERFORM.
(b) is WP030's "correct the training rows" logic, which failed there -- but a red card is
a far cleaner shock than a minutes-based availability index.

BAR (WP038's calibration anchor): raw shot count added +0.00484 OOS R^2 for next-match
goals and FAILED as a channel. Below ~+0.005, presume null.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
for p in (WP / "wp007_structural_fixes", WP / "wp008_market_benchmark"):
    sys.path.insert(0, str(p))
from analysis import attach_meta, load_checkpoints  # noqa: E402
from market_analysis import CODE_TO_FD  # noqa: E402


def boot_corr(x, y, g, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(g); idx = {k: np.where(g == k)[0] for k in uq}
    out = np.empty(n)
    for i in range(n):
        sel = np.concatenate([idx[k] for k in rng.choice(uq, len(uq), replace=True)])
        out[i] = np.corrcoef(x[sel], y[sel])[0, 1]
    return np.corrcoef(x, y)[0, 1], *np.percentile(out, [2.5, 97.5])


def main():
    fd = pd.read_pickle(WP / "wp044_league_scan" / "odds_all_leagues.pkl")["EPL"].copy()
    fd["Date"] = pd.to_datetime(fd.Date, dayfirst=True, errors="coerce").dt.normalize()
    fd = fd.dropna(subset=["HomeTeam", "AwayTeam", "HR", "AR"])

    # long panel of (team, date, reds_received)
    rows = []
    for _, r in fd.iterrows():
        rows.append((r.HomeTeam, r.Date, float(r.HR)))
        rows.append((r.AwayTeam, r.Date, float(r.AR)))
    cards = pd.DataFrame(rows, columns=["team_fd", "d", "reds"]).sort_values("d")
    # a red in the PREVIOUS match => a suspension NOW. Strictly prior information.
    cards["reds_prev"] = cards.groupby("team_fd").reds.shift(1)
    key = cards.set_index(["team_fd", "d"]).reds_prev

    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = attach_meta(m[m.arm == "W075"], df)
    m["d"] = pd.to_datetime(m.date).dt.normalize()
    m["h_fd"] = m.home.map(CODE_TO_FD); m["a_fd"] = m.away.map(CODE_TO_FD)
    m["susp_h"] = key.reindex(pd.MultiIndex.from_arrays([m.h_fd, m.d])).to_numpy()
    m["susp_a"] = key.reindex(pd.MultiIndex.from_arrays([m.a_fd, m.d])).to_numpy()
    cov = m[["susp_h", "susp_a"]].notna().all(axis=1)
    print(f"champion test set n={len(m)}, suspension flag resolved for {cov.mean():.1%}")
    m = m[cov].copy()
    print(f"  matches where the HOME side has a suspension: {(m.susp_h > 0).mean():.1%}")
    print(f"  matches where the AWAY side has a suspension: {(m.susp_a > 0).mean():.1%}")
    print(f"  either side: {((m.susp_h > 0) | (m.susp_a > 0)).mean():.1%}  "
          f"(n={((m.susp_h>0)|(m.susp_a>0)).sum()})")

    # performance residual vs the model's own expectation
    m["res_h"] = m.goals_home - m.lambda_home
    m["res_a"] = m.goals_away - m.lambda_away
    # differential: >0 means the home side did better than the model expected
    m["tilt"] = m.res_h - m.res_a
    # suspension differential: >0 means the AWAY side is the one short-handed
    m["susp_diff"] = (m.susp_a > 0).astype(float) - (m.susp_h > 0).astype(float)
    g = m.window.to_numpy()

    print("\n=== 1. does a suspension predict the model's error? ===")
    print("    susp_diff > 0 = the AWAY side is short-handed, so hypothesis (a)")
    print("    predicts a POSITIVE correlation with the home tilt.")
    c, lo, hi = boot_corr(m.susp_diff.to_numpy(), m.tilt.to_numpy(), g)
    print(f"  corr(susp_diff, tilt) = {c:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"  [for scale: intactness +0.1469, Elo +0.047, stakes +0.040, congestion -0.020]")

    print("\n=== 2. split by side, in goals ===")
    for lab, k in (("home side suspended", (m.susp_h > 0).to_numpy()),
                   ("away side suspended", (m.susp_a > 0).to_numpy()),
                   ("neither", ((m.susp_h == 0) & (m.susp_a == 0)).to_numpy())):
        if k.sum() < 60:
            print(f"  {lab:22s} n={k.sum():5d}  too few"); continue
        t = m.tilt.to_numpy()[k]
        rng = np.random.default_rng(1)
        gg = g[k]; uq = np.unique(gg); idx = {x: np.where(gg == x)[0] for x in uq}
        bs = np.array([t[np.concatenate([idx[x] for x in rng.choice(uq, len(uq), True)])].mean()
                       for _ in range(3000)])
        print(f"  {lab:22s} n={k.sum():5d}  tilt {t.mean():+.4f} "
              f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
    print("\n  hypothesis (a) says a suspended side UNDERPERFORMS: home-suspended tilt")
    print("  should be NEGATIVE and away-suspended POSITIVE. Hypothesis (b) -- the model")
    print("  over-reacted to a 10-man result and now under-rates them -- predicts the")
    print("  OPPOSITE signs.")

    print("\n=== 3. incremental OOS R^2 for next-match goals (WP038's bar) ===")
    long = pd.concat([
        m[["d", "home", "season", "goals_home", "lambda_home", "susp_h"]]
          .rename(columns={"home": "team", "goals_home": "goals",
                           "lambda_home": "lam", "susp_h": "susp"}),
        m[["d", "away", "season", "goals_away", "lambda_away", "susp_a"]]
          .rename(columns={"away": "team", "goals_away": "goals",
                           "lambda_away": "lam", "susp_a": "susp"}),
    ]).sort_values("d")
    long["susp_bin"] = (long.susp > 0).astype(float)
    d = long.dropna(subset=["lam", "goals", "susp_bin"]).copy()

    def loso(cols):
        errs, base = [], []
        for s in sorted(d.season.unique()):
            tr, te = d[d.season != s], d[d.season == s]
            X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
            b, *_ = np.linalg.lstsq(X, tr.goals.to_numpy(), rcond=None)
            Xe = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
            errs.append(((te.goals.to_numpy() - Xe @ b) ** 2).sum())
            base.append(((te.goals.to_numpy() - tr.goals.mean()) ** 2).sum())
        return 1 - sum(errs) / sum(base)

    a = loso(["lam"]); b = loso(["lam", "susp_bin"])
    print(f"  model lambda only        OOS R^2 {a:+.5f}")
    print(f"  + suspension flag        OOS R^2 {b:+.5f}   ({b-a:+.5f})")
    print(f"  BAR: raw shot count scored +0.00484 here and FAILED as a channel.")
    print(f"  => {'worth building' if b-a > 0.005 else 'presume NULL, do not build'}")


if __name__ == "__main__":
    main()
