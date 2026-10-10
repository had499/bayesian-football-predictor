"""WP039 premise — is the MARKET a better measurement channel than xG?

THE IDEA. WP037 established that across ~25 arms only LIKELIHOOD changes have ever
improved this model: new OBSERVATIONS about the same matches. WP038 then closed the
shot-level feed. The one untried source of genuinely new information is the BETTING
MARKET itself.

WHY IT IS LEGITIMATE, and not leakage. A measurement channel observes something about
PAST matches to sharpen the latents; it needs nothing about the match being priced. So
when predicting a match in round R, the channel reads the market's CLOSING line for
matches in rounds < R only. The test match's own line is never touched. That is exactly
how the xG channel works -- xG for the priced match is also unavailable in advance.

WHY IT SHOULD CARRY INFORMATION THE MODEL LACKS. WP010 measured that the model is
already at PARITY with Pinnacle's pre-close line, that only ~29% of the gap to the
closing line is late information, and that the model's deficit is concentrated in
HIGH-LINE-MOVEMENT matches. The closing line therefore contains something about team
strength that our latents never see -- and for PAST matches it is free to use.

THE CALIBRATION BAR (WP038). On this exact test, raw shot count adds +0.00484 OOS R^2
for next-match goals -- the largest of any shot-level candidate -- and WP019 BUILT it as
a channel and it was NULL, actively cancelling deep. So **below ~+0.005 OOS R^2, presume
null.** The market has to clear that bar by a distance to be worth building, and the
`full + goals` control (+0.00170) proves the test can detect a true signal at all.

The market-implied rates are obtained by inverting a Poisson scoreline model on the
de-vigged 1X2 probabilities: two free probabilities, two rates, exactly identified.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
from market_analysis import CODE_TO_FD, add_devig_probs  # noqa: E402

MAXG = 10
PIN_CLOSE = ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close", "pinnacle_1x2_away_close"]
PIN_OPEN = ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"]


def poisson_probs(lh, la):
    ph = poisson.pmf(np.arange(MAXG + 1), lh)
    pa = poisson.pmf(np.arange(MAXG + 1), la)
    g = np.outer(ph, pa)
    return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])


def invert(p):
    """de-vigged (p_home, p_draw, p_away) -> (lambda_home, lambda_away)."""
    def resid(x):
        return poisson_probs(np.exp(x[0]), np.exp(x[1])) - p
    r = least_squares(resid, x0=np.log([1.5, 1.2]), method="lm")
    return np.exp(r.x[0]), np.exp(r.x[1])


def main():
    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    odds = pd.read_pickle(WP / "wp008_market_benchmark" / "odds_raw.pkl")

    # match-level panel rows (home perspective), mapped to football-data names
    mt = df[df.is_home == 1][["datetime", "team", "opp_team", "goals_home", "goals_away",
                              "season", "round"]].copy()
    mt["date"] = pd.to_datetime(mt.datetime).dt.normalize()
    mt["home_fd"] = mt.team.map(CODE_TO_FD)
    mt["away_fd"] = mt.opp_team.map(CODE_TO_FD)
    assert mt.home_fd.notna().all() and mt.away_fd.notna().all(), "CODE_TO_FD gap"

    o = odds.copy()
    o["date"] = pd.to_datetime(o.date).dt.normalize()
    cols = ["date", "home_team", "away_team", "fthg", "ftag"] + PIN_CLOSE + PIN_OPEN
    j = mt.merge(o[cols], left_on=["date", "home_fd", "away_fd"],
                 right_on=["date", "home_team", "away_team"], how="inner")
    # drop fixtures where the score disagrees -> the join found the wrong match
    ok = (j.goals_home == j.fthg) & (j.goals_away == j.ftag)
    print(f"joined {len(j)}/{len(mt)} matches ({len(j)/len(mt):.1%}); "
          f"{(~ok).sum()} score mismatches dropped")
    j = j[ok].copy()
    for name, c in (("pinc", PIN_CLOSE), ("pino", PIN_OPEN)):
        j = add_devig_probs(j, c, name)

    print("\ninverting 1X2 -> Poisson rates ...")
    for name in ("pinc", "pino"):
        p = j[[f"{name}_shin_h", f"{name}_shin_d", f"{name}_shin_a"]].to_numpy()
        good = np.isfinite(p).all(1) & (p.sum(1) > 0.99) & (p.sum(1) < 1.01)
        lh = np.full(len(j), np.nan); la = np.full(len(j), np.nan)
        for i in np.where(good)[0]:
            lh[i], la[i] = invert(p[i])
        j[f"{name}_lh"], j[f"{name}_la"] = lh, la
        print(f"  {name}: {good.sum()} rows inverted, "
              f"mean lambda_home {np.nanmean(lh):.3f}, away {np.nanmean(la):.3f}")
    print(f"  (model's own means for reference: home 1.571, away 1.264)")

    # -> long per-team panel, so the measure is comparable with xG
    recs = []
    for _, r in j.iterrows():
        recs.append(dict(team=r.team, date=r.date.date(), season=r.season,
                         mkt_c=r.pinc_lh, mkt_o=r.pino_lh, goals=r.goals_home))
        recs.append(dict(team=r.opp_team, date=r.date.date(), season=r.season,
                         mkt_c=r.pinc_la, mkt_o=r.pino_la, goals=r.goals_away))
    s = pd.DataFrame(recs)

    # the team's own xG in that match, for the head-to-head
    g = df[["datetime", "team", "xG"]].copy()
    g["date"] = pd.to_datetime(g.datetime).dt.date
    s = s.merge(g.drop(columns="datetime"), on=["team", "date"], how="inner")
    assert len(s) > 0.9 * 2 * len(j), f"xG join dropped too much: {len(s)}"
    s = s.sort_values("date")
    print(f"\nper-team panel: {len(s)} rows")

    K = 10
    for c in ("xG", "mkt_c", "mkt_o", "goals"):
        s[f"r_{c}"] = (s.groupby("team")[c]
                        .transform(lambda x: x.shift(1).rolling(K, min_periods=5).mean()))
    d = s.dropna(subset=[f"r_{c}" for c in ("xG", "mkt_c", "mkt_o", "goals")]).copy()
    print(f"usable rows after rolling: {len(d)}")

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

    solo = loso_r2(["xG"])
    print("\n=== OOS R^2 predicting the team's NEXT-match goals (leave-one-season-out) ===")
    print(f"  {'specification':38s} {'OOS R^2':>10s} {'vs xG alone':>13s}")
    print(f"  {'xG only (the incumbent channel)':38s} {solo:+10.5f} {'—':>13s}")
    for lab, use in (("xG + goals   [CONTROL, real +0.0017]", ["xG", "goals"]),
                     ("market CLOSING only", ["mkt_c"]),
                     ("market OPENING only", ["mkt_o"]),
                     ("xG + market closing", ["xG", "mkt_c"]),
                     ("xG + market opening", ["xG", "mkt_o"]),
                     ("xG + goals + market closing", ["xG", "goals", "mkt_c"])):
        r = loso_r2(use)
        print(f"  {lab:38s} {r:+10.5f} {r - solo:+13.5f}")

    print("\n  BAR: raw shot count scored +0.00484 here and FAILED as a channel (WP019).")
    print("  The market must clear ~+0.005 by a distance to be worth building.")


if __name__ == "__main__":
    main()
