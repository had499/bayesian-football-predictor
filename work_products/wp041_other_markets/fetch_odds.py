"""WP041 — fetch OVER/UNDER 2.5 and ASIAN HANDICAP odds.

WP040 established there is no edge against the 1X2 OPENING price, well-powered. But the
model produces a full SCORELINE GRID and every test so far has scored only its 1X2
collapse, which throws the total-goals information away. The xG channel -- the project's
one established win -- informs goal RATES directly, which is exactly what a totals market
prices. So totals is the market where the model's strongest component is most directly
expressed and least tested.

Same GitHub mirror WP008 used (football-data.co.uk itself is proxy-blocked here, the
mirror is not). The CA bundle must be pointed at certifi or TLS fails on this machine.
"""
import os
from pathlib import Path

import certifi
import pandas as pd

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

HERE = Path(__file__).resolve().parent
MIRROR = "https://raw.githubusercontent.com/AnishKhetani/premier-league-data/main/data/processed"
OUT = HERE / "odds_other.pkl"

# Totals: the line is a fixed 2.5, so no line column is needed and the market is
# unambiguous. Opening AND closing for each book, so the open/close asymmetry WP010
# measured on 1X2 can be checked here too.
TOTALS = [f"{b}_{s}{suf}" for b in ("pinnacle", "market_avg", "market_max")
          for s in ("over25", "under25") for suf in ("", "_close")]
# Asian handicap: needs the LINE, which only some books publish. Collected so the
# handicap test can run where a line exists.
AH = [f"{b}_ah_{s}{suf}" for b in ("pinnacle", "market_avg", "market_max")
      for s in ("home", "away") for suf in ("", "_close")]
# The GENERIC line columns are the ones that actually cover 2019+ at 100%. The
# book-specific ones (ah_line_betbrain, bet365_ah_line, ...) are all 0% from 2019 --
# an easy miss that makes the AH prices useless, since a handicap price without its
# line means nothing.
LINES = ["ah_line", "ah_line_close",
         "ah_line_betbrain", "bet365_ah_line", "gamebookers_ah_line", "ladbrokes_ah_line"]


def main():
    res = pd.read_csv(f"{MIRROR}/results.csv", low_memory=False)
    odds = pd.read_csv(f"{MIRROR}/results_with_odds.csv", low_memory=False)
    want = [c for c in TOTALS + AH + LINES if c in odds.columns]
    missing = [c for c in TOTALS if c not in odds.columns]
    if missing:
        raise ValueError(f"mirror schema changed — missing totals columns {missing}")
    keep = ["match_id", "date", "home_team", "away_team", "fthg", "ftag"]
    df = res[[c for c in keep if c in res.columns]].merge(
        odds[["match_id", *want]], on="match_id", how="left")
    df["date"] = pd.to_datetime(df.date)
    print(f"{len(df)} matches, {len(want)} odds columns")
    for c in ("pinnacle_over25", "pinnacle_over25_close",
              "pinnacle_ah_home", "ah_line_betbrain"):
        if c in df.columns:
            print(f"  {c:28s} non-null {df[c].notna().mean():.1%}")
    recent = df[df.date >= "2014-08-01"]
    print(f"\nsince 2014-08 ({len(recent)} matches):")
    for c in ("pinnacle_over25", "pinnacle_over25_close", "pinnacle_ah_home"):
        if c in recent.columns:
            print(f"  {c:28s} non-null {recent[c].notna().mean():.1%}")
    OUT.write_bytes(pd.to_pickle(df, OUT) or b"") if False else df.to_pickle(OUT)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
