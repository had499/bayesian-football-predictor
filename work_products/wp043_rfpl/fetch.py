"""WP043 — Russian Premier League: fetch Understat match data and odds.

WHY RFPL. WP040/WP041/WP042 closed EPL betting on measurement: no information beyond the
opening price, none on totals, and NEGATIVE edge via winner's curse. The model itself is
decent (0.19692 RPS, ~0.003 behind the sharpest line in world football) -- the problem is
the VENUE, not the model. RFPL is the one league where Understat provides the xG data our
only working channel depends on, while market attention is far lower than the EPL's.

ODDS LIMITATION, stated up front: football-data.co.uk's "extra leagues" files carry
CLOSING odds only (PSCH/PSCD/PSCA + Max/Avg), no openers. So WP042's CLV test cannot run
here. What CAN run is the blend test against the CLOSING price -- a HARDER benchmark than
the opening price the EPL model already failed against, so a positive result would be
strong.

(football-data.co.uk was recorded as proxy-blocked; it responded fine on 2026-10-05.)
"""
import os
from pathlib import Path

import certifi
import pandas as pd

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

HERE = Path(__file__).resolve().parent
ODDS_URL = "https://www.football-data.co.uk/new/RUS.csv"


def fetch_odds():
    d = pd.read_csv(ODDS_URL)
    d = d[d.League.str.contains("Premier", na=False)].copy()
    d["Date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce")
    print(f"RFPL odds: {len(d)} matches, {d.Date.min().date()} to {d.Date.max().date()}")
    print(f"  columns: {[c for c in d.columns if c not in ('Country','League')]}")
    for c in ("PSCH", "MaxCH", "AvgCH"):
        if c in d.columns:
            print(f"  {c:6s} non-null {d[c].notna().mean():.1%}")
    d.to_pickle(HERE / "odds_rfpl.pkl")
    return d


def fetch_understat():
    import sys
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from football_model.data.get_data import get_understat_data
    years = [str(y) for y in range(2014, 2026)]
    df = get_understat_data(years=years, leagues=["RFPL"])
    print(f"\nUnderstat RFPL: {len(df)} rows")
    if "season" in df.columns:
        print(df.groupby("season").size().to_string())
    df.to_pickle(HERE / "understat_rfpl.pkl")
    return df


if __name__ == "__main__":
    o = fetch_odds()
    try:
        u = fetch_understat()
    except Exception as e:
        print(f"\nUnderstat fetch FAILED: {type(e).__name__}: {str(e)[:200]}")
