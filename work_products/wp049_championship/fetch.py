"""WP049 — Championship (and League One) results, to replace the external Elo dependency.

WHY. WP025's Elo anchor is worth -0.00021 overall but the gain is CONCENTRATED where our
latents are blind: -0.00049 on matches involving a PROMOTED team (CI excluding zero) vs
-0.00010 on established teams only, and -0.00065 in season rounds 1-5 vs -0.00014 later.
Our model starts a promoted club at a prior saying "average team" when WP007 measured
them ~0.37 below average in attack. Clubelo knows better only because it has watched
them play a whole Championship season.

So the information is Championship form -- and we can compute that OURSELVES instead of
importing a rating. That removes the external dependency the user declined Elo for
(api.clubelo.com is 502/proxy-blocked and the main site is a JS app with no historical
data in the HTML).

SOURCE. openfootball/england on GitHub: results with dates for EVERY season 2000-01
onward, Championship and League One. Reachable (football-data.co.uk is 503-blocked right
now and intermittent generally). No xG, no shots, no odds -- but an Elo has none of those
either, and the Elo result proves goals-only Championship information is worth ~-0.0005
on promoted matches.

NOTE: files include play-off Finals as well as the regular season.
"""
import os
import re
from pathlib import Path

import pandas as pd

B = str((Path(__file__).resolve().parents[2] / ".certs/bundle.pem"))
os.environ["SSL_CERT_FILE"] = B
os.environ["REQUESTS_CA_BUNDLE"] = B
import requests  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/openfootball/england/master"
SEASONS = [f"{y}-{(y+1) % 100:02d}" for y in range(2013, 2025)]
DIVS = {"2-championship.txt": "Championship", "3-league1.txt": "League One"}
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

# openfootball uses TWO formats across seasons and both must be handled:
#   up to 2023-24 : "Fri Aug 4"            then  "  20:00  Home FC   1-2 (0-1)  Away FC"
#   from 2024-25  : "  Fri Aug 9 2024"     then  "  20:00  Home FC  v Away FC   4-2 (1-0)"
# The year is implicit in the old format (season spans two calendar years) and explicit
# in the new one. Parsing only the old shape silently yielded ZERO matches for 2024-25.
DATE_RE = re.compile(r"^\s*[A-Z][a-z]{2}\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
# A: home, score, away
MATCH_A = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?(.+?)\s{2,}(\d+)-(\d+)(?:\s*\(\d+-\d+\))?\s{2,}(.+?)\s*$")
# B: home, "v", away, score
MATCH_B = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?(.+?)\s+v\s+(.+?)\s{2,}(\d+)-(\d+)(?:\s*\(\d+-\d+\))?\s*$")

def parse(text, season):
    y0 = int(season[:4])
    rows, cur = [], None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("=", "#", "▪", "»")):
            continue
        d = DATE_RE.match(line)
        if d:
            mon, day, yr = MONTHS[d.group(1)], int(d.group(2)), d.group(3)
            # season spans two calendar years: Jul-Dec = y0, Jan-Jun = y0+1
            year = int(yr) if yr else (y0 if mon >= 7 else y0 + 1)
            cur = pd.Timestamp(year=year, month=mon, day=day)
            continue
        if cur is None:
            continue
        b = MATCH_B.match(line)
        if b:
            rows.append((cur, b.group(1).strip(), int(b.group(3)),
                         int(b.group(4)), b.group(2).strip()))
            continue
        a = MATCH_A.match(line)
        if a:
            rows.append((cur, a.group(1).strip(), int(a.group(2)),
                         int(a.group(3)), a.group(4).strip()))
    return pd.DataFrame(rows, columns=["date", "home", "hg", "ag", "away"]).assign(season=season)


def main():
    out = []
    for s in SEASONS:
        for fn, div in DIVS.items():
            r = requests.get(f"{RAW}/{s}/{fn}", timeout=35)
            if r.status_code != 200:
                print(f"  {s} {div}: HTTP {r.status_code}")
                continue
            d = parse(r.text, s).assign(division=div)
            if not len(d):
                print(f"  {s} {div}: PARSED ZERO MATCHES — format change?"); continue
            out.append(d)
            print(f"  {s} {div:13s} {len(d):4d} matches, {d.home.nunique():2d} teams, "
                  f"{d.date.min().date()} .. {d.date.max().date()}")
    df = pd.concat(out, ignore_index=True)
    # sanity: a league season should be ~552 (24 teams) regular + playoffs
    bad = df.groupby(["season", "division"]).size()
    print("\nmatches parsed per season/division:")
    print(bad.to_string())
    thin = bad[bad < 400]
    assert thin.empty, f"a season parsed too few matches:\n{thin}"
    assert df.hg.between(0, 15).all() and df.ag.between(0, 15).all(), "implausible score"
    df.to_pickle(HERE / "lower_divisions.pkl")
    print(f"\ntotal {len(df)} matches across {df.season.nunique()} seasons, "
          f"{df.division.nunique()} divisions -> lower_divisions.pkl")


if __name__ == "__main__":
    main()
