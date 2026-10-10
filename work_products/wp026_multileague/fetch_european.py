"""WP026 step 2 — European fixtures from openfootball (the CROSS-LINKS).

These matches are the whole point. Without them the five leagues are disconnected
and their latents are only identified WITHIN league -- there is no common scale, and
an EPL prediction gains nothing from La Liga data. With them, Liverpool's latent is
informed by a Champions League result against a Real Madrid latent informed by La Liga.

SOURCE. Understat has no European competitions (404 on /league/CL) and FBref blocks
scrapers (403). openfootball/champions-league is reachable and carries cl.txt, el.txt
and conf.txt per season from 2011-12 to 2025-26, covering the whole CV grid. It is
RESULTS ONLY -- no xG for these matches -- so the goals likelihood carries them while
the xG channel applies to domestic matches only. That is a real downgrade for the
cross-links specifically, and it is the price of the only obtainable source.

USABILITY. A match is only usable if BOTH sides are in the five modelled leagues:
"Arsenal beat Porto" is uninterpretable without a Porto latent. Matches against
clubs outside the big five are dropped rather than given weakly-identified latents.

    python work_products/wp026_multileague/fetch_european.py
"""
import pickle
import re
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "european_raw.pkl"
REPO = "openfootball/champions-league"
SEASONS = [f"{y}-{str(y+1)[2:]}" for y in range(2014, 2026)]
FILES = {"cl.txt": "CL", "el.txt": "EL", "conf.txt": "UECL"}
BIG5 = {"ENG", "ESP", "GER", "ITA", "FRA"}

MONTHS = dict(Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6,
              Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)
DATE_RE = re.compile(r"^\s{1,4}\w{3}\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(
    r"(?:\d{1,2}:\d{2}\s+)?(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)\s+(\d+)-(\d+)")


def parse(text, comp, season):
    """-> list of dicts. Dates cascade: a bare 'Wed Sep 20' inherits the last year."""
    year = None
    out = []
    cur = None
    for line in text.split("\n"):
        d = DATE_RE.match(line)
        if d:
            mon, day, yr = d.group(1), int(d.group(2)), d.group(3)
            if yr:
                year = int(yr)
            if year and mon in MONTHS:
                cur = pd.Timestamp(year=year, month=MONTHS[mon], day=day)
            continue
        m = MATCH_RE.search(line)
        if m and cur is not None:
            h, hc, a, ac, hg, ag = m.groups()
            out.append(dict(comp=comp, season=season, date=cur,
                            home=h.strip(), home_cc=hc, away=a.strip(), away_cc=ac,
                            goals_home=int(hg), goals_away=int(ag)))
    return out


def main():
    S = requests.Session()
    S.headers["User-Agent"] = "Mozilla/5.0"
    rows = []
    for season in SEASONS:
        got = 0
        for fn, comp in FILES.items():
            u = f"https://raw.githubusercontent.com/{REPO}/master/{season}/{fn}"
            r = S.get(u, timeout=40)
            if r.status_code != 200:
                continue
            recs = parse(r.text, comp, season)
            rows += recs
            got += len(recs)
        print(f"  {season}: {got} matches", flush=True)
    d = pd.DataFrame(rows)
    d["both_big5"] = d.home_cc.isin(BIG5) & d.away_cc.isin(BIG5)
    OUT.write_bytes(pickle.dumps(d))
    print(f"\n{len(d)} European matches parsed, {d.comp.nunique()} competitions")
    print(f"  involving >=1 big-five club: {(d.home_cc.isin(BIG5) | d.away_cc.isin(BIG5)).sum()}")
    print(f"  BOTH sides big-five (usable cross-links): {int(d.both_big5.sum())}")
    print(f"\nper competition (usable only):")
    print(d[d.both_big5].groupby('comp').size().to_string())
    print(f"\nusable per season: {d[d.both_big5].groupby('season').size().to_dict()}")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
