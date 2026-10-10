"""WP027 step 1 — every competitive fixture for EPL clubs, all competitions.

WHY. WP009 tested rest days and found nothing, but it measured LEAGUE-ONLY rest: a
team playing Thursday in Europe then Sunday looked identically rested to one with a
clear week. Roughly 7-8 of 20 EPL clubs are in Europe, and all 20 enter both domestic
cups, so a large share of matches were mislabelled. The BACKLOG records that null as
"league-only rest-day differential doesn't help", NOT "congestion doesn't matter", and
flags cup/European fixture dates as the proper fix.

WHY THIS IS A GOOD FIT FOR THE AVAILABLE DATA. Rest days need only fixture DATES --
no xG, no results, no opponent latent. openfootball has no xG and covers clubs far
outside our model, but neither limitation matters here. This is the one open idea
whose data requirements the accessible sources actually meet.

WHY THE MODEL CANNOT ALREADY KNOW THIS. Cup and European fixtures are absent from
Understat entirely, so a midweek Europa League trip is invisible to the model. Unlike
most covariates tried here, this is not information the latent walk could have
absorbed from past league results -- it is not in the training data at all.

SOURCES
  domestic cups   openfootball/england/<season>/facup.txt, eflcup.txt
  European        openfootball/champions-league/<season>/{cl,el,conf}.txt
  league          Understat (already in the CV data)

    python work_products/wp027_congestion/fetch_fixtures.py
"""
import pickle
import re
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures_raw.pkl"
SEASONS = [f"{y}-{str(y+1)[2:]}" for y in range(2014, 2026)]
ENG_FILES = {"facup.txt": "FACup", "eflcup.txt": "EFLCup"}
EUR_FILES = {"cl.txt": "CL", "el.txt": "EL", "conf.txt": "UECL"}

MONTHS = dict(Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6,
              Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)
DATE_RE = re.compile(r"^\s{1,4}\w{3}\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
# cups have no country codes; Europe does. Accept either, and tolerate "pen." lines.
M_EUR = re.compile(r"^\s*(?:\d{1,2}:\d{2})?\s+(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)\s+\d+-\d+")
M_CUP = re.compile(r"^\s*(?:\d{1,2}:\d{2})?\s+(.+?)\s{2,}v\s+(.+?)\s{2,}\d+-\d+")
TIME_LEAK = re.compile(r"^\d{1,2}:\d{2}\s+")


def _clean(name):
    """Strip a leaked kickoff time. The optional time group in M_CUP/M_EUR can be
    skipped by the regex engine, letting the lazy name group absorb '12:30  '. That
    produces distinct 'names' per kickoff time, so a club silently fails to match and
    its fixtures vanish -- which corrupts rest days rather than merely omitting them."""
    return TIME_LEAK.sub("", name).strip()


def parse(text, comp, season, european):
    year, cur, out = None, None, []
    for line in text.split("\n"):
        d = DATE_RE.match(line)
        if d:
            mon, day, yr = d.group(1), int(d.group(2)), d.group(3)
            if yr:
                year = int(yr)
            if year and mon in MONTHS:
                try:
                    cur = pd.Timestamp(year=year, month=MONTHS[mon], day=day)
                except ValueError:
                    cur = None
            continue
        m = (M_EUR if european else M_CUP).search(line)
        if m and cur is not None:
            g = m.groups()
            h, a = (g[0], g[2]) if european else (g[0], g[1])
            hc, ac = (g[1], g[3]) if european else ("ENG", "ENG")
            h, a = _clean(h), _clean(a)
            out.append(dict(comp=comp, season=season, date=cur,
                            team=h, cc=hc, opp=a, opp_cc=ac))
            out.append(dict(comp=comp, season=season, date=cur,
                            team=a, cc=ac, opp=h, opp_cc=hc))
    return out


def main():
    S = requests.Session()
    S.headers["User-Agent"] = "Mozilla/5.0"
    rows = []
    for season in SEASONS:
        got = {}
        for repo, files, eur in (("openfootball/england", ENG_FILES, False),
                                 ("openfootball/champions-league", EUR_FILES, True)):
            for fn, comp in files.items():
                u = f"https://raw.githubusercontent.com/{repo}/master/{season}/{fn}"
                r = S.get(u, timeout=40)
                if r.status_code != 200:
                    continue
                recs = parse(r.text, comp, season, eur)
                rows += recs
                got[comp] = len(recs) // 2
        print(f"  {season}: {got}", flush=True)
    d = pd.DataFrame(rows)
    OUT.write_bytes(pickle.dumps(d))
    print(f"\n{len(d)} team-fixture rows ({len(d)//2} matches) across {d.comp.nunique()} competitions")
    print(d.groupby('comp').size().div(2).astype(int).to_string())
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
