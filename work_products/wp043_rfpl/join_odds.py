"""WP043 — map Understat RFPL team names to football-data's and validate the join.

Only 12 of 32 names match exactly ('Dinamo Moscow' vs 'Dynamo Moscow', 'FK Akhmat' vs
'Akhmat Grozny', 'Anzhi Makhachkala' vs 'FK Anzi Makhackala', ...). Rather than hand-code
a map and hope, this fuzzy-matches and then VALIDATES against the recorded SCORES: if a
name mapped to the wrong club, the goals will not agree and the row is dropped. That is
the same guard WP008's EPL join uses, and it is what turns a guessed mapping into a
checkable one.
"""
import difflib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "odds_joined_rfpl.pkl"


def main():
    d = pickle.loads((HERE / "cv_shared_data_rfpl.pkl").read_bytes())["df_cv"]
    o = pd.read_pickle(HERE / "odds_rfpl.pkl").copy()
    o["Date"] = pd.to_datetime(o.Date, dayfirst=True, errors="coerce").dt.normalize()

    u_names = sorted(d.team_long.unique())
    o_names = sorted(set(o.Home.dropna()) | set(o.Away.dropna()))

    # EXPLICIT map, written by hand after fuzzy matching failed badly. difflib mapped
    # 'FC Yenisey Krasnoyarsk' AND 'Kuban Krasnodar' both onto 'Krasnodar' (three
    # different clubs), and left 'Zenit St. Petersburg' -- the league's dominant side,
    # in ~1/8 of all matches -- unmapped entirely. The score validator did not catch
    # those because a wrong name matches NO date rather than the wrong fixture, so it
    # showed 0 mismatches while silently dropping 25% of the data. Fuzzy matching on
    # club names is not safe; this is the kind of thing that has to be read.
    MANUAL = {
        "Zenit St. Petersburg": "Zenit",
        "Mordovya": "M. Saransk",            # Mordovia Saransk
        "Kuban Krasnodar": "Kuban",          # NOT Krasnodar - different club
        "FC Yenisey Krasnoyarsk": "Yenisey", # NOT Krasnodar - different club
        "FC Krasnodar": "Krasnodar",
        "Nizhny Novgorod": "Pari NN",        # the modern club, NOT Volga N. Novgorod
        "Dinamo Moscow": "Dynamo Moscow",
        "FK Akhmat": "Akhmat Grozny",
        "Anzhi Makhachkala": "FK Anzi Makhackala",
        "FC Rostov": "FK Rostov",
        "FC Rotor Volgograd": "R. Volgograd",
        "Fakel": "Fakel Voronezh",
        "Krylya Sovetov Samara": "Krylya Sovetov",
        "PFC Sochi": "Sochi",
        "SKA-Khabarovsk": "SKA Khabarovsk",
        "Tom Tomsk": "Tomsk",
        "FC Tambov": "Tambov",
        "FC Ufa": "Ufa",
        "FC Orenburg": "Orenburg",
        "Akron": "Akron Togliatti",
        "Alania Vladikavkaz": "Vladikavkaz",
        "Terek Grozny": "Akhmat Grozny",     # renamed in 2017
        "Torpedo Moscow": "Torpedo Moscow",
        "Dynamo Makhachkala": "Dynamo Makhachkala",
    }
    mapping = {}
    for u in u_names:
        if u in MANUAL:
            mapping[u] = MANUAL[u]
        elif u in o_names:
            mapping[u] = u
        else:
            mapping[u] = None
    bad = [v for v in mapping.values() if v is not None and v not in o_names]
    assert not bad, f"MANUAL map points at names the odds file does not have: {bad}"
    unmapped = [u for u, v in mapping.items() if v is None]
    print(f"mapped {sum(v is not None for v in mapping.values())}/{len(u_names)} names")
    for u in u_names:
        if mapping[u] != u:
            print(f"    {u:28s} -> {mapping[u]}")
    if unmapped:
        print(f"  UNMAPPED (their matches will be dropped): {unmapped}")

    m = d[d.is_home == 1][["datetime", "team", "opp_team", "team_long", "opp_team_long",
                           "goals_home", "goals_away", "season", "round",
                           "cum_round"]].copy()
    m["Date"] = pd.to_datetime(m.datetime).dt.normalize()
    m["h_fd"] = m.team_long.map(mapping)
    m["a_fd"] = m.opp_team_long.map(mapping)

    j = m.merge(o[["Date", "Home", "Away", "HG", "AG", "PSCH", "PSCD", "PSCA",
                   "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA"]],
                left_on=["Date", "h_fd", "a_fd"],
                right_on=["Date", "Home", "Away"], how="left")
    found = j.Home.notna()
    print(f"\njoined on (date, home, away): {found.sum()}/{len(j)} ({found.mean():.1%})")

    # THE VALIDATOR: a wrong name mapping shows up as disagreeing scores
    agree = (j.goals_home == j.HG) & (j.goals_away == j.AG)
    bad = found & ~agree
    print(f"  score MISMATCHES (wrong fixture / bad name map): {bad.sum()}")
    if bad.sum():
        print(j[bad][["Date", "team_long", "opp_team_long", "Home", "Away",
                      "goals_home", "HG", "goals_away", "AG"]].head(8).to_string(index=False))
    j = j[found & agree].copy()
    print(f"  usable after validation: {len(j)} matches")
    print(f"  with Pinnacle closing:   {j[['PSCH','PSCD','PSCA']].notna().all(axis=1).sum()}")
    print(f"  by season:")
    print(j.groupby("season").size().to_string())
    j.to_pickle(OUT)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
