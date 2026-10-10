"""WP044 — rank the Understat leagues by how SOFT their market is, before modelling.

WP043 (clean) showed the model is +0.00336 behind RFPL's CLOSING line vs +0.0139 behind
the EPL's -- a 4x smaller deficit -- so the venue thesis has real support but RFPL only
publishes CLOSING odds, which is the hardest price and the reason no edge appeared.

football-data's MAIN-league files (F1/I1/SP1/D1/E0) carry Pinnacle OPENING *and* CLOSING
1X2 plus opening/closing over-2.5. So the four non-EPL Understat leagues are fully
testable, including WP042's CLV test (11-17x the power of ROI).

This scan uses ODDS ONLY -- no model, no compute -- to answer two questions that decide
where to point the model:

  1. MARKET SKILL over each league's own base rates. Lower = softer line = more room.
     (RFPL closing scored +0.02954 vs the EPL's +0.04077, i.e. 28% less skilful.)
  2. The OPEN -> CLOSE improvement. A large spread means the opening price is soft
     relative to the close, which is exactly the window a pre-match model can exploit.
     On the EPL that spread was worth 0.0107 RPS.
"""
import io
import os
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

os.environ["SSL_CERT_FILE"] = certifi.where()
CTX = ssl.create_default_context(cafile=certifi.where())
HERE = Path(__file__).resolve().parent
LEAGUES = {"EPL": "E0", "Ligue_1": "F1", "Serie_A": "I1",
           "La_Liga": "SP1", "Bundesliga": "D1"}
SEASONS = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2014, 2025)]
OPEN = ["PSH", "PSD", "PSA"]
CLOSE = ["PSCH", "PSCD", "PSCA"]


def devig(o):
    p = 1.0 / o
    return p / p.sum(1, keepdims=True)


def rps3(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return ((p.cumsum(1) - o.cumsum(1))[:, :2] ** 2).sum(1)


def fetch(code):
    frames = []
    for s in SEASONS:
        try:
            u = f"https://www.football-data.co.uk/mmz4281/{s}/{code}.csv"
            raw = urllib.request.urlopen(u, timeout=30, context=CTX).read()
            d = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
            d["_season"] = s
            frames.append(d)
        except Exception as e:
            print(f"    {code} {s}: {type(e).__name__}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    out = {}
    cache = HERE / "odds_all_leagues.pkl"
    store = {}
    for name, code in LEAGUES.items():
        d = fetch(code)
        if not len(d):
            continue
        d = d.dropna(subset=["FTHG", "FTAG"])
        store[name] = d
        y = np.where(d.FTHG > d.FTAG, 0, np.where(d.FTHG == d.FTAG, 1, 2))
        base = np.array([(y == k).mean() for k in range(3)])
        nb = rps3(np.tile(base, (len(y), 1)), y).mean() / 2
        row = {"n": len(d), "base_rps": nb,
               "home_rate": base[0], "draw_rate": base[1]}
        for lab, cols in (("open", OPEN), ("close", CLOSE)):
            if not all(c in d.columns for c in cols):
                continue
            k = d[cols].notna().all(axis=1)
            if k.sum() < 500:
                continue
            p = devig(d.loc[k, cols].to_numpy())
            r = rps3(p, y[k.to_numpy()]).mean() / 2
            nb_k = rps3(np.tile(base, (k.sum(), 1)), y[k.to_numpy()]).mean() / 2
            row[f"{lab}_n"] = int(k.sum())
            row[f"{lab}_rps"] = r
            row[f"{lab}_skill"] = nb_k - r
            row[f"{lab}_overround"] = (1 / d.loc[k, cols].to_numpy()).sum(1).mean() - 1
        out[name] = row
    pd.to_pickle(store, cache)

    t = pd.DataFrame(out).T
    print("\n=== market quality by league (odds only, no model) ===")
    cols = ["n", "open_n", "close_n", "base_rps", "open_rps", "close_rps",
            "open_skill", "close_skill", "open_overround", "close_overround"]
    print(t[[c for c in cols if c in t.columns]].astype(float).round(5).to_string())

    print("\n=== ranked by CLOSING-line skill (lower = softer market) ===")
    s = t.sort_values("close_skill")
    for name, r in s.iterrows():
        print(f"  {name:12s} close_skill {r.close_skill:+.5f}   open_skill "
              f"{r.open_skill:+.5f}   open->close gain {r.open_rps - r.close_rps:+.5f}"
              f"   overround open {r.open_overround:.2%}")
    print("\n  EPL is the reference: close_skill +0.04077 measured in WP043, and the")
    print("  model sits +0.00315 behind its OPEN. A league with materially lower")
    print("  close_skill AND a comparable open->close gap is the place to model next.")


if __name__ == "__main__":
    main()
