"""WP045 — Ligue 1: is the market's edge over us CONSTANT across leagues?

THE QUESTION. WP043 (clean) and the EPL give two data points where the market beats us by
a near-identical margin despite very different market quality:

    league   market close skill   our skill   market ahead by
    EPL          +0.03877          +0.03576      0.00301
    RFPL         +0.02954          +0.02669      0.00285

If that gap is structural, a softer venue buys nothing -- low market skill just means the
league is less predictable for everyone. Ligue 1 has the SOFTEST closing line of the five
Understat leagues (+0.02920 vs the EPL's +0.03877, WP044's scan), so it is the decisive
third point.

Ligue 1 also has Pinnacle OPENING odds (football-data main-league file), so unlike RFPL
this can be scored against the price a bettor can actually take, and WP042's CLV test
becomes available.
"""
import io
import os
import pickle
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import load_checkpoints, paired_diff, verdict, wide  # noqa: E402

B = str((WP.parent / ".certs/bundle.pem").resolve())
os.environ["SSL_CERT_FILE"] = B; os.environ["REQUESTS_CA_BUNDLE"] = B
CTX = ssl.create_default_context(cafile=B)
EPS = 1e-9
OPEN, CLOSE = ["PSH", "PSD", "PSA"], ["PSCH", "PSCD", "PSCA"]


def devig(o):
    p = 1.0 / o; return p / p.sum(1, keepdims=True)


def rps3(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return ((p.cumsum(1) - o.cumsum(1))[:, :2] ** 2).sum(1) / 2


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean()


def blend(pm_, pk, w):
    z = np.exp((1 - w) * np.log(pk + EPS) + w * np.log(pm_ + EPS))
    return z / z.sum(1, keepdims=True)


def fd_odds():
    """Reuse WP044's cache. Re-fetching is unreliable: football-data returned 200 for
    every F1 season during WP044's scan and HTTPError for all of them an hour later
    (the proxy is intermittent -- the same reason the Understat fetch needed a keychain
    CA bundle). Never re-fetch what is already on disk."""
    cache = HERE / "odds_ligue1.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    store = pd.read_pickle(WP / "wp044_league_scan" / "odds_all_leagues.pkl")
    d = store["Ligue_1"].copy()
    d["Date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce").dt.normalize()
    d.to_pickle(cache)
    print(f"  odds from WP044 cache: {len(d)} rows")
    return d


def main():
    shared = pickle.loads((HERE / "cv_shared_data_ligue1.pkl").read_bytes())
    df = shared["df_cv"]
    m, _ = load_checkpoints(HERE / "checkpoints")
    full = m.groupby("arm").window.nunique()
    arms = sorted(full[full == full.max()].index)
    com = set.intersection(*[set(m[m.arm == a].window) for a in arms])
    mm = m[m.window.isin(com)]
    print(f"\n=== 1. xG channel in Ligue 1 ({len(com)} windows) ===")
    R = wide(mm[mm.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window").to_numpy()
    for a in arms:
        print(f"  {a:6s} RPS {R[a].mean():.5f}")
    if {"L0", "XG75"} <= set(arms):
        d, lo, hi = paired_diff(R["XG75"], R["L0"], groups=g)
        print(f"  xG channel: {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
        print(f"  [EPL -0.00258 | RFPL -0.00177]")

    # --- odds join, fixture-keyed, score-validated ---
    o = fd_odds()
    pred = mm[mm.arm == "XG75"].copy()
    key = (df[df.is_home == 1][["cum_round", "team", "opp_team", "datetime"]]
           .rename(columns={"team": "h", "opp_team": "a"}))
    key["d"] = pd.to_datetime(key.datetime).dt.normalize()
    pred = pred.merge(key, left_on=["round", "home", "away"],
                      right_on=["cum_round", "h", "a"], how="inner")
    # football-data uses its own club names; map by fuzzy+validate on score
    import difflib
    fd_names = sorted(set(o.HomeTeam.dropna()) | set(o.AwayTeam.dropna()))
    us = sorted(set(pred.h) | set(pred.a))
    mp = {}
    for u in us:
        if u in fd_names:
            mp[u] = u; continue
        c = difflib.get_close_matches(u, fd_names, n=1, cutoff=0.55)
        mp[u] = c[0] if c else None
    miss = [u for u, v in mp.items() if v is None]
    pred["h_fd"] = pred.h.map(mp); pred["a_fd"] = pred.a.map(mp)
    j = pred.merge(o[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *OPEN, *CLOSE]],
                   left_on=["d", "h_fd", "a_fd"],
                   right_on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    ok = (j.goals_home == j.FTHG) & (j.goals_away == j.FTAG)
    print(f"\n  odds join: {len(j)}/{len(pred)} matched, score agreement "
          f"{ok.sum()}/{len(j)}; unmapped names: {miss if miss else 'none'}")
    j = j[ok].dropna(subset=OPEN + CLOSE).copy()

    y = np.where(j.goals_home > j.goals_away, 0,
                 np.where(j.goals_home == j.goals_away, 1, 2))
    pm_ = j[["p_home", "p_draw", "p_away"]].to_numpy()
    base = np.array([(y == k).mean() for k in range(3)])
    nb = rps3(np.tile(base, (len(y), 1)), y).mean()
    print(f"\n=== 2. model vs the market ({len(j)} matches) ===")
    print(f"  base-rate RPS {nb:.5f}")
    print(f"  model  RPS {rps3(pm_, y).mean():.5f}   skill {nb - rps3(pm_, y).mean():+.5f}")
    for lab, cols in (("OPEN", OPEN), ("CLOSE", CLOSE)):
        pk = devig(j[cols].to_numpy())
        rk = rps3(pk, y).mean()
        d, lo, hi = paired_diff(pd.Series(rps3(pm_, y)), pd.Series(rps3(pk, y)),
                                groups=j.window.to_numpy())
        print(f"  market {lab:5s} RPS {rk:.5f}   skill {nb - rk:+.5f}"
              f"   gap {d:+.5f} [{lo:+.5f},{hi:+.5f}]"
              f"   {'MODEL AHEAD' if d < 0 else 'market ahead'}")

    print(f"\n=== 3. IS THE GAP CONSTANT ACROSS LEAGUES? ===")
    pk = devig(j[CLOSE].to_numpy())
    ours, theirs = nb - rps3(pm_, y).mean(), nb - rps3(pk, y).mean()
    print(f"  {'league':10s} {'market skill':>13s} {'our skill':>11s} {'market ahead':>13s}")
    print(f"  {'EPL':10s} {0.03877:+13.5f} {0.03576:+11.5f} {0.00301:+13.5f}")
    print(f"  {'RFPL':10s} {0.02954:+13.5f} {0.02669:+11.5f} {0.00285:+13.5f}")
    print(f"  {'Ligue_1':10s} {theirs:+13.5f} {ours:+11.5f} {theirs - ours:+13.5f}")
    print("\n  a third value near 0.0029 means the market's edge is STRUCTURAL and no")
    print("  venue change fixes it. Materially smaller means the thesis lives.")

    print(f"\n=== 4. does the model add to the OPENING price? (blend, LOSO) ===")
    po = devig(j[OPEN].to_numpy())
    j = j.reset_index(drop=True)
    seas = pd.to_datetime(j.d).dt.year - (pd.to_datetime(j.d).dt.month < 8)
    ws, lk, lb = [], [], []
    for s in sorted(seas.unique()):
        tr, te = (seas != s).to_numpy(), (seas == s).to_numpy()
        if tr.sum() < 300 or te.sum() < 80:
            continue
        r = minimize_scalar(lambda w: logloss(blend(pm_[tr], po[tr], w), y[tr]),
                            bounds=(0, 1), method="bounded")
        ws.append(float(r.x))
        lk.append(logloss(po[te], y[te]))
        lb.append(logloss(blend(pm_[te], po[te], float(r.x)), y[te]))
    print(f"  w = {np.mean(ws):.3f} [{min(ws):.3f},{max(ws):.3f}] over {len(ws)} seasons")
    print(f"  logloss  market {np.mean(lk):.5f} -> blend {np.mean(lb):.5f} "
          f"({np.mean(lb)-np.mean(lk):+.5f})"
          f"  {'*** BEATS THE PRICE ***' if np.mean(lb) < np.mean(lk) else ''}")
    print(f"  [EPL w=0.046, blend +0.00027 WORSE]")


if __name__ == "__main__":
    main()
