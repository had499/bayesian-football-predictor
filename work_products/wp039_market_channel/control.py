"""WP039 negative control — is the market channel's gain INFORMATION, or structure?

The smoke fit moved window 70 from 0.18470 to 0.17454, about -0.010: three times the
entire remaining gap to Pinnacle's opening line. An effect that large on a channel this
simple deserves a control before a 3-hour sweep.

CONTROL: keep everything identical but SHUFFLE the market-implied rates across training
matches (within season, so the marginal distribution and any season-level drift are
preserved). The shuffled rates carry zero information about which match they belong to.

  - if the gain VANISHES, the information is real and match-specific.
  - if the gain SURVIVES, the channel is helping for a structural reason -- e.g. it is
    acting as a regulariser or shifting the overall level through kappa -- and the
    "market information" story is wrong.

A second arm uses the market's OPENING line instead of its closing line. Opening prices
are strictly less informed, so a real information effect should be slightly SMALLER
there; if opening and closing score the same, the channel is not exploiting the
closing-line sharpness that WP010 measured.
"""
import pickle
import subprocess
import sys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
WP = HERE.parent
REPO = HERE.parents[1]
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from market_analysis import CODE_TO_FD, add_devig_probs  # noqa: E402
from arms import SAMPLER, sampler_args  # noqa: E402

PIN_OPEN = ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"]
MAXG = 10
CHAMPION = {"use_xg_channel": True, "xg_channel_weight": 0.75,
            "use_deep_channel": True, "deep_channel_weight": 0.5,
            "use_stakes_effort": True, "stakes_effort_sd": 0.3}


def poisson_probs(lh, la):
    g = np.outer(poisson.pmf(np.arange(MAXG + 1), lh),
                 poisson.pmf(np.arange(MAXG + 1), la))
    return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])


def invert(p):
    r = least_squares(lambda x: poisson_probs(np.exp(x[0]), np.exp(x[1])) - p,
                      x0=np.log([1.5, 1.2]), method="lm")
    return np.exp(r.x[0]), np.exp(r.x[1])


def build_shuffled():
    d = pickle.loads((HERE / "cv_shared_data_market.pkl").read_bytes())
    df = d["df_cv"].copy()
    rng = np.random.default_rng(7)
    # shuffle the (lh, la) PAIRS across matches, within season, keeping weights put
    for s in df.season.unique():
        k = (df.season == s) & (df.market_obs_w > 0)
        idx = np.where(k)[0]
        perm = rng.permutation(idx)
        df.loc[df.index[idx], "market_lh"] = df.market_lh.to_numpy()[perm]
        df.loc[df.index[idx], "market_la"] = df.market_la.to_numpy()[perm]
    d["df_cv"] = df
    out = HERE / "cv_shared_data_market_shuffled.pkl"
    out.write_bytes(pickle.dumps(d))
    print(f"shuffled file written; mean lh {df.market_lh[df.market_obs_w>0].mean():.3f} "
          f"(unchanged by construction)")
    return out


def build_opening():
    d = pickle.loads((HERE / "cv_shared_data_market.pkl").read_bytes())
    df = d["df_cv"].copy()
    odds = pd.read_pickle(WP / "wp008_market_benchmark" / "odds_raw.pkl")
    mt = df[df.is_home == 1][["datetime", "team", "opp_team",
                              "goals_home", "goals_away"]].copy()
    mt["date"] = pd.to_datetime(mt.datetime).dt.normalize()
    mt["home_fd"] = mt.team.map(CODE_TO_FD); mt["away_fd"] = mt.opp_team.map(CODE_TO_FD)
    o = odds.copy(); o["date"] = pd.to_datetime(o.date).dt.normalize()
    j = mt.merge(o[["date", "home_team", "away_team", "fthg", "ftag"] + PIN_OPEN],
                 left_on=["date", "home_fd", "away_fd"],
                 right_on=["date", "home_team", "away_team"], how="left")
    bad = j.fthg.notna() & ((j.goals_home != j.fthg) | (j.goals_away != j.ftag))
    j.loc[bad, PIN_OPEN] = np.nan
    j = add_devig_probs(j, PIN_OPEN, "pino")
    P = j[["pino_shin_h", "pino_shin_d", "pino_shin_a"]].to_numpy()
    good = np.isfinite(P).all(1) & (P.sum(1) > 0.99) & (P.sum(1) < 1.01)
    lh = np.zeros(len(j)); la = np.zeros(len(j)); w = np.zeros(len(j))
    for i in np.where(good)[0]:
        lh[i], la[i] = invert(P[i]); w[i] = 1.0
    w = w * (lh > 0.2) * (lh < 5.0) * (la > 0.2) * (la < 5.0)
    lh = np.where(w > 0, lh, 1.0); la = np.where(w > 0, la, 1.0)
    mt2 = j[["team", "date"]].copy()
    mt2["market_lh"], mt2["market_la"], mt2["market_obs_w"] = lh, la, w
    df["date"] = pd.to_datetime(df.datetime).dt.normalize()
    key = mt2.set_index(["team", "date"])
    idx = pd.MultiIndex.from_arrays([df.team.where(df.is_home == 1, df.opp_team), df.date])
    got = key.reindex(idx)
    for c in ("market_lh", "market_la", "market_obs_w"):
        df[c] = got[c].to_numpy()
    df[["market_lh", "market_la"]] = df[["market_lh", "market_la"]].fillna(1.0)
    df["market_obs_w"] = df.market_obs_w.fillna(0.0)
    df = df.drop(columns="date")
    d["df_cv"] = df
    out = HERE / "cv_shared_data_market_opening.pkl"
    out.write_bytes(pickle.dumps(d))
    print(f"opening-line file written; coverage {df.market_obs_w.mean():.1%}")
    return out


def run(tag, data, cfg, window=70):
    ck = HERE / "control" / tag / f"w{window:03d}.pkl"
    if not ck.exists():
        ck.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(REPO / "scripts" / "run_cv_window.py"),
               "--data-path", str(data), "--checkpoint-path", str(ck),
               "--window-index", str(window), "--use-xg", "false", "--use-dc", "false",
               "--config-json", json.dumps(cfg), "--predict-mode", "plugin",
               *sampler_args()]
        env = dict(os.environ); env["PYTENSOR_FLAGS"] = "cxx="
        env.setdefault("XLA_FLAGS",
                       f"--xla_force_host_platform_device_count={SAMPLER['chains']}")
        with open(ck.parent / f"w{window:03d}.log", "w") as lf:
            subprocess.run(cmd, check=True, env=env, stdout=lf, stderr=subprocess.STDOUT)
    r = pickle.loads(ck.read_bytes())["results"]
    r = r[0] if isinstance(r, list) else r
    return r


def main():
    shuf = build_shuffled()
    opn = build_opening()
    real = HERE / "cv_shared_data_market.pkl"
    mk = {**CHAMPION, "use_market_channel": True, "market_channel_weight": 0.5}

    champ = pickle.loads((WP / "wp032_xg_weight/checkpoints/W075/w070.pkl").read_bytes())
    champ = champ["results"]; champ = champ[0] if isinstance(champ, list) else champ
    print(f"\nchampion window 70 RPS {champ['rps']:.5f}\n")
    print(f"  {'arm':22s} {'RPS':>9s} {'dRPS':>10s} {'mkt_kappa':>10s} {'mkt_shape':>10s}")
    for tag, data in (("real closing", real), ("SHUFFLED (control)", shuf),
                      ("opening line", opn)):
        r = run(tag.split()[0].lower().replace("(", ""), data, mk)
        pm_ = r["param_means"]
        print(f"  {tag:22s} {r['rps']:9.5f} {r['rps']-champ['rps']:+10.5f} "
              f"{pm_.get('mkt_kappa', float('nan')):10.3f} "
              f"{pm_.get('mkt_shape', float('nan')):10.2f}")
    print("\n  if SHUFFLED keeps most of the gain, the story is wrong and this is")
    print("  structure, not information. If it loses it, the effect is match-specific.")


if __name__ == "__main__":
    main()
