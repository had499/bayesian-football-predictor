"""Analysis helpers for WP007 — kept out of the notebook so it stays a lean
readout, and so the statistics here are importable/testable.

Everything is PAIRED: arms are scored on exactly the same test matches, so a
comparison is a per-match difference in RPS, not a difference of two noisy
pooled means.
"""
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

WP_DIR = Path(__file__).resolve().parent
CKPT_DIR = WP_DIR / "checkpoints"

# Understat team code -> football-data.co.uk name (same map WP003 uses; covers
# every side in the 2020-21+ test period).
CODE_TO_FD = {
    "ARS": "Arsenal", "AVL": "Aston Villa", "BOU": "Bournemouth", "BRE": "Brentford",
    "BRI": "Brighton", "BUR": "Burnley", "CHE": "Chelsea", "CRY": "Crystal Palace",
    "EVE": "Everton", "FLH": "Fulham", "IPS": "Ipswich", "LED": "Leeds",
    "LEI": "Leicester", "LIV": "Liverpool", "LUT": "Luton", "MCI": "Man City",
    "MUN": "Man United", "NEW": "Newcastle", "NOR": "Norwich", "NOT": "Nott'm Forest",
    "SHE": "Sheffield United", "SOU": "Southampton", "SUN": "Sunderland", "TOT": "Tottenham",
    "WAT": "Watford", "WBA": "West Brom", "WHU": "West Ham", "WOL": "Wolves",
}


# ------------------------------------------------------------------ loading --
def load_checkpoints(ckpt_dir=CKPT_DIR):
    """-> (matches, windows): per-match and per-window DataFrames, all arms."""
    match_rows, window_rows = [], []
    for f in sorted(glob.glob(str(Path(ckpt_dir) / "*" / "w*.pkl"))):
        arm = Path(f).parent.name
        with open(f, "rb") as fh:
            ck = pickle.load(fh)
        for r in ck["results"]:
            window_rows.append({"arm": arm, **{k: v for k, v in r.items() if k != "param_means"},
                                **{f"param_{k}": v for k, v in (r.get("param_means") or {}).items()}})
        for m in ck["cv_match_predictions"]:
            match_rows.append({"arm": arm, **m})
    return pd.DataFrame(match_rows), pd.DataFrame(window_rows)


def complete_windows(matches, arms):
    """Windows that finished for EVERY arm in `arms` — the only ones a paired
    comparison may use (a partial sweep must not compare unequal match sets)."""
    have = matches.groupby("arm")["window"].agg(set)
    common = set.intersection(*[have.get(a, set()) for a in arms])
    return sorted(common)


def attach_meta(matches, df_cv):
    """Add date / season / season_round via the (round, home, away) identity
    that run_cv_window records for every prediction."""
    home = df_cv[df_cv["is_home"] == 1][["round", "team", "opp_team", "datetime", "season"]].rename(
        columns={"team": "home", "opp_team": "away"})
    out = matches.merge(home, on=["round", "home", "away"], how="left", validate="many_to_one")
    assert out["datetime"].notna().all(), "unjoined predictions — identity columns disagree with df_cv"
    first = df_cv.groupby("season")["round"].min()
    out["date"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["season_round"] = out["round"] - out["season"].map(first) + 1
    out["result"] = np.where(out.goals_home > out.goals_away, "H", np.where(out.goals_home == out.goals_away, "D", "A"))
    return out


def wide(matches, col, arms):
    """(match key) x arm matrix of `col`, asserting every arm saw identical matches."""
    key = ["window", "round", "home", "away"]
    piv = matches[matches.arm.isin(arms)].pivot_table(index=key, columns="arm", values=col)
    assert not piv[arms].isna().any().any(), "arms cover different matches — use complete_windows()"
    return piv[arms]


# --------------------------------------------------------------- statistics --
def bootstrap_mean_ci(values, n_boot=5000, alpha=0.05, seed=0, groups=None):
    """Mean and percentile CI. With `groups` (e.g. CV window ids) resamples whole
    groups instead of individual matches — matches inside one window share a
    single fit, so they aren't independent and the match-level CI is too narrow.
    Reported side by side; the cluster CI is the conservative one."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        bm = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    else:
        g = np.asarray(groups)
        ids = np.unique(g)
        sums = np.array([v[g == i].sum() for i in ids])
        cnts = np.array([(g == i).sum() for i in ids], float)
        pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
        bm = sums[pick].sum(1) / cnts[pick].sum(1)
    lo, hi = np.percentile(bm, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_diff(a, b, groups=None, **kw):
    """mean(a - b) with CI; negative = a better (lower RPS)."""
    return bootstrap_mean_ci(np.asarray(a) - np.asarray(b), groups=groups, **kw)


def verdict(lo, hi, lower_is_better=True):
    if lo > 0:
        return "worse" if lower_is_better else "better"
    if hi < 0:
        return "better" if lower_is_better else "worse"
    return "no clear difference"


def reliability_table(p, y, n_bins=8):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            rows.append({"bin": f"{edges[b]:.2f}-{edges[b+1]:.2f}", "n": int(m.sum()),
                         "pred": round(p[m].mean(), 3), "actual": round(y[m].mean(), 3)})
    return pd.DataFrame(rows)


def rps_hda(p, actual):
    """RPS for (n,3) H/D/A probabilities and 'H'/'D'/'A' outcomes."""
    p = np.asarray(p, float)
    code = np.array([{"H": 0, "D": 1, "A": 2}[a] for a in actual])
    onehot = np.eye(3)[code]
    return 0.5 * ((np.cumsum(p, 1)[:, :2] - np.cumsum(onehot, 1)[:, :2]) ** 2).sum(1)


# ------------------------------------------------------------------- odds ----
def devig_proportional(o):
    inv = 1.0 / np.asarray(o, float)
    return inv / inv.sum()


def devig_shin(o):
    inv = 1.0 / np.asarray(o, float)
    B = inv.sum()
    if B <= 1.0:
        return inv / B

    def probs(z):
        return (np.sqrt(z * z + 4 * (1 - z) * inv * inv / B) - z) / (2 * (1 - z))

    try:
        z = brentq(lambda z: probs(z).sum() - 1.0, 1e-9, 0.5)
    except ValueError:
        z = 0.0
    p = probs(z)
    return p / p.sum()


def load_odds(season_codes=("2021", "2122", "2223", "2324", "2425", "2526"), cache=None):
    """football-data.co.uk EPL odds, cached. Returns None (not an error) when
    unavailable so the notebook degrades to 'benchmark pending' instead of dying.
    Reuses WP003's cache if it exists."""
    candidates = [Path(cache)] if cache else []
    candidates += [WP_DIR / "odds_raw.pkl", WP_DIR.parent / "wp003_bookmaker_benchmark" / "odds_raw.pkl"]
    for c in candidates:
        if c.exists():
            return pd.read_pickle(c)
    frames = []
    for code in season_codes:
        try:
            df = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv", encoding="latin-1")
            df["season_code"] = code
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"odds {code} unavailable: {type(e).__name__}")
            return None
    odds = pd.concat(frames, ignore_index=True)
    odds["Date"] = pd.to_datetime(odds["Date"], dayfirst=True).dt.normalize()
    odds.to_pickle(WP_DIR / "odds_raw.pkl")
    return odds


def attach_pinnacle(match_df, odds):
    """Join Pinnacle closing odds (PSC*, falling back to PS*) onto per-match rows
    (one arm's rows or the arm-independent identity columns), add de-vigged probs."""
    m = match_df.assign(home_fd=match_df.home.map(CODE_TO_FD), away_fd=match_df.away.map(CODE_TO_FD))
    j = m.merge(odds, left_on=["date", "home_fd", "away_fd"], right_on=["Date", "HomeTeam", "AwayTeam"], how="left")
    cols = next(c for c in (("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA")) if all(x in j.columns for x in c))
    ok = j[list(cols)].notna().all(axis=1)
    P = np.full((len(j), 3), np.nan)
    Ps = np.full((len(j), 3), np.nan)
    if ok.any():
        arr = j.loc[ok, list(cols)].to_numpy(float)
        P[ok.to_numpy()] = np.array([devig_proportional(r) for r in arr])
        Ps[ok.to_numpy()] = np.array([devig_shin(r) for r in arr])
    j[["pin_h", "pin_d", "pin_a"]] = P
    j[["pin_shin_h", "pin_shin_d", "pin_shin_a"]] = Ps
    return j
