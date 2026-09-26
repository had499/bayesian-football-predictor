"""Leak-safe squad features from cached Understat rosters.

Every feature for a match at date D uses ONLY that team's matches strictly
before D. Nothing here may touch the roster of the match being predicted —
that's the whole point: these have to be computable days ahead, at the same
time the opening line is set, or they can't explain anything about it.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "rosters"

MINUTES_WINDOW = 8      # matches of history for expected-minutes EWMA
MINUTES_HALFLIFE = 3.0  # in matches; recent games weigh more
CONTRIB_WINDOW = 10     # matches of history for xG+xA contribution share
FULL_MATCH = 90.0       # minutes, for converting expected minutes -> availability
XI_WINDOW = 6           # matches compared against for staleness
RECENT_W = 3            # matches defining a player's CURRENT availability
BASE_LO, BASE_HI = 10, 4  # matches back defining their established BASELINE availability


def load_rosters(cache=CACHE):
    """Flatten cached rosters into one long (match, team, player) frame."""
    rows = []
    for f in sorted(Path(cache).glob("*.pkl")):
        for mid, rec in pickle.loads(f.read_bytes()).items():
            meta = rec["meta"]
            for side in ("h", "a"):
                team = meta["home_short"] if side == "h" else meta["away_short"]
                for p in rec[side].values():
                    rows.append({
                        "match_id": mid, "datetime": pd.Timestamp(meta["datetime"]),
                        "season": meta["season"], "team": team, "is_home": side == "h",
                        "player_id": p["player_id"], "position": p["position"],
                        "minutes": float(p["time"]),
                        "xG": float(p["xG"]), "xA": float(p["xA"]),
                    })
    df = pd.DataFrame(rows)
    df["contrib"] = df.xG + df.xA
    return df.sort_values("datetime").reset_index(drop=True)


def _starters(match_rows):
    """XI for one team in one match: non-subs if the position field marks them,
    else the 11 highest-minute players (fallback if the field is unreliable)."""
    non_sub = match_rows[match_rows.position.str.lower() != "sub"]
    if len(non_sub) >= 11:
        return set(non_sub.nlargest(11, "minutes").player_id)
    return set(match_rows.nlargest(11, "minutes").player_id)


def team_match_features(roster):
    """One row per (team, match): squad features computed from PRIOR matches only."""
    out = []
    for team, g in roster.groupby("team"):
        matches = g.groupby("match_id", sort=False).first().sort_values("datetime")
        order = list(matches.index)
        by_match = {mid: g[g.match_id == mid] for mid in order}
        xis = {mid: _starters(by_match[mid]) for mid in order}

        for i, mid in enumerate(order):
            hist = order[max(0, i - CONTRIB_WINDOW):i]      # strictly before this match
            if len(hist) < 3:                                # too little history to be meaningful
                out.append({"team": team, "match_id": mid, "datetime": matches.loc[mid, "datetime"]})
                continue

            # --- expected minutes: EWMA over the last MINUTES_WINDOW matches, where a
            # player absent from a match counts as 0 (so someone who stops playing decays).
            mins_hist = hist[-MINUTES_WINDOW:]
            players = set().union(*[set(by_match[h].player_id) for h in mins_hist])
            w = 0.5 ** (np.arange(len(mins_hist))[::-1] / MINUTES_HALFLIFE)  # most recent = weight 1
            w = w / w.sum()
            exp_min = {}
            for p in players:
                mins = [float(by_match[h].loc[by_match[h].player_id == p, "minutes"].sum()) for h in mins_hist]
                exp_min[p] = float(np.dot(w, mins))

            exp = pd.Series(exp_min).sort_values(ascending=False)
            expected_xi = set(exp.head(11).index)

            # --- instability: share of trailing minutes NOT going to the top 11.
            # A settled side concentrates minutes in 11 players; a rotating one spreads them.
            total_min = exp.sum()
            instability = float(1.0 - (exp.head(11).sum() / total_min)) if total_min > 0 else np.nan

            # --- absent-contributor share: how much of the team's recent xG+xA was
            # produced by players who are NOT expected to be on the pitch now. The
            # "key man is out" signal.
            #
            # Weighted continuously by expected availability rather than a hard
            # bench threshold. A threshold is badly behaved here: at half-life 3, a
            # player who stops playing still shows ~37 expected minutes after three
            # missed matches, so any sane cutoff needs FOUR missed games to fire —
            # long after the market has repriced. Continuous weighting starts
            # registering the absence immediately and degrades gracefully.
            ch = pd.concat([by_match[h] for h in hist])
            contrib = ch.groupby("player_id").contrib.sum()
            tot = contrib.sum()
            if tot > 0:
                unavail = np.array([1.0 - min(exp_min.get(p, 0.0) / FULL_MATCH, 1.0) for p in contrib.index])
                absent_share = float((contrib.to_numpy() * unavail).sum() / tot)
            else:
                absent_share = np.nan

            # --- absence DROP: the same idea as absent_share but measured against each
            # player's OWN norm rather than against a full 90 minutes.
            #
            # absent_share above turned out to carry a ~0.31 structural baseline for
            # every team — in any squad roughly a third of xG comes from players who
            # don't average a full match (subs, rotation), so it scores squad-production
            # DISPERSION, not "our key man is out". Chelsea and Ipswich topped it, not
            # injury-hit sides. This version asks instead: whose availability has FALLEN
            # below their own established level? A rotation player who always plays 45
            # scores 0; a starter who has stopped playing scores ~1.
            base_hist = hist[max(0, len(hist) - BASE_LO):max(0, len(hist) - BASE_HI)]
            rec_hist = hist[-RECENT_W:]
            if base_hist and tot > 0:
                def _avg_min(ms, p):
                    return float(np.mean([float(by_match[h].loc[by_match[h].player_id == p, "minutes"].sum())
                                          for h in ms])) if ms else 0.0
                drops = []
                for p in contrib.index:
                    b, c = _avg_min(base_hist, p), _avg_min(rec_hist, p)
                    drops.append(max(0.0, b - c) / b if b > 1.0 else 0.0)
                absent_drop = float((contrib.to_numpy() * np.array(drops)).sum() / tot)
            else:
                absent_drop = np.nan

            # --- staleness: how unlike the current expected XI were the XIs that
            # actually produced the recent results the model learned from.
            recent = hist[-XI_WINDOW:]
            overlaps = [len(expected_xi & xis[h]) / 11.0 for h in recent]
            staleness = float(1.0 - np.mean(overlaps)) if overlaps else np.nan

            out.append({
                "team": team, "match_id": mid, "datetime": matches.loc[mid, "datetime"],
                "instability": instability, "absent_share": absent_share,
                "absent_drop": absent_drop, "staleness": staleness, "n_hist": len(hist),
            })
    return pd.DataFrame(out)


def attach_to_matches(match_df, feats):
    """Join home/away squad features onto a per-match frame keyed by
    (date, home, away) Understat short codes."""
    f = feats.copy()
    f["date"] = f.datetime.dt.normalize()
    cols = ["instability", "absent_share", "absent_drop", "staleness"]
    h = f.rename(columns={"team": "home", **{c: f"h_{c}" for c in cols}})[["home", "date"] + [f"h_{c}" for c in cols]]
    a = f.rename(columns={"team": "away", **{c: f"a_{c}" for c in cols}})[["away", "date"] + [f"a_{c}" for c in cols]]
    m = match_df.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    n0 = len(m)
    m = m.merge(h, on=["home", "date"], how="left").merge(a, on=["away", "date"], how="left")
    assert len(m) == n0, f"squad-feature join changed row count {n0} -> {len(m)}"
    for c in cols:
        m[f"d_{c}"] = m[f"h_{c}"] - m[f"a_{c}"]   # home minus away
        m[f"max_{c}"] = m[[f"h_{c}", f"a_{c}"]].max(axis=1)
    return m
