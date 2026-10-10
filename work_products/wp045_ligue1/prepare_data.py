"""WP043 — build the Ligue 1 CV grid from raw Understat rows.

MODEL SCOPE. L0 + xG channel at weight 0.75 (WP032's tuned value), NOT the full EPL
champion. The xG channel is the only ESTABLISHED win (-0.00258, 6/6 LOSO); `deep` and
`stakes_effort` went in on expected value and would each need new Ligue 1 data (deep
completions, a league-table simulation). Not worth building before knowing whether the
league is interesting at all.

ROUND ASSIGNMENT. Understat gives dates, not rounds. Matches are bucketed by ISO week
within a season, the same scheme the EPL grid uses -- which is also why `(cum_round,
team)` is NOT a unique key there (a documented trap that once fanned a join 3,780 ->
4,812). Every join here uses `(team, date)` instead, and asserts uniqueness.

GRID. Ligue 1 is 20 teams (18 from 2023-24) so 380 then 306 matches/season, so the same
STEP=TEST_LEN=3 design yields fewer test matches per window. First test season 2016,
matching the EPL grid so the two are comparable.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "cv_shared_data_ligue1.pkl"
FIRST_TEST_SEASON = "2016"
STEP = 3
TEST_LEN = 3


def main():
    r = pd.read_pickle(HERE / "understat_ligue1.pkl").copy()
    r["datetime"] = pd.to_datetime(r.datetime)
    r = r.dropna(subset=["goals", "xG"])
    r["season"] = r.season.astype(str)

    # (1) DROP UNPLAYED FIXTURES. Ligue 1 2019-20 was abandoned in March 2020 and
    # Understat still lists the cancelled matches -- all stamped 2020-07-09 with
    # goals = 0 and xG = EXACTLY 0.0. No real match has exactly 0.00 xG, so that is the
    # placeholder signature. 202 rows in 2019 plus a few strays elsewhere.
    # Drop whole MATCHES, not individual rows: a first pass dropped single rows and
    # left 1-sided fixtures, which then failed the home/away pairing assertion.
    r["_mk"] = (pd.to_datetime(r.datetime).dt.strftime("%Y%m%d") + "_" +
                np.where(r.is_home == 1, r.team_long, r.opp_team_long) + "_" +
                np.where(r.is_home == 1, r.opp_team_long, r.team_long))
    bad_mk = set(r.loc[r.xG == 0.0, "_mk"])
    unplayed = r._mk.isin(bad_mk)
    print(f"dropping {unplayed.sum()} rows across {len(bad_mk)} unplayed/placeholder "
          f"matches (by season: {r[unplayed].groupby('season').size().to_dict()})")
    r = r[~unplayed].drop(columns="_mk").copy()

    # (2) USE FULL NAMES AS TEAM IDENTITY, NOT THE 3-LETTER CODE. Ligue 1 has two
    # collisions -- 'NAN' is BOTH Nancy and Nantes, 'AJA' is BOTH Ajaccio and GFC
    # Ajaccio -- 32 codes for 34 clubs. The codes would silently MERGE two different
    # clubs into one latent team, and the giveaway was a 'NAN vs NAN' fixture in 2016.
    # (RFPL has 32 codes for 32 clubs, so its result is unaffected.)
    n_codes, n_long = r.team.nunique(), r.team_long.nunique()
    r["team"] = r["team_long"]
    r["opp_team"] = r["opp_team_long"]
    print(f"team identity: {n_codes} short codes -> {n_long} full names "
          f"({n_long - n_codes} collisions resolved)")
    r = r.sort_values("datetime").reset_index(drop=True)

    # --- rounds from ISO week within season ---
    iso = r.datetime.dt.isocalendar()
    r["_wk"] = iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2)
    rounds = (r.groupby("season")._wk.transform(
        lambda s: pd.factorize(pd.Series(sorted(s.unique())))[0][
            pd.Series(sorted(s.unique())).searchsorted(s)]) + 1)
    r["round"] = rounds.astype(int)

    # --- cumulative round across seasons ---
    seasons = sorted(r.season.unique())
    offset, acc = {}, 0
    for s in seasons:
        offset[s] = acc
        acc += r[r.season == s]["round"].max()
    r["cum_round"] = r["round"] + r.season.map(offset)

    # *** THE HARNESS SPLITS TRAIN/TEST ON `round`, NOT `cum_round`. ***
    # prepare_model_data does `df_train = df[df["round"] <= max_round]`, and the EPL
    # frame happens to carry a CUMULATIVE counter in `round` (1-416 == cum_round). The
    # first version of this script put the within-season round (1-31) in `round`, so
    # every match passed `round <= train_end` (which reaches 349) in every window: the
    # model trained on the full dataset including its own test rows. n_train was a
    # constant 2,880 instead of growing 780 -> 4,548 as the EPL does, and the leak
    # produced a spectacular fake result (model "beating" the Russian closing line by
    # 0.0065 RPS, +17% ROI, profitable in 10/10 seasons).
    r["round_in_season"] = r["round"]
    r["round"] = r["cum_round"]
    assert (r["round"] == r["cum_round"]).all(), "harness needs round == cum_round"

    # --- one id per match, keyed on the pair + date ---
    r["match_key"] = (r.datetime.dt.strftime("%Y%m%d") + "_" +
                      np.where(r.is_home == 1, r.team, r.opp_team) + "_" +
                      np.where(r.is_home == 1, r.opp_team, r.team))
    r["match_id"] = pd.factorize(r.match_key)[0]

    # --- home/away goal and xG columns, from the two perspectives ---
    h = r[r.is_home == 1][["match_key", "goals", "xG"]].rename(
        columns={"goals": "goals_home", "xG": "xG_home"})
    a = r[r.is_home == 0][["match_key", "goals", "xG"]].rename(
        columns={"goals": "goals_away", "xG": "xG_away"})
    assert not h.match_key.duplicated().any(), "duplicate home rows per match"
    assert not a.match_key.duplicated().any(), "duplicate away rows per match"
    r = r.merge(h, on="match_key", how="left").merge(a, on="match_key", how="left")
    assert r[["goals_home", "goals_away"]].notna().all().all(), "unpaired matches"
    r = r.drop(columns="_wk")

    # the documented trap: (team, date) MUST be unique or every later join fans out
    key = pd.DataFrame({"team": r.team, "d": r.datetime.dt.normalize()})
    assert not key.duplicated().any(), "(team, date) is not unique"

    print(f"Ligue 1 engineered: {len(r)} team-rows, {r.match_id.nunique()} matches, "
          f"{r.team.nunique()} teams, seasons {seasons[0]}-{seasons[-1]}")
    print(r.groupby("season").agg(matches=("match_id", "nunique"),
                                  rounds=("round", "max")).to_string())

    # --- walk-forward windows, same design as the EPL grid ---
    first = r[r.season >= FIRST_TEST_SEASON].cum_round.min()
    last = r.cum_round.max()
    windows = []
    t = int(first)
    while t + TEST_LEN - 1 <= last:
        windows.append({"train_start": 1, "train_end": t - 1,
                        "test_start": t, "test_end": t + TEST_LEN - 1})
        t += STEP
    n_test = sum(((r.cum_round >= w["test_start"]) &
                  (r.cum_round <= w["test_end"]) & (r.is_home == 1)).sum()
                 for w in windows)
    print(f"\n{len(windows)} windows, {n_test} test matches "
          f"(EPL grid: 115 windows, 3,780 matches)")
    assert all(w["test_start"] == w["train_end"] + 1 for w in windows), "train/test overlap"
    # Guard the leak directly: the training filter the harness will apply must actually
    # EXCLUDE a growing share of the data, not pass everything through.
    n_first = (r["round"] <= windows[0]["train_end"]).sum()
    n_last = (r["round"] <= windows[-1]["train_end"]).sum()
    assert n_first < n_last < len(r), (
        f"training filter is not biting: first window keeps {n_first}, last {n_last}, "
        f"total {len(r)} — `round` is probably not the cumulative counter")
    print(f"train-filter check: window 1 keeps {n_first} rows, "
          f"window {len(windows)} keeps {n_last}, total {len(r)}  OK")

    OUT.write_bytes(pickle.dumps({"df_cv": r, "windows": windows}))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
