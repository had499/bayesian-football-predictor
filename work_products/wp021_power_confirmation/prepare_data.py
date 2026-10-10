"""WP021 — build the high-power CV grid: deep + shots on 115 windows.

WP020 found the deep-completion channel worth -0.00096 RPS [-0.00248, +0.00063] —
the best result since the xG channel, but BELOW the 30-window design's +-0.00156
detection threshold, so it could not clear zero even if real.

This grid is chosen so the test is decisive rather than another coin flip:

    windows   MDE      CI the WP020 effect would produce     verdict
        30    0.00156  [-0.00252, +0.00060]                  still null
        77    0.00097  [-0.00193, +0.00001]                  borderline   <- BACKLOG's plan
       115    0.00080  [-0.00176, -0.00016]                  CLEARS ZERO

115 windows = first test season 2016, STEP 3, TEST_LEN 3. STEP == TEST_LEN keeps
test sets contiguous and NON-overlapping, which the window-cluster bootstrap
requires — it treats windows as independent clusters, so overlapping test sets
would silently violate its assumption. Starting at 2016 rather than 2015 keeps at
least two seasons of training history in the earliest window; 2015 would buy 11
more windows at the cost of fitting the first ones on a single season.

    python work_products/wp021_power_confirmation/prepare_data.py
"""
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
WP014 = HERE.parent / "wp014_npxg_decomposition"
WP020 = HERE.parent / "wp020_deep_channel"
OUT = HERE / "cv_shared_data_big.pkl"

FIRST_TEST_SEASON = "2016"
STEP = 3
TEST_LEN = 3


def shot_counts():
    rows = []
    for f in sorted((WP014 / "shots").glob("*.pkl")):
        for rec in pickle.loads(f.read_bytes()).values():
            m = rec["meta"]
            d = pd.Timestamp(m["datetime"]).normalize()
            for side, team in (("h", m["home_short"]), ("a", m["away_short"])):
                rows.append({"team": team, "d": d, "n_shots": len(rec[side])})
    return pd.DataFrame(rows)


def main():
    shared = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())
    df = shared["df_cv"].copy()
    df["d"] = pd.to_datetime(df["datetime"]).dt.normalize()

    # --- deep completions (WP020), joined via the short-code -> title map ---
    ctx = pickle.loads((WP020 / "team_context.pkl").read_bytes())
    s2t = pickle.loads((WP020 / "short_to_title.pkl").read_bytes())
    ctx = ctx.assign(d=ctx.dt.dt.normalize())[["team", "d", "deep"]]
    assert not ctx.duplicated(subset=["team", "d"]).any()
    df["_t"] = df["team"].map(lambda x: s2t.get(x, x))
    df["_o"] = df["opp_team"].map(lambda x: s2t.get(x, x))
    n0 = len(df)
    df = df.merge(ctx.rename(columns={"team": "_t", "deep": "deep_home"}), on=["_t", "d"], how="left")
    df = df.merge(ctx.rename(columns={"team": "_o", "deep": "deep_away"}), on=["_o", "d"], how="left")

    # --- shot counts (WP019), joined on the short codes directly ---
    sh = shot_counts()
    assert not sh.duplicated(subset=["team", "d"]).any()
    df = df.merge(sh.rename(columns={"n_shots": "shots_home"}), on=["team", "d"], how="left")
    df = df.merge(sh.rename(columns={"team": "opp_team", "n_shots": "shots_away"}),
                  on=["opp_team", "d"], how="left")
    assert len(df) == n0, f"joins changed row count {n0} -> {len(df)}"

    cols = ["deep_home", "deep_away", "shots_home", "shots_away"]
    miss = df[cols].isna().any(axis=1).sum()
    if miss:
        print(f"WARNING {miss} of {n0} rows missing a channel observation — filled with column means")
    df[cols] = df[cols].fillna(df[cols].mean())

    # --- the high-power window grid ---
    last = int(df["round"].max())
    first_test_round = int(df.loc[df["season"] == FIRST_TEST_SEASON, "round"].min()) + 2
    windows, train_end = [], first_test_round - 1
    while train_end + 1 <= last:
        ts = train_end + 1
        windows.append(dict(train_start=1, train_end=train_end, test_start=ts,
                            test_end=min(ts + TEST_LEN - 1, last)))
        train_end += STEP
    assert STEP >= TEST_LEN, "test sets must not overlap — the cluster bootstrap assumes independence"

    h = df[df.is_home == 1]
    n_test = sum(((h["round"] >= w["test_start"]) & (h["round"] <= w["test_end"])).sum() for w in windows)
    print(f"{len(windows)} windows, {n_test} test matches "
          f"(WP020 had {len(shared['windows'])} windows / 1314 matches)")
    print(f"first window trains on rounds 1-{windows[0]['train_end']}, last on 1-{windows[-1]['train_end']}")
    print(f"deep  home {h.deep_home.mean():.2f} vs away {h.deep_away.mean():.2f}")
    print(f"shots home {h.shots_home.mean():.2f} vs away {h.shots_away.mean():.2f}")

    shared["df_cv"] = df.drop(columns=["d", "_t", "_o"])
    shared["windows"] = windows
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
