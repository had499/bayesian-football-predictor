"""WP055 — a TEST_LEN = 1 grid: deployment-accurate measurement.

THE PROBLEM. Every result in this project comes from a grid with STEP = TEST_LEN = 3, so
each window predicts THREE rounds ahead and the reported RPS averages forecast horizons
1, 2 and 3. But deployment refits after every round and only ever predicts the NEXT one.
Horizon 1 is the only horizon you would experience.

Subsetting the existing 115-window grid to each window's first test round showed:

    horizon   n     model RPS  market RPS      gap vs Pinnacle OPEN
          1  1205     0.19162     0.19068   +0.00093 [-0.00101,+0.00297]  CI SPANS ZERO
          2  1202     0.19885     0.19407   +0.00478 [+0.00244,+0.00721]
          3  1190     0.19796     0.19411   +0.00385 [+0.00169,+0.00600]
        ALL  3597     0.19613     0.19295   +0.00319 [+0.00198,+0.00442]

i.e. roughly two thirds of the gap we have been chasing is an artefact of the CV design.

WHY THAT SUBSET IS NOT ENOUGH, and this grid is needed. The horizon-1 rows are every
THIRD round, not a random third of matches, and the MARKET also scores better on them
(0.19068 vs 0.19407) -- so they are intrinsically more predictable for everyone.
Deployment would predict EVERY round at horizon 1, including the currently-harder ones.
Decomposing (market degrades +0.0034 from h1 to h2 = match difficulty; model degrades
+0.0072; so ~+0.0038 is model-specific horizon cost) independently implies a deployed gap
near +0.0010 -- but that is an inference, not a measurement. This grid measures it.

DESIGN. STEP = TEST_LEN = 1 from the same first test round as the 115-window grid, so
every round gets its own window and every prediction is horizon 1. Test sets remain
disjoint, which the window-cluster bootstrap requires -- and there are now ~345 clusters
of ~10 matches instead of 115 of ~33, which if anything helps the bootstrap.
"""
import pickle
from pathlib import Path

HERE = Path(__file__).resolve().parent
WP = HERE.parent
SRC = WP / "wp039_market_channel" / "cv_shared_data_market.pkl"
OUT = HERE / "cv_shared_data_h1.pkl"


def main():
    d = pickle.loads(SRC.read_bytes())
    old = d["windows"]
    first = min(w["test_start"] for w in old)
    last = max(w["test_end"] for w in old)
    windows = [{"train_start": 1, "train_end": t - 1, "test_start": t, "test_end": t}
               for t in range(first, last + 1)]
    df = d["df_cv"]
    n_test = sum(((df.cum_round == w["test_start"]) & (df.is_home == 1)).sum()
                 for w in windows)
    print(f"old grid: {len(old)} windows, STEP/TEST_LEN 3, rounds {first}-{last}")
    print(f"new grid: {len(windows)} windows, STEP/TEST_LEN 1, {n_test} test matches")
    assert all(w["test_start"] == w["train_end"] + 1 for w in windows), "train/test overlap"
    assert all(w["test_start"] == w["test_end"] for w in windows), "not horizon 1"
    starts = [w["test_start"] for w in windows]
    assert len(set(starts)) == len(starts), "test sets must be disjoint for the bootstrap"
    # the training filter must actually bite (the WP043 leak lesson)
    n_first = (df["round"] <= windows[0]["train_end"]).sum()
    n_last = (df["round"] <= windows[-1]["train_end"]).sum()
    assert n_first < n_last < len(df), \
        f"training filter not biting: {n_first} -> {n_last} of {len(df)}"
    print(f"train-filter check: window 1 keeps {n_first} rows, "
          f"window {len(windows)} keeps {n_last}, total {len(df)}  OK")
    d["windows"] = windows
    OUT.write_bytes(pickle.dumps(d))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
