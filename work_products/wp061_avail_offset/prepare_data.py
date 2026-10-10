"""WP061 — add availability columns to the horizon-1 grid, changing nothing else.

NaNs are filled with 0.0 (= no adjustment) BEFORE saving, deliberately: if a NaN reached
prepare_model_data and it drops rows with missing values, the grid would silently shrink
and the arm would no longer be comparable to wp055's champion baseline. That is exactly
the class of bug that produced WP043's fake result. Row count and window list are asserted
identical to the source grid.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
SRC = WP / "wp055_horizon1" / "cv_shared_data_h1.pkl"
OUT = HERE / "cv_shared_data_avail.pkl"

d = pickle.loads(SRC.read_bytes())
df = d["df_cv"].copy()
n0, w0 = len(df), list(d["windows"])

av = pd.read_pickle(WP / "wp059_injuries" / "availability.pkl")
av["out_tot"] = av.out_w_new + av.out_w_old
look = av.set_index(["team", "d"]).out_tot

df["_d"] = pd.to_datetime(df.datetime).dt.normalize()
home = np.where(df.is_home == 1, df.team, df.opp_team)
away = np.where(df.is_home == 1, df.opp_team, df.team)
df["avail_out_h"] = look.reindex(pd.MultiIndex.from_arrays([home, df._d])).to_numpy()
df["avail_out_a"] = look.reindex(pd.MultiIndex.from_arrays([away, df._d])).to_numpy()
cov = df[["avail_out_h", "avail_out_a"]].notna().all(1).mean()
print(f"availability present on both sides: {cov:.1%}")
df["avail_out_h"] = df.avail_out_h.fillna(0.0)
df["avail_out_a"] = df.avail_out_a.fillna(0.0)
df = df.drop(columns=["_d"])

assert len(df) == n0, f"row count changed {n0} -> {len(df)}"
assert list(d["windows"]) == w0, "window list changed"
assert df[["avail_out_h", "avail_out_a"]].notna().all().all(), "NaN survived the fill"
assert (df.avail_out_h.between(0, 0.95)).all(), "out share outside [0, 0.95]"
print(f"rows {len(df)}, windows {len(w0)}  (both unchanged)")
print(f"avail_out_h mean {df.avail_out_h.mean():.4f}  nonzero {(df.avail_out_h>0).mean():.1%}")
d["df_cv"] = df
OUT.write_bytes(pickle.dumps(d))
print(f"wrote {OUT.name}")
