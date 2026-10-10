"""WP031 — add the crowdless indicator to WP030's CV data file.

Premier League behind closed doors: Project Restart (2020-06-17) until limited
crowds returned for the final rounds of 2020-21 (~2021-05-09). A scheduling fact,
published months in advance, so legitimately known at the opening price.
"""
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "wp030_squad_intactness" / "cv_shared_data_intact.pkl"
OUT = HERE / "cv_shared_data_crowd.pkl"

FROM, TO = pd.Timestamp("2020-06-17"), pd.Timestamp("2021-05-09")

d = pickle.loads(SRC.read_bytes())
df = d["df_cv"].copy()
dt = pd.to_datetime(df["datetime"])
df["crowdless"] = ((dt >= FROM) & (dt <= TO)).astype(float)
d["df_cv"] = df
OUT.write_bytes(pickle.dumps(d))

n = int(df[df.is_home == 1].crowdless.sum())
tot = int((df.is_home == 1).sum())
print(f"crowdless matches: {n}/{tot} ({n/tot:.1%})")
print("by season:")
print(df[df.is_home == 1].groupby("season").crowdless.agg(["sum", "mean"]).round(3).to_string())
print(f"\nwrote {OUT.name}")
