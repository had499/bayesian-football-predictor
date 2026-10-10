"""WP029 step 1 — one dataset carrying BOTH stakes and congestion.

The champion now includes the stakes effort term, so its baseline is WP028's
`I_effort` arm, and any new arm must have stakes columns too. This merges WP027's
congestion covariates into WP028's stakes dataset.

    python work_products/wp029_congestion_effort/prepare_data.py
"""
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
OUT = HERE / "cv_shared_data_both.pkl"
NEED = ["rest_home", "rest_away", "load_home", "load_away"]


def main():
    stk = pickle.loads((WP / "wp028_stakes/cv_shared_data_stakes.pkl").read_bytes())
    cong = pickle.loads((WP / "wp027_congestion/cv_shared_data_congestion.pkl").read_bytes())
    a, b = stk["df_cv"], cong["df_cv"]
    assert len(a) == len(b), f"row counts differ {len(a)} vs {len(b)}"
    key = ["match_id", "team", "opp_team"]
    merged = a.merge(b[key + NEED], on=key, how="left", validate="one_to_one")
    assert len(merged) == len(a), f"join fanned out {len(a)} -> {len(merged)}"
    miss = merged[NEED].isna().any(axis=1).sum()
    assert miss == 0, f"{miss} rows missing congestion columns"
    stk["df_cv"] = merged
    OUT.write_bytes(pickle.dumps(stk))
    h = merged[merged.is_home == 1]
    print(f"{len(merged)} rows with stakes AND congestion")
    print(f"  stakes_home mean {h.stakes_home.mean():.4f}  rest_home z mean {h.rest_home.mean():+.4f}")
    print(f"  corr(stakes, rest) = {h.stakes_home.corr(h.rest_home):+.3f}  "
          f"(low = the two effort contributions stay identifiable)")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
