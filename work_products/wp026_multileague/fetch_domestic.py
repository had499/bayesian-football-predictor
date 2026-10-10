"""WP026 step 1 — Understat match data for the five big leagues.

The EPL-only model cannot see its teams' European matches. To USE those matches we
need latents for the foreign opponents, which means fitting the five leagues jointly.
Understat carries xG for all five, so the measurement channel that actually works
(-0.0026 RPS, 6/6 seasons) applies league-wide rather than being lost.

RFPL is excluded: its clubs rarely meet the big five in Europe now, so it would add
~200 latents while contributing almost no cross-links.

    python work_products/wp026_multileague/fetch_domestic.py
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "domestic_raw.pkl"
LEAGUES = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
YEARS = [str(y) for y in range(2014, 2026)]


def main():
    from football_model.data.get_data import get_understat_data
    frames = []
    for lg in LEAGUES:
        df = get_understat_data(leagues=[lg], years=YEARS)
        if df is None or not len(df):
            print(f"  {lg}: EMPTY")
            continue
        df["league"] = lg
        frames.append(df)
        print(f"  {lg}: {len(df)} rows, {df.team.nunique()} teams, "
              f"seasons {df.season.min()}-{df.season.max()}", flush=True)
    import pandas as pd
    all_df = pd.concat(frames, ignore_index=True)
    OUT.write_bytes(pickle.dumps(all_df))
    print(f"\n{len(all_df)} rows, {all_df.team.nunique()} teams across "
          f"{all_df.league.nunique()} leagues -> {OUT.name}")


if __name__ == "__main__":
    main()
