"""WP025 — per-team entry Elo for promoted sides.

THE DEFECT THIS TARGETS. `ar1_team_process` gives a promoted team a single global
`entry_shift`. WP007 fitted it at **-0.06** against a real gap of **-0.37**, and the
BACKLOG flags it as "likely mis-specified, not just under-powered — a per-team
version is untested". A global offset also treats every promoted side as identical,
which they plainly are not: measured pre-debut Elo across 33 promotions runs from
Huddersfield 2017 at **1475** to Burnley 2023 at **1726**, a 251-point spread, against
an incumbent mean of 1759.

External Elo is the right information for this, because a promoted team's strength is
exactly where the model has NO match history to learn from, and clubelo rates them on
their Championship form (plus cups) which the EPL-only pipeline never sees.

WHY NOT A COVARIATE OR A CHANNEL. A channel needs E[obs] = f(lambda); Elo is a rating,
not a rate. An additive covariate on theta competes directly with attack - defence, and
Elo correlates 0.936 with the model's own implied strength -- the documented failure
mode of the original rolling-mean xG covariate, fitted at ~zero weight because the
latents already explained it. So Elo enters as a PRIOR on the latent: it scales the
value a team's walk starts at when it enters the league.

DATA SOURCE. api.clubelo.com is blocked on this network (https times out on 443, http
returns 502 empty -- same signature as football-data.co.uk). The main clubelo.com
domain works but serves HTML. The practical route is the GitHub mirror
xgabora/Club-Football-Match-Data-2000-2025, which carries twice-monthly clubelo
snapshots: 273,972 rows, 2000-2026, 984 clubs. Twice-monthly is ample for a rating
that moves slowly and is only read once per team, at entry.

NAME TRAP. clubelo carries BOTH "Nott'm Forest" (41 rows, 2025 onward) and
"Nottm Forest" (547 rows, 2000 onward) for the same club. CODE_TO_FD maps NOT to
football-data's apostrophe spelling, which silently matched the partial series and lost
Forest's 2022 promotion. It is the only such duplicate among the 123 ENG clubs.

    python work_products/wp025_elo_channel/prepare_data.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP021 = HERE.parent / "wp021_power_confirmation"
sys.path.insert(0, str(HERE.parent / "wp007_structural_fixes"))
from analysis import CODE_TO_FD  # noqa: E402

OUT = HERE / "cv_shared_data_elo.pkl"
# clubelo's spelling where it differs from football-data's (see NAME TRAP above)
CLUBELO_FIX = {"Nott'm Forest": "Nottm Forest"}


def elo_name(code):
    fd = CODE_TO_FD.get(code, code)
    return CLUBELO_FIX.get(fd, fd)


def main():
    shared = pickle.loads((WP021 / "cv_shared_data_big.pkl").read_bytes())
    df = shared["df_cv"]
    E = pd.read_csv(HERE / "elo_ratings.csv")
    E["date"] = pd.to_datetime(E.date)
    E = E[E.country == "ENG"].sort_values("date")

    # RE-PROMOTION TRAP: teams enter the league more than once -- Norwich in
    # 2015/2019/2021, Fulham 2018/2020/2022, Burnley 2016/2023/2025, plus Watford,
    # Leeds, Bournemouth and Sheffield United twice each. Their Elo differs at each
    # entry (Norwich: 1618 / 1635 / 1655), so ONE value per team is wrong. Store the
    # full (round x team) grid of standardised Elo and let the model's own entry mask
    # pick the right cell. This is also what the general "reversion target" arm needs.
    teams = sorted(set(df.team) | set(df.opp_team))
    dfd = df.assign(dt=pd.to_datetime(df.datetime))
    round_date = dfd.groupby("cum_round").dt.min().sort_index()

    epl_names = [elo_name(t) for t in teams]
    epl = E[E.club.isin(epl_names)]
    mu, sd = epl.elo.mean(), epl.elo.std()
    piv = E.pivot_table(index="date", columns="club", values="elo").sort_index()

    grid = pd.DataFrame(0.0, index=round_date.index, columns=teams)
    filled = 0
    for t in teams:
        club = elo_name(t)
        if club not in piv.columns:
            print(f"WARNING {t} ({club}) absent from clubelo entirely")
            continue
        s_ = piv[club].dropna()
        # most recent snapshot STRICTLY BEFORE each round's first match (leak-free)
        vals = s_.reindex(s_.index.union(round_date.values)).ffill().reindex(round_date.values)
        z = (vals.to_numpy() - mu) / sd
        grid[t] = np.nan_to_num(z, nan=0.0)
        filled += 1

    print(f"{len(teams)} teams, {filled} found in clubelo; EPL Elo mean {mu:.0f} sd {sd:.0f}")
    print(f"grid shape {grid.shape} (rounds x teams)")
    print(f"z range {grid.to_numpy().min():+.2f} .. {grid.to_numpy().max():+.2f}")

    # sanity: what each promotion actually gets
    tb = {ss: set(df[df.season == ss].team) | set(df[df.season == ss].opp_team)
          for ss in sorted(df.season.unique())}
    seasons = sorted(tb)
    print("\n  re-promotions now get distinct entry Elo:")
    for t in ("NOR", "FLH", "BUR"):
        got = []
        for i in range(1, len(seasons)):
            if t in tb[seasons[i]] - tb[seasons[i - 1]]:
                r = dfd[(dfd.season == seasons[i]) & ((dfd.team == t) | (dfd.opp_team == t))].cum_round.min()
                got.append(f"{seasons[i]}:{grid.loc[r, t]:+.2f}")
        if got:
            print(f"    {t}: {'  '.join(got)}")

    shared["elo_z_grid"] = grid          # (cum_round x team) standardised Elo
    shared["elo_mu_sd"] = (float(mu), float(sd))
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
