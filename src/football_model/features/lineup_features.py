"""Leakage-free lineup-quality feature (WP008).

Three stages, each an expanding (all-history-to-date) rolling computation —
the same convention `prepare_model_data.py`'s team-level rolling xG already
uses, kept consistent deliberately rather than introducing a different
windowing scheme with no specific reason to:

1. `add_rolling_player_rate` — each player's own per-90 (xG+xA) rate, using
   only THEIR strictly-earlier matches, blended toward a population mean
   when their history is thin (a trust weight, not a hard cutoff).
2. `compute_starting_xi_rate` — one number per (team, match): the
   trust-weighted average rolling rate of that match's actual STARTING XI
   only (`started == True` — substitutes excluded; see the module docstring
   in `get_player_data.py` and the WP008 design notes for why: their
   in-match minutes are a consequence of the match being predicted, not
   pre-match information).
3. `add_lineup_deviation` — each team's own rolling normal of #2's output,
   and the log-ratio of today's lineup rate against it. This, not the
   absolute rate, is what should ever reach `theta` — an absolute score
   would just redescribe "this is a good team," which `attack` already
   captures; the deviation is specifically "today's XI vs this team's own
   recent normal."

Every rolling quantity here is computed with `groupby(...).cumsum() - this
row's own value`, which sums only STRICTLY EARLIER rows of the same group —
this is the actual leakage boundary and the reason it's spelled out
identically in three places below rather than abstracted into one helper:
each stage's "earlier" means something different (earlier matches for that
PLAYER, vs. earlier matches for that TEAM), so collapsing them risks
silently computing the wrong group's history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_rolling_player_rate(
    match_logs: pd.DataFrame,
    min_minutes_for_full_trust: float = 900.0,
    population_rate: float | None = None,
) -> pd.DataFrame:
    """Adds `rolling_rate` (per-90 xG+xA, blended toward the population mean
    for thin histories) and `trust_weight` (0 with no history, ramping
    linearly to 1 at `min_minutes_for_full_trust` minutes of past playing
    time — 900 minutes is 10 full matches) to a copy of `match_logs`.

    `population_rate` defaults to the dataset's own overall per-90 rate if
    not given — pass it explicitly at prediction time so a screening subset
    doesn't silently use a different population mean than training did.
    """
    df = match_logs.sort_values(["player_id", "date"]).reset_index(drop=True)
    contribution = df["xG"].fillna(0) + df["xA"].fillna(0)
    minutes = df["time"].fillna(0)

    grp_id = df["player_id"]
    past_minutes = minutes.groupby(grp_id).cumsum() - minutes
    past_contribution = contribution.groupby(grp_id).cumsum() - contribution

    if population_rate is None:
        total_minutes = minutes.sum()
        population_rate = float(contribution.sum() / total_minutes * 90) if total_minutes > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        own_rate = np.where(past_minutes > 0, past_contribution / past_minutes.replace(0, np.nan) * 90, population_rate)
    own_rate = np.nan_to_num(own_rate, nan=population_rate)

    trust_weight = np.clip(past_minutes / min_minutes_for_full_trust, 0.0, 1.0)

    df = df.copy()
    df["past_minutes"] = past_minutes
    df["rolling_rate"] = trust_weight * own_rate + (1 - trust_weight) * population_rate
    df["trust_weight"] = trust_weight
    return df


def compute_starting_xi_rate(match_logs_with_rates: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, date): the trust-weighted average `rolling_rate`
    of that match's confirmed starters. `trust_weight` is floored at 0.05
    (never fully zero) purely to avoid a divide-by-zero if an entire
    starting XI somehow had zero playing history — vanishingly unlikely,
    just a defensive floor, not a meaningful modelling choice.
    """
    starters = match_logs_with_rates[match_logs_with_rates["started"]].copy()
    starters["_w"] = starters["trust_weight"].clip(lower=0.05)

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "today_rate": float((g["rolling_rate"] * g["_w"]).sum() / g["_w"].sum()),
            "n_starters": len(g),
        })

    out = starters.groupby(["team", "date"], as_index=False).apply(_agg, include_groups=False)
    return out.reset_index(drop=True)


def add_lineup_deviation(team_match_rates: pd.DataFrame, eps: float = 1e-4) -> pd.DataFrame:
    """Adds `normal_rate` (that team's own expanding average of `today_rate`
    over strictly earlier matches) and `lineup_dev` (the log-ratio of today
    vs. that normal — zero for a team's very first tracked match, since
    there's no prior history to compare against)."""
    df = team_match_rates.sort_values(["team", "date"]).reset_index(drop=True)
    grp = df["today_rate"].groupby(df["team"])
    past_sum = grp.cumsum() - df["today_rate"]
    past_n = grp.cumcount()  # 0-indexed position within group = count of strictly earlier rows

    with np.errstate(divide="ignore", invalid="ignore"):
        normal_rate = np.where(past_n > 0, past_sum / past_n.replace(0, np.nan), np.nan)
    df = df.copy()
    df["normal_rate"] = pd.Series(normal_rate).fillna(df["today_rate"])  # no history yet -> deviation 0
    df["lineup_dev"] = np.log((df["today_rate"] + eps) / (df["normal_rate"] + eps))
    return df


def build_lineup_deviation_table(
    match_logs: pd.DataFrame,
    min_minutes_for_full_trust: float = 900.0,
    population_rate: float | None = None,
) -> pd.DataFrame:
    """The full pipeline: match_logs -> one row per (team, date) with
    `lineup_dev` ready to join onto `df_cv` by (date, team). Convenience
    wrapper around the three stages above."""
    with_rates = add_rolling_player_rate(
        match_logs, min_minutes_for_full_trust=min_minutes_for_full_trust,
        population_rate=population_rate,
    )
    xi_rate = compute_starting_xi_rate(with_rates)
    return add_lineup_deviation(xi_rate)
