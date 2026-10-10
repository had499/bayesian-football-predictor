"""WP028 — end-of-season stakes: how much does THIS result change a team's season?

WHY THIS IDEA IS DIFFERENT FROM THE ONES THAT FAILED. Incentive exists only NOW. A
mid-table side in May has nothing to play for, and that fact is not in past results --
the latent walk learns "they played badly in April" only AFTER those matches. It is
forward-looking in a way the walk structurally cannot anticipate. It also needs no new
data: the league table is already in df_cv.

DEFINITION. Stakes = how much the outcome of this match moves the team's end-of-season
probabilities. For each (season, round) the remaining fixtures are simulated many
times; within those simulations the team's current-round result is conditioned on
(win / draw / loss) and P(outcome) recomputed. Stakes is the spread.

    stakes = max_r P(outcome | result=r) - min_r P(outcome | result=r)

This derives the four situations (title race, Europe, relegation, nothing to play for)
from the arithmetic instead of imposing hand-drawn categories, and it correctly
separates "five points clear with three to play" from "level on points".

THREE BOUNDARIES, deliberately not more. P(top 4) and P(top 5) sensitivities are
near-collinear for most teams in most weeks, and WP011's crux test looked significant
purely because correlated features split into cancelling coefficients. So:

  relegation   bottom 3     unambiguous in every season, no cup cascades
  europe       top N        N from the season's OBSERVED qualifier count where
                            available (7-8 from 2020-21), else assumed 6
  title        1st          unambiguous but touches few teams in few weeks

LEAK SAFETY. The simulation's scoring rates come only from matches STRICTLY BEFORE the
current round, and the remaining fixture list is the schedule, which is known in
advance. So stakes is genuine pre-match information.
"""
import numpy as np
import pandas as pd

# European qualifiers by season (Understat label). Observed from openfootball
# participation where it carries EL/UECL (2020-21 onward); ASSUMED 6 before that,
# because openfootball has no EL data pre-2020 and England had ~4 CL + 2-3 EL places
# throughout. The assumption covers 6 of 12 seasons -- report those separately.
EUROPE_N = {"2014": 6, "2015": 6, "2016": 6, "2017": 6, "2018": 6, "2019": 6,
            "2020": 7, "2021": 7, "2022": 7, "2023": 8, "2024": 7, "2025": 7}
EUROPE_OBSERVED_FROM = "2020"
N_SIM = 3000


def season_tables(df):
    """-> per-season match list with running points, home perspective only."""
    h = df[df.is_home == 1][["season", "cum_round", "team", "opp_team",
                             "goals_home", "goals_away", "datetime"]].copy()
    return h.sort_values(["season", "cum_round"])


def _pts(gh, ga):
    return (3 if gh > ga else (1 if gh == ga else 0),
            3 if ga > gh else (1 if gh == ga else 0))


def _history_rates(prior, ti, n, lg_prior=1.35, pseudo=8):
    """Attack/defence multipliers from EVERY prior match, not just this season.

    Using season-to-date only was a real mis-specification. In September it makes all
    twenty teams look alike, and when teams are alike the final table is decided purely
    by simulated noise -- so 3 points moves you several places and P(relegation) looks
    highly sensitive. Counter-intuitively, MORE shrinkage made this worse (early
    relegation stakes rose 0.0555 -> 0.0774 going from 5 to 40 pseudo-matches), because
    shrinkage is what destroys the heterogeneity. Real heterogeneity is what makes an
    individual September result unimportant, so the fix is a BETTER strength estimate.
    """
    gf = np.zeros(n); ga = np.zeros(n); mp = np.zeros(n)
    for _, hm, aw, gh, gaa, _dt in prior:
        if hm in ti:
            gf[ti[hm]] += gh; ga[ti[hm]] += gaa; mp[ti[hm]] += 1
        if aw in ti:
            gf[ti[aw]] += gaa; ga[ti[aw]] += gh; mp[ti[aw]] += 1
    tot = max(mp.sum(), 1)
    lg = (gf.sum() + 20 * lg_prior) / (tot + 20)
    att = (gf + pseudo * lg) / (mp + pseudo) / lg
    dfn = (ga + pseudo * lg) / (mp + pseudo) / lg
    return lg, att, dfn


def compute(df, n_sim=N_SIM, seed=0):
    """-> DataFrame of (season, cum_round, team, stakes_relegation/europe/title)."""
    rng = np.random.default_rng(seed)
    H = season_tables(df)
    allm = H[["cum_round", "team", "opp_team", "goals_home", "goals_away",
              "datetime"]].to_numpy(object)
    out = []
    for season, sg in H.groupby("season"):
        teams = sorted(set(sg.team) | set(sg.opp_team))
        ti = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        rounds = sorted(sg.cum_round.unique())
        eu_n = EUROPE_N.get(season, 6)

        # full fixture list and results for the season
        # NOTE the date column: `cum_round` is an ISO-week bucket, so a team playing
        # twice in one week produces TWO rows. Keying downstream joins on
        # (team, cum_round) silently fans out -- WP009 hit exactly this and its join
        # went 1205 -> 2285 rows. The BACKLOG fix is to key on (team, DATE).
        fx = sg[["cum_round", "team", "opp_team", "goals_home", "goals_away",
                 "datetime"]].to_numpy(object)
        for ri, rnd in enumerate(rounds):
            played = fx[np.array([r < rnd for r, *_ in fx], bool)]
            today = fx[np.array([r == rnd for r, *_ in fx], bool)]
            future = fx[np.array([r > rnd for r, *_ in fx], bool)]
            if len(today) == 0:
                continue
            # points so far THIS season (what the table shows)
            P0 = np.zeros(n)
            for _, hm, aw, gh, gaa, _dt in played:
                ph, pa = _pts(gh, gaa)
                P0[ti[hm]] += ph; P0[ti[aw]] += pa
            # strength from ALL matches strictly before this round, any season
            prior = allm[np.array([r < rnd for r, *_ in allm], bool)]
            lg, att, dfn = _history_rates(prior, ti, n)

            def sim_points(fixtures, draws):
                """-> (draws, n) points accumulated over `fixtures`"""
                acc = np.zeros((draws, n))
                for _, hm, aw, _gh, _ga, _dt in fixtures:
                    i, j = ti[hm], ti[aw]
                    lh = lg * att[i] * dfn[j] * 1.15
                    la = lg * att[j] * dfn[i] / 1.15
                    gh = rng.poisson(lh, draws); gaa = rng.poisson(la, draws)
                    acc[:, i] += np.where(gh > gaa, 3, np.where(gh == gaa, 1, 0))
                    acc[:, j] += np.where(gaa > gh, 3, np.where(gh == gaa, 1, 0))
                return acc

            rest = sim_points(future, n_sim) if len(future) else np.zeros((n_sim, n))

            # for each team playing today, condition on its own result
            for _, hm, aw, gh, gaa, mdt in today:
                for me, opp, is_home in ((hm, aw, True), (aw, hm, False)):
                    i, j = ti[me], ti[opp]
                    base = P0 + rest
                    # other matches today, simulated
                    other = [f for f in today if f[1] != hm or f[2] != aw]
                    oth = sim_points(other, n_sim) if other else np.zeros((n_sim, n))
                    probs = {}
                    for res, (pm, po) in (("W", (3, 0)), ("D", (1, 1)), ("L", (0, 3))):
                        tot = base + oth
                        tot[:, i] += pm; tot[:, j] += po
                        # rank: 1 = best
                        order = (-tot).argsort(axis=1).argsort(axis=1) + 1
                        pos = order[:, i]
                        probs[res] = (float((pos > n - 3).mean()),
                                      float((pos <= eu_n).mean()),
                                      float((pos == 1).mean()))
                        tot[:, i] -= pm; tot[:, j] -= po
                    v = np.array(list(probs.values()))
                    out.append(dict(season=season, cum_round=rnd,
                                    date=pd.Timestamp(mdt).normalize(), team=me,
                                    stakes_relegation=float(v[:, 0].max() - v[:, 0].min()),
                                    stakes_europe=float(v[:, 1].max() - v[:, 1].min()),
                                    stakes_title=float(v[:, 2].max() - v[:, 2].min()),
                                    matches_left=int(len(future) // (n // 2) if n else 0)))
    return pd.DataFrame(out)
