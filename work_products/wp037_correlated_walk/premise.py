"""WP037 premise — would correlating the attack and defence walks actually help?

THE IDEA (the last structural lead standing). attack[t,team] and defence[t,team] are
currently INDEPENDENT AR(1) walks: no shared team-quality factor, no LKJ, no cholesky
anywhere in src/. A correlated bivariate walk would pool information across the two, so
a team's attack would inform its defence and vice versa.

TWO THINGS MUST BOTH BE TRUE, and either one failing kills it:

  Q1. The two latents must actually CO-MOVE across teams. If corr(attack, defence) ~ 0
      there is nothing to pool and the correlation parameter fits at zero.

  Q2. There must be a POPULATION WHERE POOLING PAYS -- somewhere each latent is
      individually data-starved so that borrowing from its partner helps. The candidate
      is promoted teams and early season, which is exactly where WP025's Elo ENTRY prior
      fitted at 169% of its prior SD before being declined on external-data grounds.
      If the model is NOT unusually weak there, pooling buys nothing even if Q1 holds:
      with a full season of matches each latent is well identified on its own.

Q1 needs the posterior latents, which the checkpoints do not store, so it runs one fit
in-process. Q2 runs off the 3,780 existing predictions.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import load_checkpoints, paired_diff  # noqa: E402


def q2_population():
    """Is the model unusually weak on promoted teams / early season?"""
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = m[m.arm == "W075"].copy()
    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    meta = (df[df.is_home == 1][["round", "team", "opp_team", "season"]]
            .rename(columns={"team": "home", "opp_team": "away"}))
    m = m.merge(meta, on=["round", "home", "away"], how="left")
    assert m.season.notna().all()

    # which teams are NEW to the league this season?
    seasons = sorted(df.season.unique())
    teams_by_season = {s: set(df[df.season == s].team) | set(df[df.season == s].opp_team)
                       for s in seasons}
    promoted = {}
    for i, s in enumerate(seasons):
        promoted[s] = (teams_by_season[s] - teams_by_season[seasons[i - 1]]) if i else set()

    m["n_promoted"] = [sum(t in promoted[s] for t in (h, a))
                       for s, h, a in zip(m.season, m.home, m.away)]
    # round WITHIN the season
    first = df.groupby("season")["round"].min()
    m["season_round"] = m["round"] - m.season.map(first) + 1

    print("=== Q2a. is the model weaker where latents are DATA-STARVED? ===")
    print("    RPS is not comparable across subsets (upsets are inherently harder), so")
    print("    the reference is the NAIVE league-base-rate RPS on the SAME matches --")
    print("    that absorbs how predictable each subset is in principle.")
    base = np.array([0.0, 0.0, 0.0])
    out = m.copy()
    # naive = overall observed base rates on the full test set
    gh, ga = out.goals_home.to_numpy(), out.goals_away.to_numpy()
    oc = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
    p0 = np.array([(oc == k).mean() for k in range(3)])
    y = np.zeros((len(oc), 3)); y[np.arange(len(oc)), oc] = 1
    naive = ((np.tile(p0, (len(oc), 1)).cumsum(1) - y.cumsum(1))[:, :2] ** 2).sum(1) / 2
    out["naive"] = naive
    out["skill"] = out.naive - out.rps          # higher = more skill over base rates

    print(f"  {'subset':28s} {'n':>5s} {'RPS':>8s} {'naive':>8s} {'skill':>9s}")
    rows = [("all", np.ones(len(out), bool)),
            ("season rounds 1-5", (out.season_round <= 5).to_numpy()),
            ("season rounds 6-38", (out.season_round > 5).to_numpy()),
            ("has a promoted team", (out.n_promoted > 0).to_numpy()),
            ("established only", (out.n_promoted == 0).to_numpy()),
            ("promoted AND rounds 1-5",
             ((out.n_promoted > 0) & (out.season_round <= 5)).to_numpy())]
    skills = {}
    for lab, k in rows:
        if k.sum() < 50:
            print(f"  {lab:28s} n={k.sum():5d}  too few"); continue
        skills[lab] = out.skill[k].mean()
        print(f"  {lab:28s} {k.sum():5d} {out.rps[k].mean():8.5f} "
              f"{out.naive[k].mean():8.5f} {out.skill[k].mean():+9.5f}")

    print("\n  interpretation: if 'rounds 1-5' and 'promoted' show MUCH less skill than")
    print("  their complements, latents are starved there and pooling has a job to do.")
    for a, b in (("season rounds 1-5", "season rounds 6-38"),
                 ("has a promoted team", "established only")):
        if a in skills and b in skills:
            print(f"    {a:22s} {skills[a]:+.5f}  vs  {b:20s} {skills[b]:+.5f}"
                  f"   deficit {skills[a]-skills[b]:+.5f}")


def q1_correlation():
    """Do attack and defence co-move across teams? Needs the posterior, so: one fit."""
    import os
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
    from football_model.data.prepare_model_data import prepare_model_data
    from football_model.model.model import build_model
    from football_model.types.model_data import ModelConfig
    import pymc as pm

    shared = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                          .read_bytes())
    df, win = shared["df_cv"], shared["windows"][69]
    data = prepare_model_data(df[df["round"] <= win["train_end"]],
                              max_round=win["train_end"])
    cfg = ModelConfig.champion()
    cfg = type(cfg)(**{**cfg.__dict__, "clip_theta": 5.0, "center_team_strength": False})
    print("\n=== Q1. do attack and defence CO-MOVE across teams? (one fit, window 70) ===")
    with build_model(data, cfg):
        tr = pm.sample(draws=300, tune=300, chains=2, cores=2, progressbar=False,
                       random_seed=0, nuts_sampler="numpyro",
                       compute_convergence_checks=False)
    att = tr.posterior["attack"].mean(dim=["chain", "draw"]).values     # (time, team)
    dfn = tr.posterior["defence"].mean(dim=["chain", "draw"]).values
    act = data.active_mask[-1].astype(bool)
    print("    sign convention: theta_home = ... + attack[h] - defence[a], so a HIGH")
    print("    defence value means a GOOD defence. Good teams should be high on both,")
    print("    i.e. a POSITIVE correlation is the hypothesis.")
    for lab, t in (("final round", -1), ("mid-history", att.shape[0] // 2)):
        a, d = att[t][act], dfn[t][act]
        print(f"  {lab:14s} corr(attack, defence) = {np.corrcoef(a, d)[0, 1]:+.4f}"
              f"   (n_teams={act.sum()})")
    # across the whole panel, within-round then pooled
    cs = [np.corrcoef(att[t][act], dfn[t][act])[0, 1] for t in range(10, att.shape[0])]
    cs = np.array([c for c in cs if np.isfinite(c)])
    print(f"  across {len(cs)} rounds: mean {cs.mean():+.4f}, "
          f"p5 {np.percentile(cs,5):+.4f}, p95 {np.percentile(cs,95):+.4f}")
    print("\n  a strong positive correlation means a bivariate walk has real information")
    print("  to share; near zero means the correlation parameter fits at ~0 and the arm")
    print("  is a guaranteed null.")


if __name__ == "__main__":
    q2_population()
    q1_correlation()
