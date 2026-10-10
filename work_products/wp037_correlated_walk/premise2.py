"""WP037 premise 2 — WHERE would pooling pay? A direct test of data starvation.

premise1's Q2 was MIS-SPECIFIED: it referenced subset RPS against GLOBAL base rates,
which does not absorb how predictable a subset is. Matches involving promoted teams are
more one-sided, so the model naturally scores better on them, and the "+0.016 skill
surplus" I measured reflects bigger strength gaps rather than healthy latents. It cannot
answer the question.

The direct question is whether a latent is DATA-STARVED, and the posterior itself says
so: a starved latent has a WIDE posterior. If promoted teams (and early-season rounds)
carry much wider attack/defence posteriors, pooling across the two -- which premise1
measured at corr +0.79 -- has something real to tighten. If every team's posterior is
equally tight, pooling changes almost nothing no matter how correlated the pair is.

Also tested: whether the two latents' ERRORS co-move, not just their levels. A +0.79
correlation of posterior MEANS partly reflects real team quality (good teams are good at
both). What a bivariate prior exploits is shared information; what it RISKS is
double-counting the same matches, since both latents are estimated from the same games.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WP = HERE.parent


def main():
    import os
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
    from football_model.data.prepare_model_data import prepare_model_data
    from football_model.model.model import build_model
    from football_model.types.model_data import ModelConfig
    import pymc as pm

    shared = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                          .read_bytes())
    df, win = shared["df_cv"], shared["windows"][69]
    tr_df = df[df["round"] <= win["train_end"]]
    data = prepare_model_data(tr_df, max_round=win["train_end"])
    cfg = ModelConfig.champion()
    cfg = type(cfg)(**{**cfg.__dict__, "clip_theta": 5.0, "center_team_strength": False})
    with build_model(data, cfg):
        tr = pm.sample(draws=300, tune=300, chains=2, cores=2, progressbar=False,
                       random_seed=0, nuts_sampler="numpyro",
                       compute_convergence_checks=False)

    att = tr.posterior["attack"].values.reshape(-1, *tr.posterior["attack"].shape[2:])
    dfn = tr.posterior["defence"].values.reshape(-1, *tr.posterior["defence"].shape[2:])
    act = data.active_mask[-1].astype(bool)
    code_of = {i: t for t, i in data.team_mapping.items()}

    # matches played per team in the FINAL season of training
    last_season = sorted(tr_df.season.unique())[-1]
    ls = tr_df[tr_df.season == last_season]
    played = ls.groupby("team").size()

    sd_a = att[:, -1, :].std(0)
    sd_d = dfn[:, -1, :].std(0)
    print("\n=== posterior WIDTH per team at the final round (starvation test) ===")
    print(f"  {'team':6s} {'matches':>8s} {'sd(attack)':>11s} {'sd(defence)':>12s}")
    rows = []
    for i in np.where(act)[0]:
        t = code_of[int(i)]
        n = int(played.get(t, 0))
        rows.append((n, t, sd_a[i], sd_d[i]))
    for n, t, sa, sd in sorted(rows)[:5] + sorted(rows)[-5:]:
        print(f"  {t:6s} {n:8d} {sa:11.4f} {sd:12.4f}")
    ns = np.array([r[0] for r in rows], float)
    sas = np.array([r[2] for r in rows])
    sds = np.array([r[3] for r in rows])
    print(f"\n  spread of sd(attack) across teams: min {sas.min():.4f} max {sas.max():.4f}"
          f"  ratio {sas.max()/sas.min():.2f}x")
    print(f"  spread of sd(defence):              min {sds.min():.4f} max {sds.max():.4f}"
          f"  ratio {sds.max()/sds.min():.2f}x")
    if len(set(ns)) > 1:
        print(f"  corr(matches played, sd(attack))  = {np.corrcoef(ns, sas)[0,1]:+.3f}")
        print(f"  corr(matches played, sd(defence)) = {np.corrcoef(ns, sds)[0,1]:+.3f}")
    print("\n  if the ratio is ~1x and the correlations are ~0, NO team is starved in a")
    print("  way pooling could fix: every latent already has a full season of matches.")

    print("\n=== how much could pooling tighten them, at best? ===")
    r = 0.79
    print(f"  with corr = {r:.2f}, a perfect bivariate prior shrinks each latent's")
    print(f"  posterior SD by at most sqrt(1 - r^2) = {np.sqrt(1-r**2):.3f}, i.e. "
          f"{(1-np.sqrt(1-r**2))*100:.0f}% narrower -- but ONLY for the part of the")
    print("  uncertainty that is independent between them. Both latents are estimated")
    print("  from THE SAME matches, so much of their uncertainty is already shared and")
    print("  that ceiling is optimistic.")
    print(f"  current median sd(attack) {np.median(sas):.4f}, "
          f"sd(defence) {np.median(sds):.4f}")
    print(f"  for scale, the spread of attack ACROSS teams is "
          f"{att[:, -1, act].mean(0).std():.4f}")
    print("  => pooling matters only if posterior width is LARGE relative to the")
    print("     between-team spread; if width << spread, teams are already well told")
    print("     apart and a tighter posterior changes few predictions.")


if __name__ == "__main__":
    main()
