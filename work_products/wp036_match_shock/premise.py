"""WP036 premise — is there any WITHIN-MATCH goal correlation left to capture?

WP017 concluded the home/away tilt points at "the bivariate-Poisson structure", and
`match_effect_prior` has sat defined-but-never-called in priors.py ever since. The
obvious build is a shared per-match shock m_i added to BOTH sides' log rates.

BUT a shared additive shock on both log rates can only induce POSITIVE correlation
between home and away goals. If the residual correlation is zero or negative, sigma_m
collapses to zero and the arm is a guaranteed null -- so this decides whether to build
it at all, and if so in which form.

Measured on the CHAMPION's own out-of-sample predictions (3,780 matches), because a
defect in this model is the only thing worth fixing.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import load_checkpoints  # noqa: E402


def boot_corr(x, y, groups, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uq}
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        k = np.concatenate([idx[g] for g in pick])
        out[i] = np.corrcoef(x[k], y[k])[0, 1]
    return np.corrcoef(x, y)[0, 1], *np.percentile(out, [2.5, 97.5])


def main():
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = m[m.arm == "W075"].copy()
    print(f"champion out-of-sample, n={len(m)}, {m.window.nunique()} windows\n")

    lh, la = m.lambda_home.to_numpy(), m.lambda_away.to_numpy()
    gh, ga = m.goals_home.to_numpy(float), m.goals_away.to_numpy(float)
    g = m.window.to_numpy()
    # Pearson residuals: (obs - mean)/sd, sd=sqrt(lambda) under Poisson
    rh, ra = (gh - lh) / np.sqrt(lh), (ga - la) / np.sqrt(la)

    print("=== 1. residual correlation between the two sides of a match ===")
    print("    a shared log-rate shock can ONLY produce a POSITIVE value here.")
    c, lo, hi = boot_corr(rh, ra, g)
    print(f"  corr(resid_home, resid_away) = {c:+.4f} [{lo:+.4f},{hi:+.4f}]")
    verdict = ("POSITIVE -> a shared shock is the right form"
               if lo > 0 else
               "NEGATIVE -> a shared shock CANNOT capture this; it would fit sigma_m=0"
               if hi < 0 else
               "indistinguishable from zero -> a shared shock has nothing to capture")
    print(f"  => {verdict}")

    print("\n=== 2. is each MARGIN overdispersed BEYOND lambda heterogeneity? ===")
    print("    NOT Var(goals)/E[lambda] -- that exceeds 1 even under a perfect Poisson")
    print("    model, because lambda varies across matches. Under Poisson with varying")
    print("    lambda the law of total variance gives")
    print("        Var(goals) = E[lambda] + Var(lambda)")
    print("    so THAT is the benchmark. Excess over it is real overdispersion.")
    for nm, obs, lam in (("home", gh, lh), ("away", ga, la)):
        expected = lam.mean() + lam.var()
        ratio = obs.var() / expected
        print(f"  {nm}: Var(goals)={obs.var():.4f}   E[lam]+Var(lam)="
              f"{lam.mean():.4f}+{lam.var():.4f}={expected:.4f}   ratio={ratio:.4f}"
              f"  ({'OVER' if ratio > 1.05 else 'under' if ratio < 0.95 else 'Poisson-consistent'})")
    print("    (WP028's NegBinomial measured ADDED margin overdispersion as WORSE, so a")
    print("     shared shock would import a known cost for whatever correlation buys.)")

    print("\n=== 3. correlation by total goals — is it a scoreline artefact? ===")
    tot = gh + ga
    for lab, k in (("low  (0-1 goals)", tot <= 1), ("mid  (2-3)", (tot >= 2) & (tot <= 3)),
                   ("high (4+)", tot >= 4)):
        cc = np.corrcoef(rh[k], ra[k])[0, 1]
        print(f"  {lab:18s} n={k.sum():5d}  corr={cc:+.4f}")
    print("    (conditioning on the total forces negative correlation mechanically —")
    print("     these are for shape only, the headline is #1.)")

    print("\n=== 4. the raw goal correlation, for reference ===")
    c2, lo2, hi2 = boot_corr(gh, ga, g)
    print(f"  corr(goals_home, goals_away)  = {c2:+.4f} [{lo2:+.4f},{hi2:+.4f}]")


def dispersion_audit():
    """Follow-up: the ratio Var(goals) / (E[lam]+Var(lam)) came out 0.913 HOME vs 0.991
    AWAY. If real, the model's home lambdas are spread ~9% too WIDE -- over-confident at
    telling matches apart -- and the defect is HOME-SPECIFIC, which is the same asymmetry
    WP017 and WP018 both chased. Needs a CI before it is believed."""
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = m[m.arm == "W075"].copy()
    g = m.window.to_numpy()
    rng = np.random.default_rng(0)
    uq = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in uq}

    print("\n=== 5. dispersion ratio with window-cluster CIs ===")
    print("    ratio = Var(goals) / (E[lambda] + Var(lambda));  1.0 = correctly spread,")
    print("    < 1 = the model's lambdas vary MORE than the goals justify.")
    for nm, gcol, lcol in (("home", "goals_home", "lambda_home"),
                           ("away", "goals_away", "lambda_away")):
        obs, lam = m[gcol].to_numpy(float), m[lcol].to_numpy()
        f = lambda o, l: o.var() / (l.mean() + l.var())
        boots = np.empty(4000)
        for i in range(4000):
            pick = rng.choice(uq, size=len(uq), replace=True)
            k = np.concatenate([idx[x] for x in pick])
            boots[i] = f(obs[k], lam[k])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        flag = "REAL (CI excludes 1.0)" if hi < 1.0 or lo > 1.0 else "not distinguishable from 1.0"
        print(f"  {nm}: ratio {f(obs, lam):.4f} [{lo:.4f},{hi:.4f}]   {flag}")

    print("\n    implied correction, if real: the home lambdas' VARIANCE should be")
    obs, lam = m.goals_home.to_numpy(float), m.lambda_home.to_numpy()
    implied = obs.var() - lam.mean()
    print(f"      Var(lambda_home) = {implied:.4f} rather than {lam.var():.4f}"
          f"  -> shrink home theta deviations by ~{np.sqrt(max(implied,0)/lam.var()):.2f}x")
    obs, lam = m.goals_away.to_numpy(float), m.lambda_away.to_numpy()
    implied = obs.var() - lam.mean()
    print(f"      Var(lambda_away) = {implied:.4f} rather than {lam.var():.4f}"
          f"  -> shrink away theta deviations by ~{np.sqrt(max(implied,0)/lam.var()):.2f}x")


if __name__ == "__main__":
    main()
    dispersion_audit()
