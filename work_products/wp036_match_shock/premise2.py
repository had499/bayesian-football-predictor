"""WP036 premise 2 — does home advantage DEPEND ON TEAM STRENGTH?

WHY THIS QUESTION. The dispersion audit found the champion's home lambdas are spread
too wide: ratio 0.9132 [0.8624,0.9648], CI excluding 1.0, while away is fine at 0.9913.
Working out why points at a STRUCTURAL limit, not a calibration one:

    theta_home = intercept + attack[h] - defence[a] + hfa[h]
    theta_away = intercept + attack[a] - defence[h]
    => Var(theta_home) = Var(theta_away) + Var(hfa)  -- home is FORCED to be >= away

and the lambda-variance ratio 0.3230/0.2076 = 1.556 matches (1.571/1.264)^2 = 1.555
exactly, i.e. the model gives both sides identical theta spread and the rest is exp()
amplification. But the data implies Var(lam_home)=0.159 < Var(lam_away)=0.195 -- home
LOWER in absolute terms despite the higher mean. The current form cannot express that.

The textbook mechanism that would: home advantage is LARGER FOR WEAKER TEAMS. A strong
side already dominates, so playing at home adds less; a weak side gains more. That
compresses home lambdas relative to away ones, which is exactly the missing effect.

The model's hfa is `home_adv[team] ~ Normal(home_mu, home_sd)` with home_sd ~ 0.02 --
per-team, but with NO link to strength, and WP034 showed widening that prior does
nothing. So this would be a new term, not a re-tuning.

TEST: bucket matches by the home team's model-implied strength and compare the OBSERVED
home advantage with the model's PREDICTED one. If home advantage shrinks with strength
while the model holds it flat, the term is real.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import load_checkpoints  # noqa: E402


def boot(fn, m, g, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in uq}
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        out[i] = fn(m.iloc[np.concatenate([idx[x] for x in pick])])
    return fn(m), *np.percentile(out, [2.5, 97.5])


def main():
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = m[m.arm == "W075"].copy()
    g = m.window.to_numpy()
    # model-implied strength of the home team for THIS match: how much better the model
    # thinks it is than its opponent, on the log-rate scale, net of home advantage.
    m["strength"] = np.log(m.lambda_home) - np.log(m.lambda_away)
    # observed vs predicted home advantage, in goals
    m["obs_adv"] = m.goals_home - m.goals_away
    m["pred_adv"] = m.lambda_home - m.lambda_away
    m["tilt"] = m.obs_adv - m.pred_adv

    print(f"champion out-of-sample, n={len(m)}, {m.window.nunique()} windows\n")
    print("=== home-team strength quintiles: observed vs predicted home advantage ===")
    print("    tilt > 0 = the model UNDER-rates the home team in that bucket.")
    print("    the hypothesis predicts tilt FALLING with strength (model over-rates")
    print("    home advantage for strong sides, under-rates it for weak ones).\n")
    q = pd.qcut(m.strength, 5, labels=["1 weakest", "2", "3", "4", "5 strongest"])
    print(f"  {'bucket':12s} {'n':>5s} {'strength':>9s} {'obs adv':>8s} {'pred adv':>9s}"
          f" {'tilt':>9s}  {'CI':>22s}")
    tilts = []
    for b, k in m.groupby(q, observed=True):
        t, lo, hi = boot(lambda d: d.tilt.mean(), k, k.window.to_numpy())
        tilts.append(t)
        print(f"  {str(b):12s} {len(k):5d} {k.strength.mean():+9.3f} {k.obs_adv.mean():+8.3f}"
              f" {k.pred_adv.mean():+9.3f} {t:+9.4f}  [{lo:+.4f},{hi:+.4f}]")

    print("\n=== the gradient — is the trend real? ===")
    f = lambda d: np.polyfit(d.strength, d.tilt, 1)[0]
    s, lo, hi = boot(f, m, g)
    print(f"  d(tilt)/d(strength) = {s:+.4f} [{lo:+.4f},{hi:+.4f}]")
    if hi < 0:
        print("  => REAL and NEGATIVE: home advantage shrinks as the home side gets")
        print("     stronger, and the model does not model that. The term is worth building.")
    elif lo > 0:
        print("  => REAL but POSITIVE, the OPPOSITE of the hypothesis.")
    else:
        print("  => indistinguishable from zero: strength-dependent home advantage is")
        print("     NOT the explanation for the home over-dispersion. Do not build it.")

    print("\n=== falsification: the same gradient on the AWAY side ===")
    print("    away dispersion was FINE (0.9913), so the away tilt should show NO")
    print("    comparable gradient. If it shows the same slope, this is a generic")
    print("    strength mis-calibration, not a home-advantage effect.")
    m["tilt_away"] = -m.tilt
    f2 = lambda d: np.polyfit(-d.strength, d.tilt_away, 1)[0]
    s2, lo2, hi2 = boot(f2, m, g)
    print(f"  d(tilt_away)/d(away strength) = {s2:+.4f} [{lo2:+.4f},{hi2:+.4f}]")


if __name__ == "__main__":
    main()
