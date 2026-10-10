"""WP007 arm definitions — a cumulative ladder from the repo's validated
config (L0) to the fully restructured model (L6), plus two side arms.

Each rung changes ONE thing relative to the previous, so every step's effect
is a paired comparison on the same test matches (see README for the
diagnostic that motivated each rung). All arms share the sampler settings in
SAMPLER, so differences are model differences, not tuning differences.

Fields:
    use_xg / use_dc : the harness's legacy flags (xG *covariate*, Dixon-Coles)
    overrides       : extra ModelConfig fields on top of the harness defaults
                      (clip_theta=5.0, center_team_strength=False)
    predict_mode    : "plugin" (posterior means, frozen latent — WP001-006)
                      or "posterior" (posterior-averaged + horizon-aware)
"""

# Historical sampler: everything from WP007 through WP023's baseline refit used this.
# Kept so those work products reproduce EXACTLY; every existing runner imports it by
# this name. Do not change it.
SAMPLER_LEGACY = dict(draws=600, tune=600, chains=4, target_accept=0.9,
                      nuts_sampler="numpyro")

# THE STANDARD FOR NEW WORK (set 2026-09-29).
#
# nutpie is a Rust NUTS backend for PyMC. Measured over 230 real fits on the
# 115-window grid it is **2.41x faster** (14.22 -> 5.89 CPU-hours) and the arm effect
# it produces is identical: deep vs no-deep came out -0.00086 [-0.00180, +0.00011]
# against numpyro's -0.00083 [-0.00177, +0.00013]. Same model, same target posterior,
# same latent parameters -- a faster implementation of the same algorithm, NOT a
# precision trade (see feedback-keep-full-precision).
#
# tune stays at 600, matching SAMPLER_LEGACY and, importantly, matching the nutpie
# baseline already fitted in wp023_nutpie_baseline/checkpoints/ -- so that baseline is
# reusable as-is and a new sweep only pays for its own new arms.
#
# ACCEPTED COST: nutpie converges measurably less cleanly than numpyro at this tuning --
# median rhat 1.035 vs 1.013, 69/230 windows above 1.05 (numpyro: 3), 9 above 1.10
# (numpyro: 0), concentrated in the large-window fits. Divergences are 0 for both, the
# pooled arm effect is unchanged, and per-window RPS agrees on average (mean dRPS
# +0.00002), but individual fits ARE noisier: max per-window |dRPS| 0.00101. That is
# tolerable because effects are read from 115 paired windows, where independent
# per-window noise averages out -- it would NOT be tolerable for a single-window claim.
# Raising tune to 1000 was the alternative; it was measured to be worth doing but has
# not been benchmarked, and would have invalidated the existing baseline.
SAMPLER = dict(draws=600, tune=600, chains=4, target_accept=0.9,
               nuts_sampler="nutpie")


# WP023: a screening budget, ~2x faster. Pair with --time-grid-step 2.
# NOT IN USE. The decision (2026-09-29) is to keep full posteriors and the full
# latent grid everywhere, because the effects being chased (~0.0008 RPS) are close
# to the noise this introduces (up to 0.00042). Kept only to record the measurement.
# Measured cost: max per-window |dRPS| 0.00042 (44% of the 115-window MDE), 0
# divergences, rhat unchanged. Use for "is this idea worth pursuing"; use SAMPLER
# for "is this effect real". See work_products/PERFORMANCE.md for what was tried
# and rejected -- do not re-test vectorized chains or a lower target_accept.
SAMPLER_FAST = dict(draws=300, tune=300, chains=4, target_accept=0.9)
FAST_GRID_STEP = 2

# --- rung-by-rung building blocks --------------------------------------------
LEVEL = dict(                       # L2: identifiable level + HFA priors the data can move
    center_team_strength=True, use_intercept=True,
    home_mu=0.15, home_sd=0.15, home_adv_sd=0.06,
)
RW = dict(                          # L3: random-walk strengths, no static level
    dynamics="random_walk", sigma_att=0.05, sigma_def=0.05, rw_init_scale=0.3,
)
PROMO = dict(use_promoted_effect=True, promoted_effect_sd=0.25)       # L4
XGCH = dict(use_xg_channel=True, xg_channel_weight=0.5)               # L5

ARMS = {
    "L0_baseline":  dict(use_xg=False, use_dc=False, overrides={}, predict_mode="plugin"),
    "L1_posterior": dict(use_xg=False, use_dc=False, overrides={}, predict_mode="posterior"),
    "L2_level":     dict(use_xg=False, use_dc=False, overrides={**LEVEL}, predict_mode="posterior"),
    "L3_randomwalk": dict(use_xg=False, use_dc=False, overrides={**LEVEL, **RW}, predict_mode="posterior"),
    "L4_promoted":  dict(use_xg=False, use_dc=False, overrides={**LEVEL, **RW, **PROMO}, predict_mode="posterior"),
    "L5_xgchannel": dict(use_xg=False, use_dc=False, overrides={**LEVEL, **RW, **PROMO, **XGCH}, predict_mode="posterior"),
    "L6_dixoncoles": dict(use_xg=False, use_dc=True, overrides={**LEVEL, **RW, **PROMO, **XGCH}, predict_mode="posterior"),
    # side arms
    "S1_xgweight1":  dict(use_xg=False, use_dc=False,
                          overrides={**LEVEL, **RW, **PROMO, **XGCH, "xg_channel_weight": 1.0},
                          predict_mode="posterior"),
    "S2_dc_only_fixed": dict(use_xg=False, use_dc=True, overrides={}, predict_mode="plugin"),
}

LADDER = ["L0_baseline", "L1_posterior", "L2_level", "L3_randomwalk", "L4_promoted",
          "L5_xgchannel", "L6_dixoncoles"]


def sampler_args(sampler=None):
    """CLI flags for run_cv_window.py from a SAMPLER dict.

    Use this in new runners instead of hand-writing the flags: `nuts_sampler` is
    easy to forget, and forgetting it silently falls back to numpyro, which makes
    the arm non-comparable to a nutpie baseline. The sampler shift is ~10% of the
    effects this project measures, so a silent mismatch is a real hazard, not a
    tidiness issue.

        cmd = [sys.executable, SCRIPT, ..., *sampler_args()]
    """
    s = SAMPLER if sampler is None else sampler
    return ["--draws", str(s["draws"]), "--tune", str(s["tune"]),
            "--chains", str(s["chains"]), "--target-accept", str(s["target_accept"]),
            "--nuts-sampler", s.get("nuts_sampler", "numpyro")]
