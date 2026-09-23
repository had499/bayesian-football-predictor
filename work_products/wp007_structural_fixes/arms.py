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

SAMPLER = dict(draws=600, tune=600, chains=4, target_accept=0.9)

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
