from dataclasses import dataclass
import numpy as np
import pandas as pd
import pytensor.tensor as pt

@dataclass(frozen=True)
class ModelData:
    # --- basic metadata ---
    n_teams: int
    n_matches: int
    n_time: int

    # --- index arrays (int32) ---
    t_idx: np.ndarray          # shape=(n_obs,)
    team_idx: np.ndarray       # shape=(n_obs,)
    opp_idx: np.ndarray        # shape=(n_obs,)
    match_idx: np.ndarray      # shape=(n_obs,)
    home: np.ndarray           # shape=(n_obs,)

    # --- observed outputs ---
    goals_home: np.ndarray     # shape=(n_obs,)
    goals_away: np.ndarray     # shape=(n_obs,)
    xG_home: np.ndarray        # shape=(n_obs,)
    xG_away: np.ndarray        # shape=(n_obs,)
    
    # --- team mapping ---
    team_mapping: dict = None  # maps team name -> index

    # --- multi-season bookkeeping ---
    active_mask: np.ndarray = None        # shape=(n_time, n_teams); 1 if team is in the league at time t, else 0
    season_start_mask: np.ndarray = None  # shape=(n_time,); 1 if t falls in a season's opening window

    # --- realised in-match xG, one per side (WP007 xG measurement channel) ---
    # Distinct from xG_home/xG_away above, which are leak-free ROLLING BASELINES
    # used as a predictor. These are the match's own outcome-side xG: an
    # observation for training rows, never a predictor for the row itself.
    xG_obs_home: np.ndarray = None        # shape=(n_obs,)
    xG_obs_away: np.ndarray = None        # shape=(n_obs,)

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    # -------------------------
    # AR1 / latent team dynamics
    # -------------------------
    sigma_att: float = 0.008        # standard deviation of attack random walk (balanced: flexibility + convergence)
    sigma_def: float = 0.008        # standard deviation of defense random walk (balanced: flexibility + convergence)
    rho_att_alpha: float = 29.0     # beta prior alpha for attack AR1 (Beta(29,1) → rho~0.97)
    rho_att_beta: float = 1.0       # beta prior beta for attack AR1
    rho_def_alpha: float = 29.0     # beta prior alpha for defense AR1 (Beta(29,1) → rho~0.97)
    rho_def_beta: float = 1.0       # beta prior beta for defense AR1

    # -------------------------
    # Per-team sigma (WP006, partial pooling)
    # -------------------------
    # When True, sigma_att/sigma_def above are reinterpreted as the scale of
    # a population-level HalfNormal hyperprior that each team's own sigma is
    # partially pooled toward (ar1_hierarchical_sigma in priors.py), instead
    # of being the one global innovation SD every team shares. Only affects
    # the standard (non-form-decomposition) AR1 branch in build_model.
    use_per_team_sigma: bool = False

    # -------------------------
    # Form decomposition (ability + form)
    # -------------------------
    use_form_decomposition: bool = False  # split into ability (long-term) + form (short-term)
    sigma_att_ability: float = 0.003      # long-term ability: very stable
    sigma_def_ability: float = 0.003
    rho_ability_alpha: float = 49.0       # Beta(49,1) → rho~0.98 (very persistent)
    rho_ability_beta: float = 1.0
    sigma_att_form: float = 0.015         # short-term form: more volatile
    sigma_def_form: float = 0.015
    rho_form_alpha: float = 8.0           # Beta(8,1) → rho~0.89 (mean-reverting)
    rho_form_beta: float = 1.0

    # -------------------------
    # Home advantage
    # -------------------------
    home_mu: float = 0.13
    home_sd: float = 0.03
    home_adv_sd: float = 0.02       # per-team home advantage variation

    # -------------------------
    # Match effect
    # -------------------------
    sigma_match: float = 0.10

    # -------------------------
    # Goal likelihood
    # -------------------------
    goal_alpha: float = 2.0         # NegativeBinomial dispersion (data shows minimal overdispersion)

    # -------------------------
    # xG likelihood
    # -------------------------
    sigma_xG_mu: float = 0.1
    sigma_xG_sd: float = 0.05

    # -------------------------
    # Opponent-adjusted xG
    # -------------------------
    use_opponent_adjusted_xG: bool = False  # adjust xG trust based on opponent quality
    xG_adjustment_strength: float = 0.3     # how much to adjust (0=none, 1=full adjustment)

    # -------------------------
    # Model options
    # -------------------------
    center_team_strength: bool = True  # center team strengths at each time point
    soft_center_team_strength: bool = True  # if not centering, softly pin per-time means near 0 to avoid drift
    soft_center_sd: float = 1.0        # strength of soft-centering (larger = weaker)
    likelihood: str = "negbin"      # "poisson" or "negbin"
    use_xG: bool = False            # include xG as weighted feature (default: False)
    clip_theta: float = 2.0         # soft clip parameter
    init_scale: float = 0.2         # initial scale for AR1 / team strengths

    # -------------------------
    # Multi-season handling
    # -------------------------
    season_start_window: int = 5          # rounds at the start of each season treated as high-uncertainty
    season_start_sigma_mult: float = 3.0  # multiplier on sigma_att/sigma_def during that window

    # -------------------------
    # Dixon-Coles low-score correlation correction
    # -------------------------
    use_dixon_coles: bool = False   # correct 0-0/1-0/0-1/1-1 for home/away goal correlation
    rho_dc_sd: float = 0.1          # prior scale for the learned correlation parameter

    # -------------------------
    # WP007 structural changes (all opt-in; defaults reproduce earlier configs)
    # -------------------------
    # Latent-strength dynamics. "ar1_static" = per-team static att_0 + a
    # fast-reverting AR(1) around it (everything up to WP006). "random_walk" =
    # no static term, rho fixed at 1: a team's level itself wanders. Real
    # year-on-year attack change is SD ~0.12 (noise-corrected) vs ~0.04 the
    # prior-implied change of the AR(1)+static structure, and no single sigma
    # knob fixes that — the reversion to a static mean is the constraint.
    # Only affects the standard (non-form-decomposition) branch.
    dynamics: str = "ar1_static"    # "ar1_static" | "random_walk"
    rw_init_scale: float = 0.3      # SD of a team's t=0 level under random_walk

    # A team's level shifts by a learned amount in the round it (re)enters the
    # league (promoted sides are ~-0.37 attack / ~-0.26 defence vs continuing
    # teams; N(0, init) says "average"). One shared shift each for attack and
    # defence, prior N(0, promoted_effect_sd).
    use_promoted_effect: bool = False
    promoted_effect_sd: float = 0.25

    # Explicit log-rate intercept. With hard centering (center_team_strength
    # =True) attack/defence sum to zero, leaving the overall goal level with
    # NO free parameter — the default config implied 1.14 home / 1.00 away vs
    # ~1.54 / ~1.26 actual. Prior centred on log(1.35) league-average away rate.
    use_intercept: bool = False
    intercept_mu: float = 0.30
    intercept_sd: float = 0.25

    # xG as a second, tempered measurement channel on the same latent rate
    # (see components.xg_measurement_channel). Needs ModelData.xG_obs_*.
    use_xg_channel: bool = False
    xg_channel_weight: float = 0.5
    xg_channel_shape_alpha: float = 2.0
    xg_channel_shape_beta: float = 0.5

