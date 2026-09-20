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

    # --- lineup-quality covariate (WP008) ---
    # log-ratio of today's starting-XI quality vs. that team's own recent
    # normal (see football_model.features.lineup_features) — zero, not
    # missing, when lineup data isn't available for a match, so this is
    # always safe to read even when use_lineup_xg=False.
    lineup_dev_home: np.ndarray = None    # shape=(n_obs,)
    lineup_dev_away: np.ndarray = None    # shape=(n_obs,)

    # --- lineup-continuity covariate (WP013) ---
    # Each side's OWN standardised defence-unit continuity (how usual its
    # keeper + back line is; see football_model.features.continuity_features).
    # A side's continuity is what affects its OPPONENT's scoring, so the
    # model reads `defence_cont_away` for the home side's theta and
    # `defence_cont_home` for the away side's. Zero (the neutral, mean value
    # after standardising) when unavailable, so always safe to read.
    defence_cont_home: np.ndarray = None  # shape=(n_obs,)
    defence_cont_away: np.ndarray = None  # shape=(n_obs,)

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
    # of being the one global innovation SD every team shares.
    use_per_team_sigma: bool = False

    # -------------------------
    # Home advantage
    # -------------------------
    home_mu: float = 0.13
    home_sd: float = 0.03
    home_adv_sd: float = 0.02       # per-team home advantage variation

    # Multi-league home advantage (WP011): how much a league's own average
    # home advantage is allowed to differ from the shared global mean
    # (home_mu). Unused by build_model's single-league path; only
    # build_multileague_model reads this.
    home_mu_league_sd: float = 0.03

    # -------------------------
    # Opponent-adjusted xG
    # -------------------------
    use_opponent_adjusted_xG: bool = False  # adjust xG trust based on opponent quality
    xG_adjustment_strength: float = 0.3     # how much to adjust (0=none, 1=full adjustment)

    # -------------------------
    # Lineup-quality covariate (WP008)
    # -------------------------
    use_lineup_xg: bool = False     # add beta_lineup * lineup_dev to theta (starting-XI xG/xA vs. team's own normal)

    # -------------------------
    # Lineup-continuity covariate (WP013)
    # -------------------------
    use_continuity: bool = False    # add beta_continuity * (OPPONENT's defence continuity, standardised) to theta
    continuity_beta_sd: float = 0.1  # prior SD of beta_continuity (Normal(0, sd): sign is not forced)

    # -------------------------
    # Model options
    # -------------------------
    center_team_strength: bool = True  # center team strengths at each time point
    soft_center_team_strength: bool = True  # if not centering, softly pin per-time means near 0 to avoid drift
    soft_center_sd: float = 1.0        # strength of soft-centering (larger = weaker)
    use_xG: bool = False            # include xG as weighted feature (default: False)
    clip_theta: float = 2.0         # soft clip parameter
    init_scale: float = 0.2         # initial scale for AR1 / team strengths

    # -------------------------
    # Multi-season handling
    # -------------------------
    season_start_sigma_mult: float = 3.0  # multiplier on sigma_att/sigma_def during that window

    # -------------------------
    # Dixon-Coles low-score correlation correction
    # -------------------------
    use_dixon_coles: bool = False   # correct 0-0/1-0/0-1/1-1 for home/away goal correlation
    rho_dc_sd: float = 0.1          # prior scale for the learned correlation parameter