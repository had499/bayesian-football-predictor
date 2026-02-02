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
    likelihood: str = "negbin"      # "poisson" or "negbin"
    use_xG: bool = False            # include xG as weighted feature (default: False)
    clip_theta: float = 2.0         # soft clip parameter
    init_scale: float = 0.2         # initial scale for AR1 / team strengths