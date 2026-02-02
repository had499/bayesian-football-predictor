from football_model.model.priors import *
from football_model.model.components import *
from football_model.types.model_data import ModelData, ModelConfig

default_config = ModelConfig()

def build_model(data: ModelData , config: ModelConfig = default_config):

    with pm.Model() as model:

        # --- Priors ---
        if config.use_form_decomposition:
            # Form decomposition: ability (long-term) + form (short-term)
            # Ability: high persistence (rho~0.98), low variance
            sigma_att_ability, rho_att_ability = ar1_hyperpriors(
                "att_ability", config.sigma_att_ability, 
                config.rho_ability_alpha, config.rho_ability_beta
            )
            sigma_def_ability, rho_def_ability = ar1_hyperpriors(
                "def_ability", config.sigma_def_ability,
                config.rho_ability_alpha, config.rho_ability_beta
            )
            
            # Form: lower persistence (rho~0.89), higher variance
            sigma_att_form, rho_att_form = ar1_hyperpriors(
                "att_form", config.sigma_att_form,
                config.rho_form_alpha, config.rho_form_beta
            )
            sigma_def_form, rho_def_form = ar1_hyperpriors(
                "def_form", config.sigma_def_form,
                config.rho_form_alpha, config.rho_form_beta
            )
            
            att_0 = team_strength_prior("att_0", data.n_teams, scale=config.init_scale)
            def_0 = team_strength_prior("def_0", data.n_teams, scale=config.init_scale)
            
            # Ability processes (stable long-term strength)
            att_ability = ar1_team_process(
                "att_ability",
                data.n_time,
                data.n_teams,
                sigma_att_ability,
                rho_att_ability,
            )
            def_ability = ar1_team_process(
                "def_ability",
                data.n_time,
                data.n_teams,
                sigma_def_ability,
                rho_def_ability,
            )
            
            # Form processes (volatile short-term fluctuations)
            att_form = ar1_team_process(
                "att_form",
                data.n_time,
                data.n_teams,
                sigma_att_form,
                rho_att_form,
            )
            def_form = ar1_team_process(
                "def_form",
                data.n_time,
                data.n_teams,
                sigma_def_form,
                rho_def_form,
            )
            
            # Total strength = initial + ability + form
            if config.center_team_strength:
                attack = centered_over_teams(att_0 + att_ability + att_form, "attack")
                defense = centered_over_teams(def_0 + def_ability + def_form, "defense")
            else:
                attack = pm.Deterministic('attack', att_0 + att_ability + att_form)
                defense = pm.Deterministic('defence', def_0 + def_ability + def_form)
        else:
            # Standard single AR(1) process
            sigma_att, rho_att = ar1_hyperpriors(
                "att", config.sigma_att, config.rho_att_alpha, config.rho_att_beta
            )
            sigma_def, rho_def = ar1_hyperpriors(
                "def", config.sigma_def, config.rho_def_alpha, config.rho_def_beta
            )

            att_0 = team_strength_prior("att_0", data.n_teams, scale=config.init_scale)
            def_0 = team_strength_prior("def_0", data.n_teams, scale=config.init_scale)

            # --- Latent dynamics ---
            att_rw = ar1_team_process(
                "att_rw",
                data.n_time,
                data.n_teams,
                sigma_att,
                rho_att,
            )
            def_rw = ar1_team_process(
                "def_rw",
                data.n_time,
                data.n_teams,
                sigma_def,
                rho_def,
            )
            if config.center_team_strength == True:
                attack = centered_over_teams(att_0 + att_rw, "attack")
                defense = centered_over_teams(def_0 + def_rw, "defense")
            else:
                attack = pm.Deterministic('attack',att_0 + att_rw)
                defense = pm.Deterministic('defence',def_0 + def_rw)
        
        home_adv = home_advantage_prior(data.n_teams,  config.home_mu, config.home_sd, config.home_adv_sd)
            
        # --- Optional: xG as weighted feature ---
        if config.use_xG:
            # Learn how much to trust xG vs team strengths
            xG_home_data = pm.Data('xG_home', data.xG_home)
            xG_away_data = pm.Data('xG_away', data.xG_away)
            
            log_xG_home = pm.math.log(xG_home_data + 0.01)
            log_xG_away = pm.math.log(xG_away_data + 0.01)
            
            # beta_xG: weight on xG feature (centered at 1.0, but can learn to down-weights)
            beta_xG = pm.HalfNormal('beta_xG', sigma=0.1)
            
            # Opponent-adjusted xG: trust xG more vs strong defenses
            if config.use_opponent_adjusted_xG:
                # Adjustment factor based on opponent defensive quality
                # Strong defense (positive defense value) → trust xG more
                # Weak defense (negative defense value) → trust xG less
                adj_factor_home = 1.0 + config.xG_adjustment_strength * defense[data.t_idx, data.opp_idx]
                adj_factor_away = 1.0 + config.xG_adjustment_strength * defense[data.t_idx, data.team_idx]
                
                # Clip adjustment to reasonable range [0.5, 1.5]
                adj_factor_home = pm.math.clip(adj_factor_home, 0.5, 1.5)
                adj_factor_away = pm.math.clip(adj_factor_away, 0.5, 1.5)
                
                # Apply opponent-adjusted beta
                beta_xG_home = beta_xG * adj_factor_home
                beta_xG_away = beta_xG * adj_factor_away
            else:
                # Standard: same beta for all matches
                beta_xG_home = beta_xG
                beta_xG_away = beta_xG
            
            xG_contribution_home = beta_xG_home * log_xG_home
            xG_contribution_away = beta_xG_away * log_xG_away
        else:
            # No xG - model relies purely on learned attack/defense strengths
            xG_contribution_home = 0.0
            xG_contribution_away = 0.0

        # --- Linear predictors ---
        theta_home = (
            xG_contribution_home
            + attack[data.t_idx, data.team_idx]
            - defense[data.t_idx, data.opp_idx]
            + home_adv[data.team_idx] * data.home
        )

        theta_away = (
            xG_contribution_away
            + attack[data.t_idx, data.opp_idx]
            - defense[data.t_idx, data.team_idx]
        )

        theta_home = soft_clip(theta_home, config.clip_theta)
        theta_away = soft_clip(theta_away, config.clip_theta)

        lambda_home = pm.Deterministic(
            "lambda_home", pm.math.exp(theta_home)
        )
        lambda_away = pm.Deterministic(
            "lambda_away", pm.math.exp(theta_away)
        )

        # --- Likelihood ---
        # Use Poisson since data shows minimal overdispersion (variance ≈ mean)
        pm.Poisson(
            "goals_home",
            mu=lambda_home,
            observed=data.goals_home,
        )
        pm.Poisson(
            "goals_away",
            mu=lambda_away,
            observed=data.goals_away,
        )

    return model
