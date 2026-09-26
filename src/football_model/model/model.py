from football_model.model.priors import *
from football_model.model.components import *
from football_model.types.model_data import ModelData, ModelConfig

default_config = ModelConfig()

def build_model(data: ModelData , config: ModelConfig = default_config):

    if config.use_form_decomposition and (
        config.dynamics != "ar1_static" or config.use_promoted_effect
    ):
        raise ValueError(
            "dynamics='random_walk' and use_promoted_effect only apply to the "
            "standard (non-form-decomposition) branch"
        )
    if config.use_xg_channel and data.xG_obs_home is None:
        raise ValueError("use_xg_channel needs ModelData.xG_obs_home/away (prepare_model_data fills them)")

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
                active_mask=data.active_mask,
                season_start_mask=data.season_start_mask,
                season_start_sigma_mult=config.season_start_sigma_mult,
            )
            def_ability = ar1_team_process(
                "def_ability",
                data.n_time,
                data.n_teams,
                sigma_def_ability,
                rho_def_ability,
                active_mask=data.active_mask,
                season_start_mask=data.season_start_mask,
                season_start_sigma_mult=config.season_start_sigma_mult,
            )

            # Form processes (volatile short-term fluctuations)
            att_form = ar1_team_process(
                "att_form",
                data.n_time,
                data.n_teams,
                sigma_att_form,
                rho_att_form,
                active_mask=data.active_mask,
                season_start_mask=data.season_start_mask,
                season_start_sigma_mult=config.season_start_sigma_mult,
            )
            def_form = ar1_team_process(
                "def_form",
                data.n_time,
                data.n_teams,
                sigma_def_form,
                rho_def_form,
                active_mask=data.active_mask,
                season_start_mask=data.season_start_mask,
                season_start_sigma_mult=config.season_start_sigma_mult,
            )

            # Total strength = initial + ability + form
            if config.center_team_strength:
                attack = centered_over_teams(att_0 + att_ability + att_form, "attack", active_mask=data.active_mask)
                defense = centered_over_teams(def_0 + def_ability + def_form, "defence", active_mask=data.active_mask)
            else:
                attack = pm.Deterministic('attack', att_0 + att_ability + att_form)
                defense = pm.Deterministic('defence', def_0 + def_ability + def_form)
        else:
            # Standard single latent process per team/side. Two dynamics:
            #   "ar1_static"  (default, everything up to WP006): att_0 + AR(1)
            #                 that reverts toward it (rho ~ Beta(29,1)).
            #   "random_walk" (WP007): no static term, rho == 1 — the level
            #                 itself wanders. See ModelConfig.dynamics.
            random_walk = config.dynamics == "random_walk"
            if config.dynamics not in ("ar1_static", "random_walk"):
                raise ValueError(f"unknown dynamics {config.dynamics!r}")

            if random_walk:
                rho_att = rho_def = 1.0
            else:
                rho_att = rho_prior("att", config.rho_att_alpha, config.rho_att_beta)
                rho_def = rho_prior("def", config.rho_def_alpha, config.rho_def_beta)

            if config.use_per_team_sigma:
                # Partial pooling (WP006): each team gets its own innovation
                # SD, shrunk toward a shared population value — instead of
                # forcing every team through one global sigma (WP005 found
                # loosening that single global value doesn't recover
                # resolution; this tests whether volatility varying BY TEAM
                # does, which a global knob structurally cannot represent).
                sigma_att, _ = ar1_hierarchical_sigma("att", data.n_teams, config.sigma_att)
                sigma_def, _ = ar1_hierarchical_sigma("def", data.n_teams, config.sigma_def)
            else:
                sigma_att = pm.HalfNormal("sigma_att", config.sigma_att)
                sigma_def = pm.HalfNormal("sigma_def", config.sigma_def)

            if config.use_promoted_effect:
                promo_att = pm.Normal("promo_att", 0.0, config.promoted_effect_sd)
                promo_def = pm.Normal("promo_def", 0.0, config.promoted_effect_sd)
            else:
                promo_att = promo_def = 0.0

            # --- Latent dynamics ---
            # init_scale for the walk's own t=0 value: 0.2 (the function's
            # default, unchanged) under ar1_static; rw_init_scale under
            # random_walk, where this IS the team's whole level prior.
            walk_kwargs = dict(
                active_mask=data.active_mask,
                season_start_mask=data.season_start_mask,
                season_start_sigma_mult=config.season_start_sigma_mult,
            )
            if random_walk:
                walk_kwargs["init_scale"] = config.rw_init_scale
            att_rw = ar1_team_process(
                "att_rw", data.n_time, data.n_teams, sigma_att, rho_att,
                entry_shift=promo_att, **walk_kwargs,
            )
            def_rw = ar1_team_process(
                "def_rw", data.n_time, data.n_teams, sigma_def, rho_def,
                entry_shift=promo_def, **walk_kwargs,
            )
            if random_walk:
                attack_raw, defense_raw = att_rw, def_rw
            else:
                att_0 = team_strength_prior("att_0", data.n_teams, scale=config.init_scale)
                def_0 = team_strength_prior("def_0", data.n_teams, scale=config.init_scale)
                attack_raw, defense_raw = att_0 + att_rw, def_0 + def_rw

            if config.center_team_strength == True:
                # Keep the UN-centred latent state too: forecasting a round
                # beyond the trained window must roll the raw walk forward and
                # re-centre over that round's own active set (predict.py).
                pm.Deterministic("attack_raw", attack_raw)
                pm.Deterministic("defence_raw", defense_raw)
                attack = centered_over_teams(attack_raw, "attack", active_mask=data.active_mask)
                defense = centered_over_teams(defense_raw, "defence", active_mask=data.active_mask)
            else:
                attack = pm.Deterministic('attack', attack_raw)
                defense = pm.Deterministic('defence', defense_raw)

        # --- Soft identifiability constraint (only when uncentered) ---
        # Without centering, the model is weakly identified in a "drift" direction:
        # adding a constant to both attack and defence leaves (attack - defence) unchanged.
        # This can cause NUTS mass-matrix adaptation to blow up and hang.
        if (not config.center_team_strength) and getattr(config, "soft_center_team_strength", False):
            sc_sd = float(getattr(config, "soft_center_sd", 1.0))
            # Mean over currently-active teams only — otherwise relegated
            # teams' frozen values would pin the "average team = 0" anchor
            # away from the teams actually competing that round.
            pm.Potential(
                "soft_center_attack",
                pm.logp(pm.Normal.dist(mu=0.0, sigma=sc_sd), masked_mean_over_teams(attack, data.active_mask)).sum(),
            )
            pm.Potential(
                "soft_center_defence",
                pm.logp(pm.Normal.dist(mu=0.0, sigma=sc_sd), masked_mean_over_teams(defense, data.active_mask)).sum(),
            )
        
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

        # --- Optional: explicit log-rate intercept (WP007) ---
        if config.use_intercept:
            intercept = pm.Normal("intercept", config.intercept_mu, config.intercept_sd)
        else:
            intercept = 0.0

        # --- Optional: squad-absence covariate (WP012) ---
        if config.use_absence:
            if data.absent_home is None:
                raise ValueError("use_absence needs ModelData.absent_home/away")
            beta_absence = pm.Normal("beta_absence", 0.0, config.absence_sd)
            absence_home = beta_absence * pm.Data("absent_home", data.absent_home)
            absence_away = beta_absence * pm.Data("absent_away", data.absent_away)
        else:
            absence_home = absence_away = 0.0

        # --- Linear predictors ---
        # NOTE: football_model.model.predict has a plain-numpy mirror of this
        # exact formula (compute_theta/predict_match_lambdas), used by both
        # scripts/run_cv_window.py and services/predictor/predictor.py to
        # make predictions from posterior means/samples after training —
        # PyTensor code here can't run outside a pm.Model context, so it's
        # kept in sync by hand. If this formula changes, update that module too.
        theta_home = (
            intercept
            + absence_home
            + xG_contribution_home
            + attack[data.t_idx, data.team_idx]
            - defense[data.t_idx, data.opp_idx]
            + home_adv[data.team_idx] * data.home
        )

        theta_away = (
            intercept
            + absence_away
            + xG_contribution_away
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

        # --- Optional: Dixon-Coles low-score correlation correction ---
        if config.use_dixon_coles:
            dixon_coles_adjustment(
                lambda_home, lambda_away, data.goals_home, data.goals_away, config.rho_dc_sd
            )

        # --- Optional: xG as a second measurement of the same latent rate (WP007) ---
        if config.use_xg_channel:
            xg_measurement_channel(
                lambda_home, lambda_away, data.xG_obs_home, data.xG_obs_away,
                weight=config.xg_channel_weight,
                shape_alpha=config.xg_channel_shape_alpha,
                shape_beta=config.xg_channel_shape_beta,
            )

    return model
