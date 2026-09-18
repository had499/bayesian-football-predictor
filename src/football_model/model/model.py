import numpy as np
import pytensor.tensor as pt

from football_model.model.priors import *
from football_model.model.components import *
from football_model.types.model_data import ModelData, ModelConfig

default_config = ModelConfig()

def build_model(data: ModelData , config: ModelConfig = default_config):

    with pm.Model() as model:

        # --- Priors: single AR(1) process per team, for attack and defence ---
        rho_att = rho_prior("att", config.rho_att_alpha, config.rho_att_beta)
        rho_def = rho_prior("def", config.rho_def_alpha, config.rho_def_beta)

        if config.use_per_team_sigma:
            # Partial pooling (WP006): each team gets its own innovation SD,
            # shrunk toward a shared population value — instead of forcing
            # every team through one global sigma (WP005 found loosening
            # that single global value doesn't recover resolution; this
            # tests whether volatility varying BY TEAM does, which a global
            # knob structurally cannot represent).
            sigma_att, _ = ar1_hierarchical_sigma("att", data.n_teams, config.sigma_att)
            sigma_def, _ = ar1_hierarchical_sigma("def", data.n_teams, config.sigma_def)
        else:
            sigma_att = pm.HalfNormal("sigma_att", config.sigma_att)
            sigma_def = pm.HalfNormal("sigma_def", config.sigma_def)

        att_0 = team_strength_prior("att_0", data.n_teams, scale=config.init_scale)
        def_0 = team_strength_prior("def_0", data.n_teams, scale=config.init_scale)

        # --- Latent dynamics ---
        att_rw = ar1_team_process(
            "att_rw",
            data.n_time,
            data.n_teams,
            sigma_att,
            rho_att,
            active_mask=data.active_mask,
            season_start_mask=data.season_start_mask,
            season_start_sigma_mult=config.season_start_sigma_mult,
        )
        def_rw = ar1_team_process(
            "def_rw",
            data.n_time,
            data.n_teams,
            sigma_def,
            rho_def,
            active_mask=data.active_mask,
            season_start_mask=data.season_start_mask,
            season_start_sigma_mult=config.season_start_sigma_mult,
        )
        if config.center_team_strength == True:
            attack = centered_over_teams(att_0 + att_rw, "attack", active_mask=data.active_mask)
            defense = centered_over_teams(def_0 + def_rw, "defence", active_mask=data.active_mask)
        else:
            attack = pm.Deterministic('attack', att_0 + att_rw)
            defense = pm.Deterministic('defence', def_0 + def_rw)

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

        # --- Optional: lineup-quality covariate (WP008) ---
        # data.lineup_dev_home/away is the log-ratio of today's confirmed
        # starting-XI quality vs. that team's own recent normal (built by
        # football_model.features.lineup_features) — an ADDITIVE correction
        # on top of attack/defence, not a replacement for either; see
        # WP008's README for why. Zero (not missing) when lineup data isn't
        # available for a match, so this is always safe to add even for
        # rows that predate the feature's data coverage.
        if config.use_lineup_xg:
            lineup_dev_home_data = pm.Data('lineup_dev_home', data.lineup_dev_home)
            lineup_dev_away_data = pm.Data('lineup_dev_away', data.lineup_dev_away)
            beta_lineup = pm.HalfNormal('beta_lineup', sigma=0.1)
            lineup_contribution_home = beta_lineup * lineup_dev_home_data
            lineup_contribution_away = beta_lineup * lineup_dev_away_data
        else:
            lineup_contribution_home = 0.0
            lineup_contribution_away = 0.0

        # --- Linear predictors ---
        # NOTE: football_model.model.predict has a plain-numpy mirror of this
        # exact formula (compute_theta/predict_match_lambdas), used by both
        # scripts/run_cv_window.py and services/predictor/predictor.py to
        # make predictions from posterior means/samples after training —
        # PyTensor code here can't run outside a pm.Model context, so it's
        # kept in sync by hand. If this formula changes, update that module too.
        theta_home = (
            xG_contribution_home
            + lineup_contribution_home
            + attack[data.t_idx, data.team_idx]
            - defense[data.t_idx, data.opp_idx]
            + home_adv[data.team_idx] * data.home
        )

        theta_away = (
            xG_contribution_away
            + lineup_contribution_away
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

    return model


def build_multileague_model(leagues: dict, config: ModelConfig = default_config):
    """WP011: joint model across several leagues, sharing rho_att/rho_def/
    sigma_att/sigma_def (one number each, estimated from ALL leagues'
    matches at once, instead of only ~20 EPL teams over 6 seasons) and home
    advantage's league-level mean (league_home_advantage_prior in
    priors.py), while each league keeps its own completely independent
    AR(1) attack/defence trajectories — teams in different leagues never
    share a match, so there's nothing to pool at the team level, only at
    the hyperparameters every team (in every league) gets shrunk toward.

    `leagues`: {league_name: ModelData}, one entry per league, each built by
    an ordinary `prepare_model_data(df_for_that_league, max_round=...)` call
    — every league's ModelData is exactly the same shape of object
    build_model already consumes, with its own local team-index space and
    its own time axis starting at 0 (NOT a shared global calendar across
    leagues — see WP011's README for why a shared/disjoint time axis would
    be wasteful and isn't used). One of the entries is expected to be EPL,
    but nothing here treats any league specially at training time; only
    evaluation code (predict.py, run_cv_window.py) cares which one is EPL,
    by reading that league's own attack/defence/home_adv posterior slices
    and its own ModelData exactly as it always has for the single-league
    model — no new prediction-side code exists or is needed for this.

    Out of scope for this first pass (raises if requested): use_lineup_xg,
    use_per_team_sigma, use_opponent_adjusted_xG — WP011 only covers the
    base hierarchical model, per its README's scope discipline.
    """
    if config.use_lineup_xg:
        raise NotImplementedError("build_multileague_model does not support use_lineup_xg (WP011 scope: base hierarchical model only)")
    if config.use_per_team_sigma:
        raise NotImplementedError("build_multileague_model does not support use_per_team_sigma (WP011 scope: base hierarchical model only)")
    if config.use_opponent_adjusted_xG:
        raise NotImplementedError("build_multileague_model does not support use_opponent_adjusted_xG (WP011 scope: base hierarchical model only)")

    league_names = list(leagues.keys())
    league_data = list(leagues.values())

    with pm.Model() as model:

        # --- Shared hyperparameters, fit from every league's data at once ---
        rho_att = rho_prior("att", config.rho_att_alpha, config.rho_att_beta)
        rho_def = rho_prior("def", config.rho_def_alpha, config.rho_def_beta)
        sigma_att = pm.HalfNormal("sigma_att", config.sigma_att)
        sigma_def = pm.HalfNormal("sigma_def", config.sigma_def)

        _, _, _, home_adv_by_league = league_home_advantage_prior(
            league_names,
            [d.n_teams for d in league_data],
            config.home_mu, config.home_sd,
            config.home_mu_league_sd, config.home_adv_sd,
        )

        beta_xG = pm.HalfNormal("beta_xG", sigma=0.1) if config.use_xG else None

        lambda_home_parts, lambda_away_parts = [], []
        goals_home_parts, goals_away_parts = [], []

        for l, name in enumerate(league_names):
            data = league_data[l]
            home_adv_l = home_adv_by_league[l]

            att_0 = team_strength_prior(f"att_0_{name}", data.n_teams, scale=config.init_scale)
            def_0 = team_strength_prior(f"def_0_{name}", data.n_teams, scale=config.init_scale)

            att_rw = ar1_team_process(
                f"att_rw_{name}", data.n_time, data.n_teams, sigma_att, rho_att,
                active_mask=data.active_mask, season_start_mask=data.season_start_mask,
                season_start_sigma_mult=config.season_start_sigma_mult,
            )
            def_rw = ar1_team_process(
                f"def_rw_{name}", data.n_time, data.n_teams, sigma_def, rho_def,
                active_mask=data.active_mask, season_start_mask=data.season_start_mask,
                season_start_sigma_mult=config.season_start_sigma_mult,
            )

            if config.center_team_strength:
                attack_l = centered_over_teams(att_0 + att_rw, f"attack_{name}", active_mask=data.active_mask)
                defense_l = centered_over_teams(def_0 + def_rw, f"defence_{name}", active_mask=data.active_mask)
            else:
                attack_l = pm.Deterministic(f"attack_{name}", att_0 + att_rw)
                defense_l = pm.Deterministic(f"defence_{name}", def_0 + def_rw)

            if (not config.center_team_strength) and getattr(config, "soft_center_team_strength", False):
                sc_sd = float(getattr(config, "soft_center_sd", 1.0))
                pm.Potential(
                    f"soft_center_attack_{name}",
                    pm.logp(pm.Normal.dist(mu=0.0, sigma=sc_sd), masked_mean_over_teams(attack_l, data.active_mask)).sum(),
                )
                pm.Potential(
                    f"soft_center_defence_{name}",
                    pm.logp(pm.Normal.dist(mu=0.0, sigma=sc_sd), masked_mean_over_teams(defense_l, data.active_mask)).sum(),
                )

            if config.use_xG:
                xG_home_data = pm.Data(f"xG_home_{name}", data.xG_home)
                xG_away_data = pm.Data(f"xG_away_{name}", data.xG_away)
                xG_contribution_home = beta_xG * pm.math.log(xG_home_data + 0.01)
                xG_contribution_away = beta_xG * pm.math.log(xG_away_data + 0.01)
            else:
                xG_contribution_home = 0.0
                xG_contribution_away = 0.0

            # NOTE: identical formula to build_model's theta_home/theta_away
            # (minus lineup, out of scope here) — same "single source of
            # truth" rule applies: football_model.model.predict's
            # compute_theta/predict_match_lambdas/predict_rows already
            # mirror exactly this, unchanged, because a trained league's
            # slice of this model (its own attack_<name>/defence_<name>/
            # home_adv_<name> posterior + its own ModelData) is structurally
            # identical to what build_model always produced. If this formula
            # ever changes, predict.py needs updating too, same as always.
            theta_home_l = (
                xG_contribution_home
                + attack_l[data.t_idx, data.team_idx]
                - defense_l[data.t_idx, data.opp_idx]
                + home_adv_l[data.team_idx] * data.home
            )
            theta_away_l = (
                xG_contribution_away
                + attack_l[data.t_idx, data.opp_idx]
                - defense_l[data.t_idx, data.team_idx]
            )
            theta_home_l = soft_clip(theta_home_l, config.clip_theta)
            theta_away_l = soft_clip(theta_away_l, config.clip_theta)

            lambda_home_l = pm.Deterministic(f"lambda_home_{name}", pm.math.exp(theta_home_l))
            lambda_away_l = pm.Deterministic(f"lambda_away_{name}", pm.math.exp(theta_away_l))

            pm.Poisson(f"goals_home_{name}", mu=lambda_home_l, observed=data.goals_home)
            pm.Poisson(f"goals_away_{name}", mu=lambda_away_l, observed=data.goals_away)

            lambda_home_parts.append(lambda_home_l)
            lambda_away_parts.append(lambda_away_l)
            goals_home_parts.append(np.asarray(data.goals_home))
            goals_away_parts.append(np.asarray(data.goals_away))

        # --- Optional: Dixon-Coles low-score correlation correction ---
        # ONE shared rho_dc across every league's matches, exactly as
        # use_dixon_coles is one shared correction in the single-league
        # model — concatenate every league's lambda/goals into one vector
        # so dixon_coles_adjustment scores the whole joint likelihood.
        if config.use_dixon_coles:
            all_lambda_home = pt.concatenate(lambda_home_parts)
            all_lambda_away = pt.concatenate(lambda_away_parts)
            all_goals_home = np.concatenate(goals_home_parts)
            all_goals_away = np.concatenate(goals_away_parts)
            dixon_coles_adjustment(
                all_lambda_home, all_lambda_away, all_goals_home, all_goals_away, config.rho_dc_sd
            )

    return model
