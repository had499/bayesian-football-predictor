from football_model.model.priors import *
from football_model.model.components import *
from football_model.model.components import avail_offset_terms  # noqa: E402
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
    if config.time_varying_level and config.xg_kappa_by_season:
        raise ValueError(
            "time_varying_level and xg_kappa_by_season are collinear (both mediate the "
            "latent->observation level); enable at most one"
        )
    if config.time_varying_level and not config.center_team_strength:
        raise ValueError(
            "time_varying_level needs hard centering (center_team_strength=True): with soft "
            "centering the unpinned mean is a second representation of the same level drift"
        )
    if config.split_defence and not config.use_xg_channel:
        raise ValueError(
            "split_defence needs use_xg_channel: shot-stopping is identified by the GAP "
            "between goals conceded and xG conceded, so without the xG channel the two "
            "defensive latents are not separately identified"
        )
    if config.use_xg_channel and data.xG_obs_home is None:
        raise ValueError("use_xg_channel needs ModelData.xG_obs_home/away (prepare_model_data fills them)")

    if config.use_shot_channel and data.shots_home is None:
        raise ValueError("use_shot_channel needs ModelData.shots_home/away (WP019's prepare_data fills them)")

    if config.use_market_channel and data.market_lh is None:
        raise ValueError("use_market_channel needs ModelData.market_lh/la "
                         "(WP039's prepare_data fills them)")

    if config.use_deep_channel and data.deep_home is None:
        raise ValueError("use_deep_channel needs ModelData.deep_home/away (WP020's prepare_data fills them)")

    if config.use_elo_anchor and data.entry_elo_z is None:
        raise ValueError("use_elo_anchor needs ModelData.entry_elo_z (WP025's prepare_data fills it)")

    if config.use_elo_entry and data.entry_elo_z is None:
        raise ValueError("use_elo_entry needs ModelData.entry_elo_z (WP025's prepare_data fills it)")

    if config.use_time_decay and data.match_day is None:
        raise ValueError("use_time_decay needs ModelData.match_day (prepare_model_data fills it from datetime)")

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
                suppress = defense
            else:
                attack = pm.Deterministic('attack', att_0 + att_ability + att_form)
                defense = pm.Deterministic('defence', def_0 + def_ability + def_form)
            suppress = defense
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
            # WP025: with Elo scaling, these become COEFFICIENTS on standardised Elo
            # rather than fixed offsets, so they are learned even when
            # use_promoted_effect is off (the champion leaves it off, and WP007's
            # global version measured WORSE at +0.00285).
            revert_att = revert_def = None
            if config.use_elo_anchor:
                anchor_att = pm.Normal("elo_anchor_att", 0.0, config.elo_anchor_sd)
                anchor_def = pm.Normal("elo_anchor_def", 0.0, config.elo_anchor_sd)
                revert_att = anchor_att * data.entry_elo_z
                revert_def = anchor_def * data.entry_elo_z
            entry_scale = None
            if config.use_elo_entry:
                entry_scale = data.entry_elo_z
                promo_att = pm.Normal("elo_entry_att", 0.0, config.elo_entry_sd)
                promo_def = pm.Normal("elo_entry_def", 0.0, config.elo_entry_sd)

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
            # WP037: optionally CORRELATE the attack and defence walks. The prior
            # currently asserts they are independent, which the data contradicts --
            # corr(attack, defence) across teams is +0.79 [p5 +0.72, p95 +0.86 over
            # 271 rounds], because good sides are good at both ends. Correlating the
            # standardised innovations via a Cholesky factor keeps the non-centered
            # parameterisation intact (no MvNormal, no extra scan cost):
            #     z_def = r*z_att + sqrt(1-r^2)*z_def_raw
            # so z_def is still marginally N(0,1) and r is the only new parameter.
            # At r=0 this is bit-identical to the independent walks.
            z_att = z_def = None
            if config.use_ad_correlation:
                ad_corr = pm.Uniform("ad_corr", -0.98, 0.98,
                                     initval=config.ad_corr_init)
                z_att = pm.Normal("att_rw_std", 0, 1,
                                  shape=(data.n_time, data.n_teams))
                z_def_raw = pm.Normal("def_rw_std", 0, 1,
                                      shape=(data.n_time, data.n_teams))
                z_def = ad_corr * z_att + pt.sqrt(1.0 - ad_corr ** 2) * z_def_raw
            att_rw = ar1_team_process(
                "att_rw", data.n_time, data.n_teams, sigma_att, rho_att,
                entry_shift=promo_att, entry_scale=entry_scale,
                revert_to=revert_att, z=z_att, **walk_kwargs,
            )
            def_rw = ar1_team_process(
                "def_rw", data.n_time, data.n_teams, sigma_def, rho_def,
                entry_shift=promo_def, entry_scale=entry_scale,
                revert_to=revert_def, z=z_def, **walk_kwargs,
            )
            # WP016: `def_rw` is re-read as CHANCE SUPPRESSION, and a second,
            # tighter walk carries SHOT-STOPPING. Its sigma prior is scaled down
            # because only ~a fifth of observed save variation persists season to
            # season (r=+0.22) against r=+0.65 for chance suppression.
            if config.split_defence:
                sigma_stop = pm.HalfNormal("sigma_stop", config.sigma_def * config.stop_sigma_scale)
                stop_rw = ar1_team_process(
                    "stop_rw", data.n_time, data.n_teams, sigma_stop, rho_def, **walk_kwargs,
                )
            else:
                stop_rw = None

            if random_walk:
                attack_raw, defense_raw = att_rw, def_rw
            else:
                att_0 = team_strength_prior("att_0", data.n_teams, scale=config.init_scale)
                def_0 = team_strength_prior("def_0", data.n_teams, scale=config.init_scale)
                attack_raw, defense_raw = att_0 + att_rw, def_0 + def_rw

            # WP016: when split, `defense_raw` is suppression only and the total
            # defence is suppression + stopping. `defence` keeps its original
            # meaning so predict.py / the harness / the service are untouched —
            # the split changes only what each observation channel below sees.
            suppress_raw = defense_raw
            if config.split_defence:
                defense_raw = suppress_raw + stop_rw

            if config.center_team_strength == True:
                # Keep the UN-centred latent state too: forecasting a round
                # beyond the trained window must roll the raw walk forward and
                # re-centre over that round's own active set (predict.py).
                pm.Deterministic("attack_raw", attack_raw)
                pm.Deterministic("defence_raw", defense_raw)
                attack = centered_over_teams(attack_raw, "attack", active_mask=data.active_mask)
                defense = centered_over_teams(defense_raw, "defence", active_mask=data.active_mask)
                suppress = (centered_over_teams(suppress_raw, "suppress", active_mask=data.active_mask)
                            if config.split_defence else defense)
            else:
                attack = pm.Deterministic('attack', attack_raw)
                defense = pm.Deterministic('defence', defense_raw)
                suppress = pm.Deterministic('suppress', suppress_raw) if config.split_defence else defense

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
        
        # --- Home advantage: per-team deviation, optionally around a DRIFTING mean ---
        if config.time_varying_hfa:
            hfa_t = global_walk("hfa_t", data.n_time, config.level_walk_sigma,
                                init_mu=config.home_mu, init_sd=config.home_sd)
            hfa_dev = pm.HalfNormal("home_sd", config.home_adv_sd) * pm.Normal(
                "home_adv_raw", 0.0, 1.0, shape=data.n_teams)
            pm.Deterministic("home_adv_dev", hfa_dev)
            home_adv = None            # per-row term is built below instead
        else:
            hfa_t = hfa_dev = None
            home_adv = home_advantage_prior(data.n_teams, config.home_mu, config.home_sd, config.home_adv_sd)

        # --- Optional: GBM residual-correction covariate (WP054) ---
        # z is already on the LOG-RATE scale (log of the multiplicative correction), so
        # it enters theta directly and beta = 1 reproduces WP053's wrapper. Unlike the
        # five failed additive covariates this one was CONSTRUCTED to predict the
        # champion's out-of-sample residual, so beta collapsing to ~2% of prior SD would
        # mean the Bayesian machinery rejects what the wrapper exploited.
        # WP061: availability as a FIXED-coefficient offset, i.e. a Poisson offset
        #   lambda = exp(theta + log(1 - k*out_own) + log(1 + k*out_opp))
        # A team's own absences depress its rate and lift its opponent's. Nothing is
        # estimated, so unlike every fitted covariate tried here it cannot be shrunk to
        # nothing by the latents. Being INSIDE the likelihood is the point: it explains
        # away an injury-driven dip at fitting time, so the latent keeps estimating TRUE
        # quality instead of carrying the dip forward through the AR(1) walk.
        avail_h = avail_a = 0.0
        if config.use_avail_offset:
            if data.avail_out_h is None or data.avail_out_a is None:
                raise ValueError("use_avail_offset needs ModelData.avail_out_h/avail_out_a "
                                 "(share of recent minutes unavailable per side)")
            avail_h, avail_a = avail_offset_terms(
                pm.Data("avail_oh", data.avail_out_h),
                pm.Data("avail_oa", data.avail_out_a),
                float(config.avail_k), pm.math.log)

        gbm_h = gbm_a = 0.0
        if config.use_gbm_covariate:
            if data.gbm_z_home is None or data.gbm_z_away is None:
                raise ValueError("use_gbm_covariate needs ModelData.gbm_z_home/away "
                                 "(WP054's prepare_data fills them)")
            b_gbm = pm.Normal("gbm_beta", 0.0, config.gbm_covariate_sd)
            gbm_h = b_gbm * pm.Data("gbm_zh", data.gbm_z_home)
            gbm_a = b_gbm * pm.Data("gbm_za", data.gbm_z_away)

        # --- Optional: crowdless-match shift in home advantage (WP031) ---
        # Behind-closed-doors football really did suppress home advantage: the
        # champion's out-of-sample tilt there is -0.2017 goals/match, CI excluding
        # zero. Unlike WP030's intactness this is OBSERVABLE at prediction time (a
        # scheduling fact), so predict.posterior_lambdas mirrors it rather than
        # omitting it -- withholding it would be the bug, not the safeguard.
        crowd_term = 0.0
        if config.use_crowd_regime:
            if data.crowdless is None:
                raise ValueError("use_crowd_regime needs ModelData.crowdless "
                                 "(WP031's prepare_data fills it)")
            crowd_hfa = pm.Normal("crowd_hfa", 0.0, config.crowd_hfa_sd)
            crowd_term = crowd_hfa * pm.Data("crowdless", data.crowdless)
            
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
                adj_factor_home = 1.0 + config.xG_adjustment_strength * defense[data.t_walk, data.opp_idx]
                adj_factor_away = 1.0 + config.xG_adjustment_strength * defense[data.t_walk, data.team_idx]
                
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

        # --- Global log-rate level: fixed, drifting, or absent ---
        if config.time_varying_level:
            level_t = global_walk("level_t", data.n_time, config.level_walk_sigma,
                                  init_mu=config.intercept_mu, init_sd=config.intercept_sd)
            intercept = level_t[data.t_walk]
        elif config.use_intercept:
            intercept = pm.Normal("intercept", config.intercept_mu, config.intercept_sd)
        else:
            intercept = 0.0

        # --- Optional: end-of-season stakes (WP028) ---
        # EFFORT is multiplicative on each side's own latent contribution, so a team
        # with nothing to play for expresses less of its quality. Deliberately NOT an
        # additive covariate: congestion, absence, lineups and keeper all fitted at
        # ~2% of a prior SD because an additive term competes with latents that
        # already explain the variance.
        # --- Optional: squad intactness (WP030) ---
        # NOTE a DELIBERATE asymmetry with predict.py's numpy mirror: the mirror does
        # NOT implement this term. That is correct, not a sync bug. The term is
        # b*(intact - 1) and prediction assumes a normal XI (intact = 1), so it is
        # exactly zero there. Adding it to the mirror would make the model consume the
        # predicted match's team sheet, which is unavailable at the opening price and
        # would destroy the whole point of the design.
        intact_h = intact_a = 0.0
        if config.use_intactness:
            if data.intact_home is None or data.intact_away is None:
                raise ValueError("use_intactness needs ModelData.intact_home/away "
                                 "(WP030's prepare_data fills them)")
            # NOTE ON NAMES: these are PER-SIDE coefficients, not per-role. b_ia
            # multiplies the HOME side's intactness (+theta_home, -theta_away) and
            # b_id the AWAY side's, so each constrains a team's attack and defence
            # response to be equal. A per-role split is a different, untested model.
            b_ia = pm.Normal("intact_att", 0.0, config.intactness_sd)
            b_id = pm.Normal("intact_def", 0.0, config.intactness_sd)
            # CENTRING (WP030b). Centring on 1.0 makes the term zero at FULL strength,
            # but prediction then evaluates every match at full strength while the
            # training mean is 0.989 -- a constant +1.1% inflation of both rates on
            # every match. WP030's first arm did exactly that and lost +0.00090
            # UNIFORMLY across every slice. Centring on the TRAINING mean instead
            # makes the term zero at AVERAGE availability, which is what an unknown
            # XI actually implies. The mean is taken from the data array handed in,
            # which during fitting holds training rows only -- so it cannot leak.
            if config.intactness_center_mean:
                c_h = float(np.mean(data.intact_home))
                c_a = float(np.mean(data.intact_away))
            else:
                c_h = c_a = 1.0
            intact_h = b_ia * (pm.Data("intact_h", data.intact_home) - c_h)
            intact_a = b_id * (pm.Data("intact_a", data.intact_away) - c_a)

        # Contributions SUM inside one multiplier, so each stays separately
        # identifiable: effort = 1 + g_stakes*stakes + g_cong*rest.
        eff_h = eff_a = None
        _ch, _ca = [], []
        if config.use_stakes_effort:
            if data.stakes_home is None or data.stakes_away is None:
                raise ValueError("use_stakes_effort needs ModelData.stakes_home/away")
            g_eff = pm.Normal("stakes_effort", 0.0, config.stakes_effort_sd)
            _ch.append(g_eff * pm.Data("stakes_h", data.stakes_home))
            _ca.append(g_eff * pm.Data("stakes_a", data.stakes_away))
        if config.use_congestion_effort:
            cm = config.congestion_effort_mode
            hv = data.rest_home if cm == "rest" else data.load_home
            av = data.rest_away if cm == "rest" else data.load_away
            if hv is None or av is None:
                raise ValueError(f"use_congestion_effort mode='{cm}' needs "
                                 f"ModelData.{cm}_home/away")
            g_c = pm.Normal(f"{cm}_effort", 0.0, config.congestion_effort_sd)
            _ch.append(g_c * pm.Data(f"{cm}_eff_h", hv))
            _ca.append(g_c * pm.Data(f"{cm}_eff_a", av))
        if _ch:
            eff_h = 1.0 + sum(_ch)
            eff_a = 1.0 + sum(_ca)

        # --- Optional: fixture-congestion covariate (WP027) ---
        if config.use_congestion:
            cm = config.congestion_mode
            hv = data.rest_home if cm == "rest" else data.load_home
            av = data.rest_away if cm == "rest" else data.load_away
            if hv is None or av is None:
                raise ValueError(f"use_congestion mode='{cm}' needs ModelData."
                                 f"{cm}_home/away (WP027's prepare_data fills them)")
            beta_cong = pm.Normal(f"beta_{cm}", 0.0, config.congestion_sd)
            cong_home = beta_cong * pm.Data(f"{cm}_home", hv)
            cong_away = beta_cong * pm.Data(f"{cm}_away", av)
        else:
            cong_home = cong_away = 0.0

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
            + avail_h
            + gbm_h
            + absence_home
            + cong_home
            + intact_h
            - intact_a
            + xG_contribution_home
            + (attack[data.t_walk, data.team_idx] if eff_h is None
               else eff_h * attack[data.t_walk, data.team_idx])
            - (defense[data.t_walk, data.opp_idx] if eff_a is None
               else eff_a * defense[data.t_walk, data.opp_idx])
            + ((hfa_t[data.t_walk] + hfa_dev[data.team_idx] if config.time_varying_hfa
                else home_adv[data.team_idx]) + crowd_term) * data.home
        )

        theta_away = (
            intercept
            + avail_a
            + gbm_a
            + absence_away
            + cong_away
            + intact_a
            - intact_h
            + xG_contribution_away
            + (attack[data.t_walk, data.opp_idx] if eff_a is None
               else eff_a * attack[data.t_walk, data.opp_idx])
            - (defense[data.t_walk, data.team_idx] if eff_h is None
               else eff_h * defense[data.t_walk, data.team_idx])
        )

        # WP016: when defence is split, the xG channel must see SUPPRESSION ONLY.
        # That asymmetry is the whole identification strategy — xG conceded reflects
        # how many chances you allow, goals conceded additionally reflects whether
        # they were saved, so the gap between the two is what pins shot-stopping
        # down. Feed both channels the same rate and the two latents collapse.
        if config.split_defence:
            theta_xg_home = soft_clip(theta_home + (defense[data.t_walk, data.opp_idx]
                                                    - suppress[data.t_walk, data.opp_idx]),
                                      config.clip_theta)
            theta_xg_away = soft_clip(theta_away + (defense[data.t_walk, data.team_idx]
                                                    - suppress[data.t_walk, data.team_idx]),
                                      config.clip_theta)
        else:
            theta_xg_home = theta_xg_away = None

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
        # WP022: when time decay is on, every observation is weighted by its age,
        # which requires a Potential rather than an observed RV. decay_w is also
        # applied to the measurement channels below, so the whole likelihood
        # decays coherently rather than only the goals part.
        decay_w = (time_decay_weights(data.match_day, config.decay_half_life_days,
                                      normalise=config.decay_normalise)
                   if config.use_time_decay else None)
        # WP028 DISPERSION: alpha = alpha0 * exp(b * stakes_match), so a dead rubber
        # is MORE overdispersed and its outcome probabilities flatter. This is the
        # mechanism the premise supports (dead rubbers measured 16% more variable),
        # and it modifies the VARIANCE, so unlike every failed additive covariate it
        # cannot compete with the latents. predict.py's scoreline_grid takes a
        # matching alpha -- an overdispersed likelihood with a Poisson predictive
        # would measure nothing.
        disp_alpha = None
        if config.use_stakes_dispersion:
            if data.stakes_match is None:
                raise ValueError("use_stakes_dispersion needs ModelData.stakes_match")
            b_disp = pm.Normal("stakes_disp", 0.0, config.stakes_disp_sd)
            disp_alpha = pm.Deterministic(
                "disp_alpha",
                config.stakes_alpha0 * pm.math.exp(b_disp * pm.Data("stakes_m", data.stakes_match)))

        if decay_w is None:
            if disp_alpha is None:
                pm.Poisson("goals_home", mu=lambda_home, observed=data.goals_home)
                pm.Poisson("goals_away", mu=lambda_away, observed=data.goals_away)
            else:
                pm.NegativeBinomial("goals_home", mu=lambda_home, alpha=disp_alpha,
                                    observed=data.goals_home)
                pm.NegativeBinomial("goals_away", mu=lambda_away, alpha=disp_alpha,
                                    observed=data.goals_away)
        else:
            gl = 0.0
            for lam, obs in ((lambda_home, data.goals_home), (lambda_away, data.goals_away)):
                gl = gl + (decay_w * pm.logp(pm.Poisson.dist(mu=lam), obs)).sum()
            pm.Potential("goals_decayed", gl)

        # --- Optional: Dixon-Coles low-score correlation correction ---
        if config.use_dixon_coles:
            dixon_coles_adjustment(
                lambda_home, lambda_away, data.goals_home, data.goals_away, config.rho_dc_sd
            )

        # --- Optional: score the observed outcome directly (WP017) ---
        if config.use_outcome_likelihood:
            outcome_likelihood(
                lambda_home, lambda_away, data.goals_home, data.goals_away,
                weight=config.outcome_weight, max_goals=config.outcome_max_goals,
            )

        # --- Optional: xG as a second measurement of the same latent rate (WP007) ---
        if config.use_xg_channel:
            if config.split_defence:
                lam_xg_home = pm.Deterministic("lambda_xg_home", pm.math.exp(theta_xg_home))
                lam_xg_away = pm.Deterministic("lambda_xg_away", pm.math.exp(theta_xg_away))
            else:
                lam_xg_home, lam_xg_away = lambda_home, lambda_away
            xg_measurement_channel(
                lam_xg_home, lam_xg_away, data.xG_obs_home, data.xG_obs_away,
                weight=config.xg_channel_weight,
                shape_alpha=config.xg_channel_shape_alpha,
                shape_beta=config.xg_channel_shape_beta,
                season_idx=(data.season_idx if config.xg_kappa_by_season else None),
                n_seasons=data.n_seasons,
                kappa_tau_sd=config.xg_kappa_tau_sd,
                obs_weight=decay_w,
            )

        # --- Optional: shot COUNT as a further measurement channel (WP019) ---
        # Deliberately reads lambda_home/away, not the suppression-only rate the
        # xG channel uses under split_defence: a shot count is a property of the
        # attack meeting the defence, with no shot-stopping stage in between.
        if config.use_shot_channel:
            shot_count_channel(
                lambda_home, lambda_away, data.shots_home, data.shots_away,
                weight=config.shot_channel_weight,
                kappa_mu=config.shot_kappa_mu,
                kappa_sd=config.shot_kappa_sd,
                obs_weight=decay_w,
            )

        # --- Optional: deep completions as a measurement channel (WP020) ---
        # Reads lambda_home/away for the same reason the shot channel does: a deep
        # completion is the attack meeting the defence, with no keeper stage in it.
        if config.use_deep_channel:
            deep_completion_channel(
                lambda_home, lambda_away, data.deep_home, data.deep_away,
                weight=config.deep_channel_weight,
                kappa_mu=config.deep_kappa_mu,
                kappa_sd=config.deep_kappa_sd,
                obs_weight=decay_w,
            )

        # --- Optional: the MARKET as a measurement channel (WP039) ---
        # Pinnacle's closing 1X2 price for each PAST match, inverted through a Poisson
        # scoreline model into an implied rate per side, observed with the same Gamma
        # machinery as xG. Premise: it adds +0.0159 OOS R^2 for next-match goals on top
        # of xG, 3.3x the level at which WP019's shot count failed as a channel.
        #
        # NOT leakage, and not circular. Like xG, this is an observation about matches
        # already played: predicting round R reads closing lines from rounds < R only,
        # and `market_obs_w` is 0 wherever a price is missing. The exploitable asymmetry
        # is that WP010 measured closing lines to be SHARPER than opening ones, so this
        # distils closing-quality information about the past to price future OPENING
        # lines. predict.py needs no mirror: the channel is a Potential on training
        # rows and never enters the goal predictive.
        #
        # `kappa` matters more here than for xG. Inverting 1X2 under Poisson
        # INDEPENDENCE compresses both rates to reproduce the market's draw price
        # (real football has more draws than independent Poisson), so the implied
        # rates average 1.430/1.114 against the model's 1.571/1.264 -- about 10% low.
        # The learned kappa is what absorbs that level offset (WP014 established it
        # absorbs exactly this kind of drift).
        if config.use_market_channel:
            w_mkt = data.market_obs_w
            if w_mkt is None:
                w_mkt = 1.0
            elif decay_w is not None:
                w_mkt = w_mkt * decay_w
            xg_measurement_channel(
                lambda_home, lambda_away, data.market_lh, data.market_la,
                weight=config.market_channel_weight,
                shape_alpha=config.market_channel_shape_alpha,
                shape_beta=config.market_channel_shape_beta,
                shape_fixed=config.market_channel_shape_fixed,
                obs_weight=w_mkt,
                name="mkt",
            )

    return model
