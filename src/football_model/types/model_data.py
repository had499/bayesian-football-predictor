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
    t_idx: np.ndarray          # shape=(n_obs,); the TRUE round — identification only
    team_idx: np.ndarray       # shape=(n_obs,)
    opp_idx: np.ndarray        # shape=(n_obs,)
    match_idx: np.ndarray      # shape=(n_obs,)
    home: np.ndarray           # shape=(n_obs,)

    # --- observed outputs ---
    goals_home: np.ndarray     # shape=(n_obs,)
    goals_away: np.ndarray     # shape=(n_obs,)
    xG_home: np.ndarray        # shape=(n_obs,)
    xG_away: np.ndarray        # shape=(n_obs,)

    # Index into the latent WALK. Equals t_idx unless prepare_model_data is called
    # with time_grid_step > 1, which collapses k rounds onto one node to cut
    # sampling cost (quadratic in n_time). Everything that indexes attack/defence
    # uses THIS; everything that identifies a match by round uses t_idx.
    t_walk: np.ndarray = None  # shape=(n_obs,)
    
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

    # --- realised in-match SHOT COUNTS, one per side (WP019 shot-count channel) ---
    # Same status as xG_obs_*: an observation for training rows, never a predictor
    # for the row itself. Total xG only constrains shots x quality; observing the
    # count separately lets the two carry different weight.
    shots_home: np.ndarray = None         # shape=(n_obs,)
    shots_away: np.ndarray = None         # shape=(n_obs,)

    # --- deep completions, one per side (WP020 deep-completion channel) ---
    # Understat `deep`: passes completed within ~20m of the opposing goal, excluding
    # crosses. Same status as xG_obs_*/shots_*: an observation for training rows,
    # never a predictor for the row itself. NOT derived from the shot list, which is
    # the whole point — see components.deep_completion_channel.
    deep_home: np.ndarray = None          # shape=(n_obs,)
    deep_away: np.ndarray = None          # shape=(n_obs,)

    # --- match date as days since the first match (WP022 time decay) ---
    # Needed to weight each observation by its age. Days rather than rounds
    # because rounds are not uniform in time (winter/summer breaks).
    match_day: np.ndarray = None          # shape=(n_obs,)

    # --- standardised external Elo per (walk step, team) (WP025) ---
    # Used only at a team's ENTRY round, to scale how far below/above average a
    # promoted side starts. 0 everywhere = the old "average team" prior.
    entry_elo_z: np.ndarray = None        # shape=(n_time, n_teams)

    # --- fixture congestion, one per side (WP027) ---
    # Standardised days of rest before this match, counting EVERY competition
    # (league + FA Cup + EFL Cup + CL/EL/UECL). WP009's rest-day null used a
    # LEAGUE-ONLY measure which is wrong for 27.9% of matches -- a Thursday
    # European trip looked like a clear week. Legitimate pre-match information,
    # and genuinely invisible to the model: cup and European fixtures are absent
    # from Understat entirely, so the latent walk never saw them.
    rest_home: np.ndarray = None          # shape=(n_obs,)
    rest_away: np.ndarray = None          # shape=(n_obs,)
    load_home: np.ndarray = None          # shape=(n_obs,)  fixtures in prior 14 days
    load_away: np.ndarray = None          # shape=(n_obs,)

    # --- end-of-season stakes, one per side (WP028) ---
    # How much this result moves the team's end-of-season probabilities, summed over
    # the relegation / Europe / title boundaries. Simulation-based; 0 = nothing left
    # to play for. Forward-looking in a way the latent walk cannot anticipate: the
    # walk learns "they played badly in April" only AFTER those matches.
    stakes_home: np.ndarray = None        # shape=(n_obs,)
    stakes_away: np.ndarray = None        # shape=(n_obs,)
    # min(home, away) -- the match-level "is this a dead rubber" summary, used for
    # dispersion. Dead rubbers were measured 16% MORE variable (var ratio 1.160).
    stakes_match: np.ndarray = None       # shape=(n_obs,)

    # --- squad intactness, one per side (WP030) ---
    # Share of a team's NORMAL starting strength that actually took the field, from
    # minutes-weighted player importance (EWMA half-life 15), normalised against that
    # team's own recent norm. 1.0 = usual XI.
    #
    # Used ONLY on training rows: it explains why a depleted team underperformed, so
    # the attack latent is not dragged down by results it should not own. At prediction
    # time a normal XI is assumed (intact = 1), the term is exactly zero, and NO lineup
    # information about the match being priced is consumed -- which is what keeps the
    # model usable against OPENING prices.
    intact_home: np.ndarray = None        # shape=(n_obs,)
    intact_away: np.ndarray = None        # shape=(n_obs,)

    # --- GBM residual-correction covariate, per side (WP054) ---
    # log of the multiplicative correction a gradient-boosted model predicts for this
    # match's rate, trained WALK-FORWARD on 253 features from strictly earlier CV
    # windows (WP053). theta += beta * gbm_z, so beta = 1 reproduces WP053's post-hoc
    # wrapper exactly and a fitted beta lets the model DECLINE the correction. 0 where
    # no composite exists (the earliest windows have no training base), which makes the
    # term contribute nothing rather than guess.
    # WP061: share of each side's recent minutes that is UNAVAILABLE (injured), as an
    # in-model OFFSET rather than a fitted covariate. Post-hoc correction (WP059) only
    # fixes the predicted lambda; an offset ALSO keeps an injury-driven dip out of the
    # latent, which the AR(1) walk would otherwise propagate forward as a permanent
    # quality change. The coefficient is FIXED (config.avail_k), so there is no parameter
    # for the ~29,000 latents to shrink -- the documented failure mode of every fitted
    # covariate tried so far.
    avail_out_h: np.ndarray | None = None
    avail_out_a: np.ndarray | None = None
    gbm_z_home: np.ndarray = None         # shape=(n_obs,)
    gbm_z_away: np.ndarray = None         # shape=(n_obs,)

    # --- market-implied goal rates per side (WP039) ---
    # Pinnacle's CLOSING 1X2 price for the match, de-vigged (Shin) and inverted through
    # a Poisson scoreline model into a rate for each side. Same status as xG_obs_*: an
    # observation for training rows, never a predictor for the row being forecast.
    # `market_obs_w` is 0 where no price exists (about 4% of matches), so the channel
    # simply does not fire there rather than needing imputation.
    market_lh: np.ndarray = None          # shape=(n_obs,)
    market_la: np.ndarray = None          # shape=(n_obs,)
    market_obs_w: np.ndarray = None       # shape=(n_obs,), 1 where priced else 0

    # --- crowdless-match indicator (WP031) ---
    # 1.0 for matches played behind closed doors (Project Restart 2020-06-17 until
    # fans returned ~2021-05-09), else 0. A SCHEDULING FACT, published months ahead,
    # so unlike WP030's intactness it is legitimately known at the opening price --
    # which is why predict.py DOES mirror this term. 434 of 3,780 test matches.
    crowdless: np.ndarray = None          # shape=(n_obs,)

    # --- squad-absence covariate (WP012) ---
    # Each side's leak-safe "absence drop": the share of its recent xG+xA produced
    # by players whose availability has fallen below their own norm (WP011). Built
    # only from matches BEFORE the one being predicted, so it is a legitimate
    # predictor, not an outcome. Zero where no roster history exists.
    absent_home: np.ndarray = None        # shape=(n_obs,)
    absent_away: np.ndarray = None        # shape=(n_obs,)

    # --- WP014: season index per observation, for season-varying xG calibration ---
    # The xG/goals ratio drifts ~0.96 -> 1.08 across 2014-2025, so one global
    # kappa is a compromise that is wrong at both ends of the history.
    season_idx: np.ndarray = None         # shape=(n_obs,), 0-based
    n_seasons: int = 0


    def __post_init__(self):
        # t_walk defaults to t_idx, i.e. one latent node per round — the behaviour
        # of every result before WP023. Guarantees any ModelData built without it
        # (tests, older callers) is unchanged rather than crashing.
        if self.t_walk is None and self.t_idx is not None:
            object.__setattr__(self, "t_walk", self.t_idx)

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

    # Correlate the attack and defence walks (WP037). The independent-walks prior
    # is contradicted by the data: corr(attack, defence) across teams is +0.79, since
    # good sides are good at both ends. Implemented as a Cholesky factor on the
    # standardised innovations, so r=0 is bit-identical to the old behaviour.
    use_ad_correlation: bool = False
    ad_corr_init: float = 0.5

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

    # Shot COUNT as a further tempered measurement channel on the same latent
    # rate (WP019; see components.shot_count_channel). Needs ModelData.shots_*.
    # Adds no latents — one channel, two scalars — deliberately unlike WP016's
    # latent split, which made the model worse.
    use_shot_channel: bool = False
    shot_channel_weight: float = 0.5
    shot_kappa_mu: float = 2.175        # log(~8.8) shots per unit scoring rate
    shot_kappa_sd: float = 0.2

    # Deep completions as a measurement channel on the same latent rate (WP020;
    # see components.deep_completion_channel). Needs ModelData.deep_*.
    use_deep_channel: bool = False
    deep_channel_weight: float = 0.5
    deep_kappa_mu: float = 1.62         # log(6.87/1.36) deep completions per unit rate
    deep_kappa_sd: float = 0.2

    # -------------------------
    # Dixon-Coles time decay (WP022)
    # -------------------------
    # Weight every observation -- goals AND both measurement channels -- by
    # exp(-xi * age). Needs ModelData.match_day. The premise check puts the
    # optimum near a 270-day half-life. See components.time_decay_weights.
    # Latent walk time grid (WP023, performance). k>1 collapses k rounds into one
    # latent node; cost is quadratic in n_time so k=2 ~halves sampling time.
    # Consumed by prepare_model_data, not build_model. 1 = unchanged.
    time_grid_step: int = 1

    # -------------------------
    # Elo-scaled entry level for promoted teams (WP025)
    # -------------------------
    # Replaces the single global promotion offset with `beta * elo_z` at entry, so
    # each promoted side starts where its actual (Championship + cup) strength says.
    # Needs ModelData.entry_elo_z. See components.ar1_team_process's entry_scale.
    use_elo_entry: bool = False
    elo_entry_sd: float = 0.25

    # GENERAL form: Elo anchors every team's walk on every round, not just at entry.
    # The walk reverts toward `beta * elo_z` instead of toward 0 ("average team").
    # Also carries mid-season information the EPL-only pipeline cannot see, e.g. a
    # deep European run. Needs ModelData.entry_elo_z (same grid).
    use_elo_anchor: bool = False
    elo_anchor_sd: float = 0.25

    use_time_decay: bool = False
    decay_half_life_days: float = 270.0
    decay_normalise: bool = True

    # -------------------------
    # Squad-absence covariate (WP012)
    # -------------------------
    # Adds beta_absence * absence to each side's log rate. WP011 found no
    # CONTEMPORANEOUS correlation between absence and model-minus-market error,
    # but that test could not see the channel where the feature changes how the
    # latent strengths are inferred during training — which is what this tests.
    use_absence: bool = False
    absence_sd: float = 0.5      # prior scale for beta_absence (expected negative)

    # -------------------------
    # Fixture congestion (WP027)
    # -------------------------
    # Adds beta * rest (or beta * load) to each side's log rate. Tested SEPARATELY
    # because rest and load are correlated and WP011's crux was a multicollinearity
    # artefact -- two features at r=0.76 split into cancelling coefficients and
    # looked significant. BACKLOG caution 1: test correlated features alone.
    # -------------------------
    # End-of-season stakes (WP028)
    # -------------------------
    # EFFORT: scales how much of its quality a team expresses, multiplicatively:
    #   theta = intercept + (1+g*stakes_h)*attack[h] - (1+g*stakes_a)*defence[a] + hfa
    # Multiplicative on purpose. Every ADDITIVE covariate tried here has been
    # crushed by the latents (congestion, absence, lineups, keeper all fitted at
    # ~2% of a prior SD) because it competes with parameters that already explain
    # the variance. Modulating the latents sidesteps that.
    # -------------------------
    # Squad intactness (WP030)
    # -------------------------
    # Adds b_att*(intact_h - 1) - b_def*(intact_a - 1) to the log rate. ZERO when a
    # team is at full strength, which is what makes it prediction-safe: the predicted
    # match assumes a normal XI, so the term vanishes and no team-sheet information is
    # needed. Expect b_att POSITIVE (a depleted side scores less) and b_def POSITIVE
    # (a depleted opponent concedes more, via the minus sign).
    #
    # Same algebra as WP012's absence covariate, which failed -- the differences are
    # that players are weighted by MINUTES rather than xG+xA (which scored keepers at
    # 0.003), and that this EXPLAINS observed past lineups instead of FORECASTING
    # today's (WP012's threshold "fired 4 matches late").
    # Crowdless-match shift in league-wide home advantage (WP031). The champion's
    # out-of-sample home tilt during crowdless football was -0.2017 goals/match
    # [-0.3789,-0.0454] -- it over-rates home teams there. Known in advance, so it is
    # mirrored in predict.py.
    # The market as a measurement channel (WP039). Premise: +0.0159 OOS R^2 for
    # next-match goals on top of xG, vs the +0.00484 at which WP019's shot count
    # failed. Weight starts at the same 0.5 hedge as the other channels -- WP032
    # showed that hedge was UNDER-set for xG, so expect to tune it.
    # The GBM residual correction as an in-model covariate (WP054). WP053 measured the
    # post-hoc wrapper at -0.00062 but it carries NO uncertainty; a fitted coefficient
    # fixes that and tests whether the model wants the correction at all. Prior sd 0.5
    # so that beta = 1 (reproducing the wrapper) sits at 2 prior SDs -- reachable but
    # not assumed.
    use_avail_offset: bool = False
    avail_k: float = 0.20
    use_gbm_covariate: bool = False
    gbm_covariate_sd: float = 0.5

    use_market_channel: bool = False
    market_channel_weight: float = 0.5
    market_channel_shape_alpha: float = 2.0
    market_channel_shape_beta: float = 0.5
    # Pin the Gamma shape rather than learning it (WP039b). shape IS the channel's
    # trust (CV = 1/sqrt(shape)); the learned version chose 58-120, i.e. ±9%, which
    # drove max rhat to 1.412. None = learn it, as WP039 did.
    market_channel_shape_fixed: float = None

    use_crowd_regime: bool = False
    crowd_hfa_sd: float = 0.15

    use_intactness: bool = False
    intactness_sd: float = 0.5
    # Centre the covariate on the TRAINING mean (0.989) rather than on 1.0.
    # Centring on 1.0 predicts every match at full strength; see model.py.
    intactness_center_mean: bool = False

    use_stakes_effort: bool = False
    stakes_effort_sd: float = 0.3

    # WP029: congestion entering the SAME effort multiplier, not as an additive
    # covariate. WP027 tested the additive form and the coefficient came back at 2%
    # of a prior SD -- the model declined it, as it declined absence, lineups and
    # keeper. WP028 then showed non-additive forms get USED (91% and 60% of prior SD).
    # "A tired team plays below itself" is also a better mechanism than "a tired team
    # scores 0.05 fewer goals". Contributions sum INSIDE the multiplier so stakes and
    # congestion stay separately identifiable:
    #   effort = 1 + g_stakes*stakes + g_cong*rest
    use_congestion_effort: bool = False
    congestion_effort_mode: str = "rest"   # "rest" or "load"
    congestion_effort_sd: float = 0.3

    # DISPERSION: goals become NegBinomial with alpha = alpha0 * exp(b * stakes_match),
    # so a dead rubber is MORE overdispersed and gets flatter outcome probabilities.
    # This is the mechanism the premise test actually supports (variance ratio 1.160)
    # and it touches the variance, not theta, so it cannot compete with the latents.
    use_stakes_dispersion: bool = False
    stakes_disp_sd: float = 0.5
    stakes_alpha0: float = 25.0     # near-Poisson baseline dispersion

    use_congestion: bool = False
    congestion_mode: str = "rest"   # "rest" (more rest -> expect higher) or "load"
    congestion_sd: float = 0.1

    # -------------------------
    # xG calibration drift (WP014)
    # -------------------------
    # Let the xG channel's kappa vary by season instead of one global value.
    # Measured drift: xG/goals ratio 0.957 (2014-16) -> 1.078 (2023-25), near
    # monotonic. Non-centered hierarchical: kappa_s = exp(log_kappa_mu + tau*z_s).
    xg_kappa_by_season: bool = False
    xg_kappa_tau_sd: float = 0.1   # prior scale on across-season kappa spread

    # -------------------------
    # Time-varying global level / home advantage (WP015)
    # -------------------------
    # The model has free parameters for LEVELS but none for level DRIFT: team
    # strengths are the only thing allowed to move over time, every global scalar
    # is frozen. Both of these are measured to vary a lot:
    #   home advantage: 0.008 (2020, empty stadiums) .. 0.300 across seasons
    #   league scoring: log-level 0.249 .. 0.494 (2.57 .. 3.28 goals/match)
    # Unlike kappa drift or Dixon-Coles (common-mode, cancels in H/D/A), a wrong
    # home advantage is DIRECTIONAL — it pushes P(home) up and P(away) down — so
    # it corrupts outcome discrimination directly.
    #
    # Build these together with HARD centering: a time-varying intercept on top of
    # soft centering would give two ways to express the same level drift and a
    # poorly identified posterior. Do NOT combine with xg_kappa_by_season either —
    # kappa and the intercept are collinear (both mediate latent -> observation level).
    time_varying_hfa: bool = False        # league-wide home advantage follows a slow walk
    time_varying_level: bool = False      # global log scoring level follows a slow walk
    level_walk_sigma: float = 0.05        # HalfNormal scale for both walks, per round

    # -------------------------
    # Split defence: chance suppression vs shot-stopping (WP016)
    # -------------------------
    # One defence latent currently absorbs two skills with very different
    # repeatability, and is forced to learn both at one rate:
    #   chance suppression (xG faced)                season-to-season r = +0.65
    #   shot-stopping (xG faced - goals conceded)    season-to-season r = +0.22
    # Split them so each gets its own innovation SD. Identification comes from the
    # two observation channels seeing DIFFERENT combinations:
    #   xG conceded    <- opponent attack - suppression            (stopping invisible)
    #   goals conceded <- opponent attack - suppression - stopping
    # so the gap between goals and xG is what pins shot-stopping down.
    #
    # `defence` is still registered as suppression + stopping, so every consumer
    # downstream (predict.py, the CV harness, the service) is unchanged.
    split_defence: bool = False
    stop_sigma_scale: float = 0.3   # multiplier on sigma_def for the shot-stopping walk;
                                    # <1 because only ~a fifth of observed save variation persists

    # -------------------------
    # Outcome-level likelihood (WP017)
    # -------------------------
    # Score the observed H/D/A outcome directly, alongside the goal counts. The
    # model is evaluated on RPS but trains on goal-count likelihood, and those
    # objectives barely track each other (corr of goal log-lik and RPS across 30
    # CV windows: -0.19). Measured symptom: home wins over-predicted ~1.4pp,
    # away under-predicted ~1.5pp, even though the fitted home advantage matches
    # the data's own home:away goal ratio. See components.outcome_likelihood for
    # the double-counting caveat — this is an objective re-weighting, not a
    # generative addition.
    use_outcome_likelihood: bool = False
    outcome_weight: float = 1.0
    outcome_max_goals: int = 10



    # =====================================================================
    # THE CHAMPION CONFIG
    # =====================================================================
    @classmethod
    def champion(cls, **overrides):
        """The best configuration the project has evidence for (as of WP021).

        `L0_baseline` + the xG measurement channel + the deep-completion channel.

        WARNING — the two pins below are NOT ModelConfig's own defaults. The CV
        harness (scripts/run_cv_window.py) overrides `clip_theta` to 5.0 and
        `center_team_strength` to False, and EVERY cross-validated number in this
        project was produced with those values. A plain `ModelConfig()` is a
        different model from anything that has ever been measured here, so the
        champion pins them explicitly.

        Why each channel is in:

        * `use_xg_channel` (WP007, confirmed WP013) — the one unambiguous win in
          the project. -0.00258 RPS [-0.00414, -0.00095] vs L0, better in 6/6
          leave-one-season-out folds.

        * `use_deep_channel` (WP020, re-tested WP021) — included on EXPECTED-VALUE
          grounds, not because it is proven. At 115 windows it is -0.00083
          [-0.00177, +0.00013]: the CI includes zero, and it is better in only
          6/10 seasons, so it is NOT an established mechanism. But P(improves) is
          95.9%, E[improvement] is +0.00083 RPS (~19% of the remaining gap to
          Pinnacle's pre-close line), and even the worst subgroup — dropping 2016,
          its strongest season — is still -0.00044. The downside is two scalars.
          Rejecting a 96%-likely gain to protect against a 4%-likely loss that is
          4x smaller is the wrong trade for a modelling decision, even though it
          would be the right bar for CLAIMING the mechanism is real.

        * `use_stakes_effort` (WP028) — end-of-season stakes scaling how much of its
          quality a team expresses, MULTIPLICATIVELY:
            theta = intercept + (1+g*stakes_h)*attack[h] - (1+g*stakes_a)*defence[a] + hfa
          -0.00014 [-0.00031, +0.00001] at 115 windows. Like `deep`, included on
          expected value rather than significance -- but with the BEST stability
          measured in this project: negative in 10/10 leave-one-season-out folds
          within a 0.0001 range, better in 7/10 seasons, coefficient sign-consistent
          115/115 at 91% of its prior SD. It also survives residualising against
          calendar position and the model's own strength gap (r stays +0.0404), so it
          is not proxying for "mid-table" or "late season" -- the correlations are
          -0.089 and -0.004. And unlike the WP025 Elo anchor it needs NO external data:
          stakes are simulated from the league table already in df_cv.

          NOTE this makes the champion require ModelData.stakes_home/away. Build them
          with work_products/wp028_stakes/{stakes.py,prepare_data.py}; a plain
          ModelData without them will now raise.

        Deliberately NOT included, despite positive point estimates: the arms at
        P(improves) 0.77-0.86 (`J_outcome_05`, `B_npxg`, `D_shotxg`). Across the 26
        arms tested, 3 exceeded 0.80 where pure noise predicts 5.2 — that cluster is
        indistinguishable from selection effects, and none has been confirmed on an
        independent grid the way `deep` was.
        """
        cfg = dict(
            clip_theta=5.0,              # harness value, NOT the dataclass default
            center_team_strength=False,  # harness value, NOT the dataclass default
            use_xg_channel=True,
            # WP032: TUNED, not assumed. 0.5 was a hedge against double-counting
            # (goals and xG share shot-level information) and was never measured --
            # WP007 compared only 0.5 vs 1.0 on a 30-window grid, called it null, and
            # WP013 then declined to revisit it. At 115 windows the curve has a smooth
            # interior peak at 0.75: w0.25 +0.00063 (CI excludes zero), w0.75 -0.00029
            # [-0.00059,+0.00002] better in 10/10 LOSO folds, w1.00 -0.00022, and the
            # smoke fit at w1.75/w3.00 is catastrophically worse (+0.016) as kappa gets
            # dragged off 1.0. 0.75 and 1.0 are statistically indistinguishable, so what
            # is established is that 0.5 was TOO LOW, not that 0.75 is exactly the peak.
            xg_channel_weight=0.75,
            # WP046: 3 rounds rather than 5. -0.00008 [-0.00018,+0.00002], better in
            # 10/10 leave-one-season-out folds and 9/10 seasons — MORE season-consistent
            # than either `deep` (6/10) or `stakes_effort` (7/10), both of which are in
            # on expected value. P(improves) ~94%, cost zero. Coherent with WP035, where
            # `season_start_sigma_mult` was flat across 2.0-3.0 and no-inflation scored
            # +0.00004: all three results point the same weak direction, that the model
            # wants slightly LESS season-start flexibility than it was given.
            #
            # NOT claimed as a real mechanism — the effect is 1/12 of the MDE, and a
            # consistency pattern this clean is equally what a tiny systematic artefact
            # of the CV grid would look like. Included on expected value, like `deep`.
            season_start_window=3,
            use_deep_channel=True,
            deep_channel_weight=0.5,
            use_stakes_effort=True,
            stakes_effort_sd=0.3,
        )
        cfg.update(overrides)
        return cls(**cfg)
