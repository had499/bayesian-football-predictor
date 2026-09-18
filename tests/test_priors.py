import numpy as np
import pymc as pm

from football_model.model.priors import ar1_hierarchical_sigma, rho_prior, league_home_advantage_prior


def test_rho_prior_registers_expected_name_and_distribution():
    with pm.Model() as model:
        rho = rho_prior("att", rho_alpha=29.0, rho_beta=1.0)
        assert rho.name == "rho_att"
        idata = pm.sample_prior_predictive(draws=2000, random_seed=0)
    # Beta(29, 1) has mean 29/30 ≈ 0.967
    assert np.isclose(idata.prior["rho_att"].values.mean(), 29 / 30, atol=0.01)


def test_ar1_hierarchical_sigma_registers_expected_names_and_shape():
    n_teams = 5
    with pm.Model() as model:
        sigma_team, sigma_pop = ar1_hierarchical_sigma("att", n_teams, pop_scale=0.02)
        var_names = {rv.name for rv in model.free_RVs} | {d.name for d in model.deterministics}
        assert {"sigma_att_pop", "sigma_att_team_raw", "sigma_att_team"} <= var_names
        idata = pm.sample_prior_predictive(draws=5, random_seed=0)
    assert idata.prior["sigma_att_team"].shape[-1] == n_teams
    assert (idata.prior["sigma_att_team"].values > 0).all()  # HalfNormal-scaled: always positive


def test_ar1_hierarchical_sigma_is_noncentered():
    """The 'raw' variable's own distribution must not depend on sigma_pop —
    that's what makes this non-centered. Check its empirical SD stays ~1
    (its declared HalfNormal(1) scale) regardless of what sigma_pop's own
    prior scale is set to, mirroring how home_advantage_prior's
    home_adv_raw is checked for the same property."""
    with pm.Model():
        _, _ = ar1_hierarchical_sigma("att", n_teams=4, pop_scale=0.02)
        idata_small_pop = pm.sample_prior_predictive(draws=3000, random_seed=0)
    with pm.Model():
        _, _ = ar1_hierarchical_sigma("att", n_teams=4, pop_scale=5.0)
        idata_big_pop = pm.sample_prior_predictive(draws=3000, random_seed=0)

    raw_sd_small = idata_small_pop.prior["sigma_att_team_raw"].values.std()
    raw_sd_big = idata_big_pop.prior["sigma_att_team_raw"].values.std()
    # HalfNormal(1) has SD ~0.6 regardless of pop_scale, if truly non-centered
    assert np.isclose(raw_sd_small, raw_sd_big, atol=0.05)


def test_ar1_hierarchical_sigma_team_equals_raw_times_pop():
    n_teams = 6
    with pm.Model():
        sigma_team, sigma_pop = ar1_hierarchical_sigma("def", n_teams, pop_scale=0.02)
        idata = pm.sample_prior_predictive(draws=4, random_seed=1)
    raw = idata.prior["sigma_def_team_raw"].values
    pop = idata.prior["sigma_def_pop"].values[..., None]
    team = idata.prior["sigma_def_team"].values
    assert np.allclose(team, raw * pop)


def test_ar1_hierarchical_sigma_larger_pop_scale_shifts_teams_up():
    """Not a claim about any one draw — the population-level scale should
    move the whole distribution of team sigmas, on average, across many
    draws (this is the actual pooling mechanism working)."""
    with pm.Model():
        ar1_hierarchical_sigma("att", n_teams=4, pop_scale=0.01)
        small = pm.sample_prior_predictive(draws=3000, random_seed=0)
    with pm.Model():
        ar1_hierarchical_sigma("att", n_teams=4, pop_scale=0.10)
        big = pm.sample_prior_predictive(draws=3000, random_seed=0)

    assert big.prior["sigma_att_team"].values.mean() > 5 * small.prior["sigma_att_team"].values.mean()


def test_league_home_advantage_prior_registers_expected_names_and_shapes():
    league_names = ["EPL", "Bundesliga", "La_Liga"]
    league_n_teams = [4, 6, 5]
    with pm.Model() as model:
        mu_global, mu_league, sd, home_adv_by_league = league_home_advantage_prior(
            league_names, league_n_teams, mu_center=0.13, mu_scale=0.03,
            between_league_scale=0.03, sd_scale=0.02,
        )
        var_names = {rv.name for rv in model.free_RVs} | {d.name for d in model.deterministics}
        assert {"home_mu_global", "home_mu_between_league_sd", "home_mu_league_raw",
                "home_mu_league", "home_sd"} <= var_names
        for name in league_names:
            assert f"home_adv_raw_{name}" in var_names
            assert f"home_adv_{name}" in var_names
        idata = pm.sample_prior_predictive(draws=5, random_seed=0)

    assert idata.prior["home_mu_league"].shape[-1] == len(league_names)
    for name, n in zip(league_names, league_n_teams):
        assert idata.prior[f"home_adv_{name}"].shape[-1] == n
    assert len(home_adv_by_league) == len(league_names)


def test_league_home_advantage_prior_reduces_to_single_global_mean_with_one_league():
    """With a single league, home_adv should behave the same way
    home_advantage_prior's team -> global pooling already does: every
    team's home_adv centers on the same one mu (here, mu_league[0], which
    is itself just a noised copy of mu_global) -- no meaningfully different
    behaviour just from routing a single league through the multi-league
    code path."""
    with pm.Model():
        league_home_advantage_prior(["EPL"], [4], mu_center=0.13, mu_scale=0.03,
                                     between_league_scale=0.03, sd_scale=0.02)
        idata = pm.sample_prior_predictive(draws=5000, random_seed=0)
    mu_league0 = idata.prior["home_mu_league"].values[..., 0]
    home_adv = idata.prior["home_adv_EPL"].values
    # home_adv's per-draw mean across its 4 teams should track mu_league[0]
    # for that same draw (small residual noise from home_sd * raw, n=4).
    per_draw_team_mean = home_adv.mean(axis=-1)
    assert np.corrcoef(mu_league0.ravel(), per_draw_team_mean.ravel())[0, 1] > 0.9


def test_league_home_advantage_prior_between_league_scale_controls_league_spread():
    """Larger between_league_scale -> league means (home_mu_league) spread
    further apart from each other -- the actual pooling mechanism this
    function exists to add."""
    league_names, league_n_teams = ["A", "B", "C", "D"], [4, 4, 4, 4]
    with pm.Model():
        league_home_advantage_prior(league_names, league_n_teams, between_league_scale=0.001)
        tight = pm.sample_prior_predictive(draws=3000, random_seed=0)
    with pm.Model():
        league_home_advantage_prior(league_names, league_n_teams, between_league_scale=0.10)
        loose = pm.sample_prior_predictive(draws=3000, random_seed=0)

    tight_spread = tight.prior["home_mu_league"].values.std(axis=-1).mean()
    loose_spread = loose.prior["home_mu_league"].values.std(axis=-1).mean()
    assert loose_spread > 5 * tight_spread


def test_league_home_advantage_prior_sd_shared_not_per_league():
    """Only one `home_sd` variable should exist regardless of league count
    -- the within-league team spread is deliberately NOT given a per-league
    level (see the function's docstring: not evidence-backed)."""
    with pm.Model() as model:
        league_home_advantage_prior(["EPL", "Bundesliga"], [4, 4])
        var_names = [rv.name for rv in model.free_RVs]
    assert var_names.count("home_sd") == 1
