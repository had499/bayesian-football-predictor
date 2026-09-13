import numpy as np
import pymc as pm

from football_model.model.priors import ar1_hierarchical_sigma, rho_prior


def test_rho_prior_matches_ar1_hyperpriors_rho_name_and_distribution():
    with pm.Model() as model:
        rho = rho_prior("att", rho_alpha=29.0, rho_beta=1.0)
        assert rho.name == "rho_att"
        idata = pm.sample_prior_predictive(draws=2000, random_seed=0)
    # Beta(29, 1) has mean 29/30 ≈ 0.967 — matches ar1_hyperpriors' own rho
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
