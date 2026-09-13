import numpy as np
import pymc as pm
import pytensor.tensor as pt

from football_model.model.components import (
    ar1_team_process,
    centered_over_teams,
    dixon_coles_adjustment,
    dixon_coles_tau,
    masked_mean_over_teams,
    soft_clip,
)


def _draw_ar1(n_time, n_teams, active_mask=None, season_start_mask=None,
              season_start_sigma_mult=1.0, sigma=0.5, rho=0.9, seed=0):
    with pm.Model():
        x = ar1_team_process(
            "x",
            n_time,
            n_teams,
            sigma,
            rho,
            active_mask=active_mask,
            season_start_mask=season_start_mask,
            season_start_sigma_mult=season_start_sigma_mult,
        )
        idata = pm.sample_prior_predictive(draws=1, random_seed=seed)
    return idata.prior["x"].values[0, 0]  # (n_time, n_teams)


def test_ar1_freezes_while_inactive():
    n_time, n_teams = 10, 2
    active_mask = np.ones((n_time, n_teams), dtype="float32")
    # team 1 goes inactive from t=4 onward
    active_mask[4:, 1] = 0.0

    x = _draw_ar1(n_time, n_teams, active_mask=active_mask)

    frozen_value = x[3, 1]
    assert np.allclose(x[4:, 1], frozen_value), "inactive team's value should hold steady"


def test_ar1_resumes_after_reactivation():
    n_time, n_teams = 10, 2
    active_mask = np.ones((n_time, n_teams), dtype="float32")
    active_mask[3:7, 1] = 0.0  # out for rounds 3-6, back at round 7

    x = _draw_ar1(n_time, n_teams, active_mask=active_mask, sigma=0.8, seed=3)

    frozen_value = x[2, 1]
    assert np.allclose(x[3:7, 1], frozen_value)
    # once reactivated it should (almost certainly) move away from the frozen value
    assert not np.isclose(x[7, 1], frozen_value)


def test_ar1_fully_active_matches_unmasked_behaviour_in_shape():
    n_time, n_teams = 6, 3
    x_masked = _draw_ar1(n_time, n_teams, active_mask=np.ones((n_time, n_teams)), seed=7)
    x_unmasked = _draw_ar1(n_time, n_teams, active_mask=None, seed=7)
    assert x_masked.shape == x_unmasked.shape == (n_time, n_teams)


def test_season_start_window_inflates_variance():
    """With a big sigma multiplier concentrated on a short flagged window,
    the innovations inside the window should have much larger spread than
    those outside it, across repeated draws."""
    n_time, n_teams = 8, 1
    season_start_mask = np.zeros(n_time, dtype="float32")
    season_start_mask[1] = 1.0  # only t=1 is inflated

    draws = np.stack(
        [
            _draw_ar1(
                n_time,
                n_teams,
                season_start_mask=season_start_mask,
                season_start_sigma_mult=20.0,
                sigma=0.05,
                rho=0.0,  # isolate the innovation term, no persistence
                seed=s,
            )
            for s in range(200)
        ]
    )  # (n_draws, n_time, n_teams)

    inflated_step_var = draws[:, 1, 0].var()
    normal_step_var = draws[:, 2, 0].var()
    assert inflated_step_var > 5 * normal_step_var


def test_ar1_team_process_scalar_sigma_matches_manual_recursion():
    """Regression guard for WP006's broadcast change to the sigma_t line
    (season_arr[:, None]) — must still reproduce the exact same recursion
    for the scalar-sigma case every config before WP006 uses."""
    n_time, n_teams = 5, 3
    sigma, rho, init_scale = 0.3, 0.8, 0.2
    with pm.Model():
        ar1_team_process("x", n_time, n_teams, sigma, rho, init_scale=init_scale)
        idata = pm.sample_prior_predictive(draws=1, random_seed=1)
    z = idata.prior["x_std"].values[0, 0]      # (n_time, n_teams)
    x_actual = idata.prior["x"].values[0, 0]

    x_expected = np.zeros((n_time, n_teams))
    x_expected[0] = init_scale * z[0]
    for t in range(1, n_time):
        x_expected[t] = rho * x_expected[t - 1] + sigma * z[t]
    assert np.allclose(x_actual, x_expected, atol=1e-6)


def test_ar1_team_process_per_team_sigma_matches_manual_recursion():
    """WP006: sigma can be a (n_teams,) vector — one innovation SD per team
    — instead of one scalar shared by all. Cross-check against the same
    manual recursion, with sigma broadcasting per-team."""
    n_time, n_teams = 6, 3
    sigma = np.array([0.05, 0.2, 0.4])
    rho, init_scale = 0.85, 0.2
    with pm.Model():
        ar1_team_process("x", n_time, n_teams, sigma, rho, init_scale=init_scale)
        idata = pm.sample_prior_predictive(draws=1, random_seed=2)
    z = idata.prior["x_std"].values[0, 0]
    x_actual = idata.prior["x"].values[0, 0]

    x_expected = np.zeros((n_time, n_teams))
    x_expected[0] = init_scale * z[0]
    for t in range(1, n_time):
        x_expected[t] = rho * x_expected[t - 1] + sigma * z[t]
    assert np.allclose(x_actual, x_expected, atol=1e-6)


def test_ar1_team_process_per_team_sigma_gives_different_step_variance():
    """The actual point of per-team sigma: a team with a tiny sigma should
    move far less than one with a large sigma, across repeated draws —
    proving the per-team value really reaches each team's own innovations,
    not just the recursion formula in isolation."""
    n_time, n_teams = 3, 2
    sigma = np.array([0.02, 0.5])  # team 0 nearly frozen, team 1 volatile
    draws = np.stack([
        _draw_ar1(n_time, n_teams, sigma=sigma, rho=0.0, seed=s)  # rho=0 isolates the innovation term
        for s in range(200)
    ])  # (n_draws, n_time, n_teams)
    var_team0 = draws[:, 1, 0].var()
    var_team1 = draws[:, 1, 1].var()
    assert var_team1 > 50 * var_team0


def test_centered_over_teams_masked_mean_is_zero_for_active_teams():
    n_time, n_teams = 5, 4
    rng = np.random.default_rng(0)
    x_np = rng.normal(size=(n_time, n_teams)).astype("float32")
    active_mask = np.ones((n_time, n_teams), dtype="float32")
    active_mask[:, 3] = 0.0  # team 3 inactive throughout

    with pm.Model():
        x = pt.as_tensor_variable(x_np)
        centered = centered_over_teams(x, "centered", active_mask=active_mask)
        result = pm.draw(centered)

    active_part = result[:, :3]
    assert np.allclose(active_part.mean(axis=1), 0.0, atol=1e-6)


def test_centered_over_teams_without_mask_matches_plain_mean():
    n_time, n_teams = 4, 3
    rng = np.random.default_rng(1)
    x_np = rng.normal(size=(n_time, n_teams)).astype("float32")

    with pm.Model():
        x = pt.as_tensor_variable(x_np)
        centered = centered_over_teams(x, "centered")
        result = pm.draw(centered)

    assert np.allclose(result.mean(axis=1), 0.0, atol=1e-6)


def test_masked_mean_over_teams_ignores_inactive_teams():
    n_time, n_teams = 3, 3
    x_np = np.array(
        [
            [1.0, 3.0, 100.0],
            [2.0, 4.0, 100.0],
            [10.0, 10.0, 100.0],
        ],
        dtype="float32",
    )
    active_mask = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype="float32",
    )

    with pm.Model():
        mean = masked_mean_over_teams(pt.as_tensor_variable(x_np), active_mask)
        result = pm.draw(mean)

    assert np.allclose(result, [2.0, 3.0, 10.0])


def test_soft_clip_bounds_output():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    y = soft_clip(pt.as_tensor_variable(x), limit=2.0).eval()
    # tanh saturates to 1.0 in float64 well before x=1000, so extreme inputs
    # legitimately hit the limit exactly rather than staying strictly inside it
    assert np.all(np.abs(y) <= 2.0)
    assert y[2] == 0.0
    assert y[3] > 0 and y[1] < 0
    assert np.abs(y[1]) < 2.0  # moderate input stays strictly within bounds


def test_dixon_coles_tau_matches_hand_computed_values():
    lam, mu, rho = 1.4, 1.1, -0.1
    goals_home = np.array([0, 1, 0, 1, 2, 3])
    goals_away = np.array([0, 0, 1, 1, 2, 0])

    tau = dixon_coles_tau(lam, mu, goals_home, goals_away, rho).eval()

    expected = np.array([
        1 - lam * mu * rho,   # 0-0
        1 + lam * rho,        # 1-0
        1 + mu * rho,         # 0-1
        1 - rho,               # 1-1
        1.0,                   # 2-2: untouched
        1.0,                   # 3-0: untouched
    ])
    assert np.allclose(tau, expected)


def test_dixon_coles_tau_is_one_when_rho_is_zero():
    goals_home = np.array([0, 1, 0, 1, 5])
    goals_away = np.array([0, 0, 1, 1, 2])
    tau = dixon_coles_tau(1.3, 1.2, goals_home, goals_away, rho=0.0).eval()
    assert np.allclose(tau, 1.0)


def test_dixon_coles_adjustment_adds_rho_and_matches_log_tau():
    lam = np.array([1.4, 1.1])
    mu = np.array([1.1, 1.3])
    goals_home = np.array([0, 2])
    goals_away = np.array([0, 1])

    with pm.Model() as model:
        potential = dixon_coles_adjustment(
            pt.as_tensor_variable(lam), pt.as_tensor_variable(mu),
            goals_home, goals_away, sd=0.1,
        )
        assert "rho_dc" in [rv.name for rv in model.free_RVs]
        assert "dixon_coles" in [p.name for p in model.potentials]

        # Evaluate the Potential at a fixed rho (substituting the free
        # variable directly) and compare to a manual log(tau).sum().
        rho_val = -0.05
        actual_logp = potential.eval({model["rho_dc"]: rho_val})

    tau_expected = dixon_coles_tau(lam, mu, goals_home, goals_away, rho_val)
    expected_logp = pt.log(tau_expected).sum().eval()

    assert np.isclose(float(actual_logp), float(expected_logp))
