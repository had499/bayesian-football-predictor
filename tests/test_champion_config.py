"""The champion config is the project's single source of truth for 'the model'.

These tests exist because the config drifted for 21 work products as a line in
various READMEs. The specific trap they guard: the CV harness overrides two
ModelConfig defaults, so a plain ModelConfig() is NOT the model any of the
cross-validated numbers describe.
"""
import pytest

from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


def test_champion_pins_the_harness_values_not_the_dataclass_defaults():
    """clip_theta and center_team_strength differ between ModelConfig's defaults
    and what scripts/run_cv_window.py actually used for every CV result. If these
    ever silently align, the champion is no longer the validated model."""
    c, d = ModelConfig.champion(), ModelConfig()
    assert c.clip_theta == 5.0 and d.clip_theta != 5.0
    assert c.center_team_strength is False and d.center_team_strength is True


def test_champion_has_both_evidenced_channels():
    c = ModelConfig.champion()
    assert c.use_xg_channel and c.xg_channel_weight == 0.75     # WP007/WP013, tuned WP032
    assert c.season_start_window == 3                            # WP046, on expected value
    assert c.use_deep_channel and c.deep_channel_weight == 0.5  # WP020/WP021
    assert c.use_stakes_effort and c.stakes_effort_sd == 0.3    # WP028


def test_champion_excludes_everything_not_evidenced():
    """Arms that were measured null, worse, or only noise-level positive must stay
    off — several have positive point estimates and are still not included."""
    c = ModelConfig.champion()
    for flag in ("use_shot_channel",        # WP019 null; WP021 shows it CANCELS deep
                 "use_dixon_coles",         # WP002/WP007 null, mechanism understood
                 "use_outcome_likelihood",  # WP017 null, catastrophic at high weight
                 "split_defence",           # WP016 measurably WORSE
                 "time_varying_hfa",        # WP015 null
                 "time_varying_level",      # WP015 null
                 "use_absence",             # WP012 null
                 "use_per_team_sigma",      # WP006 null
                 "xg_kappa_by_season",      # WP014 null
                 "use_form_decomposition",
                 "use_stakes_dispersion",   # WP028: measurably WORSE (+0.00019)
                 "use_elo_anchor",          # WP025: works, but an external dependency
                 "use_congestion",          # WP027: definitively null
                 "use_time_decay"):         # WP022: WORSE
        assert getattr(c, flag) is False, f"{flag} is not evidenced and must stay off"


def test_champion_overrides_pass_through():
    assert ModelConfig.champion(use_deep_channel=False).use_deep_channel is False
    assert ModelConfig.champion(clip_theta=3.0).clip_theta == 3.0


def test_champion_builds_and_needs_both_channel_observations(two_season_model_data,
                                                             engineered_two_season_df):
    """The champion requires deep AND xG observations; the base fixture has no
    deep, so it must fail loudly rather than silently drop a channel."""
    import numpy as np
    from football_model.data.prepare_model_data import prepare_model_data

    with pytest.raises(ValueError, match="use_deep_channel"):
        build_model(two_season_model_data, ModelConfig.champion())

    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    md = prepare_model_data(df, max_round=df["round"].max())
    m = build_model(md, ModelConfig.champion())
    assert {"xg_channel", "deep_channel"} <= set(m.named_vars)


def test_season_start_window_is_actually_wired(engineered_two_season_df, tmp_path):
    """WP034 found ModelConfig.season_start_window was DEAD: declared but never read.

    The FIRST version of this test only grepped the harness source for the string, so it
    passed while the wiring I had just added raised UnboundLocalError on every window --
    `config` was referenced above the line that builds it, and 46 CV fits died in 3s
    each. A test that greps source proves nothing. This one RUNS the harness.
    """
    import pickle
    import subprocess
    import sys
    from pathlib import Path as _P

    import numpy as np
    import pandas as pd

    repo = _P(__file__).resolve().parents[1]
    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    for c in ("deep_home", "deep_away"):
        df[c] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    for c in ("stakes_home", "stakes_away"):
        df[c] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df.stakes_home, df.stakes_away)
    rounds = sorted(df["round"].unique())
    shared = {"df_cv": df, "windows": [{"train_start": rounds[0], "train_end": rounds[-3],
                                        "test_start": rounds[-2], "test_end": rounds[-1]}]}
    data = tmp_path / "d.pkl"; data.write_bytes(pickle.dumps(shared))
    ck = tmp_path / "w001.pkl"
    out = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_cv_window.py"),
         "--data-path", str(data), "--checkpoint-path", str(ck), "--window-index", "1",
         "--use-xg", "false", "--use-dc", "false", "--predict-mode", "plugin",
         "--config-json", '{"season_start_window": 3}',
         "--draws", "8", "--tune", "8", "--chains", "1"],
        capture_output=True, text=True, timeout=1800,
        env={**__import__("os").environ, "PYTENSOR_FLAGS": "cxx="})
    assert out.returncode == 0, f"harness crashed with season_start_window set:\n{out.stdout[-2500:]}"
    assert ck.exists()


def test_wiring_is_behaviour_preserving_at_the_default():
    from football_model.types.model_data import ModelConfig
    import inspect
    from football_model.data.prepare_model_data import prepare_model_data
    assert ModelConfig().season_start_window == \
        inspect.signature(prepare_model_data).parameters["season_start_window"].default, \
        "defaults must match, or wiring it silently changes every past result"
