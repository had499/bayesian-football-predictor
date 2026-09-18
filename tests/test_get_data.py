import unicodedata

import pandas as pd

from football_model.data.get_data import _normalize_team_names


def test_normalize_team_names_collapses_nfc_nfd_mismatch():
    """The concrete failure mode WP011 flagged: the same team name can
    arrive as NFC (single precomposed codepoint, e.g. 'e' + acute accent as
    U+00E9) or NFD (base letter + a separate combining accent codepoint) —
    displayed identically, but `==`/groupby/merge treat them as different
    strings, silently splitting one team into two. Build one name in each
    form and confirm normalization makes them equal."""
    nfc_name = unicodedata.normalize("NFC", "Muncheń")  # e + combining acute -> single precomposed char
    nfd_name = unicodedata.normalize("NFD", nfc_name)  # decompose back apart
    assert nfc_name != nfd_name  # the failure mode is real before normalizing

    df = pd.DataFrame({
        "team": [nfc_name, nfd_name],
        "opp_team": ["X", "X"],
        "team_long": [nfc_name, nfd_name],
        "opp_team_long": ["X", "X"],
    })
    out = _normalize_team_names(df)
    assert out["team"].nunique() == 1
    assert out["team_long"].nunique() == 1


def test_normalize_team_names_leaves_ascii_names_unchanged():
    """Real fetched data (checked empirically across all 5 leagues) is
    already plain ASCII -- this must be a true no-op for it, not a
    behaviour change."""
    df = pd.DataFrame({
        "team": ["ARS", "MCI"], "opp_team": ["MCI", "ARS"],
        "team_long": ["Arsenal", "Man City"], "opp_team_long": ["Man City", "Arsenal"],
    })
    out = _normalize_team_names(df.copy())
    pd.testing.assert_frame_equal(out, df)


def test_normalize_team_names_ignores_non_string_columns():
    """Only known team-name columns get touched -- goals/xG etc. must pass
    through untouched even if this is called on a full match dataframe."""
    df = pd.DataFrame({
        "team": ["ARS"], "opp_team": ["MCI"],
        "team_long": ["Arsenal"], "opp_team_long": ["Man City"],
        "goals": [2], "xG": [1.8],
    })
    out = _normalize_team_names(df.copy())
    assert out["goals"].iloc[0] == 2
    assert out["xG"].iloc[0] == 1.8
