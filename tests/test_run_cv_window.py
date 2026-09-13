import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_cv_window import _validate_config_overrides  # noqa: E402

from football_model.types.model_data import ModelConfig  # noqa: E402


def test_validate_accepts_known_modelconfig_fields():
    ov = {"init_scale": 0.35, "home_adv_sd": 0.05, "sigma_att": 0.02, "rho_att_alpha": 19.0}
    assert _validate_config_overrides(ov) == ov
    # and the overrides really do land on a ModelConfig
    cfg = ModelConfig(**ov)
    assert cfg.init_scale == 0.35 and cfg.home_adv_sd == 0.05


def test_validate_rejects_unknown_key_instead_of_silently_dropping_it():
    with pytest.raises(ValueError, match="unknown ModelConfig override"):
        _validate_config_overrides({"init_scale": 0.35, "innit_scale": 0.5})


def test_empty_override_json_is_a_noop():
    assert _validate_config_overrides(json.loads("{}")) == {}
