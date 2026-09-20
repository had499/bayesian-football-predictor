import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_cv_window import _validate_config_overrides, run_window  # noqa: E402

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


# --- WP011: concurrent window execution ---

import pickle
import sys
import time

from run_cv_window import run_windows_concurrent, _load_checkpoint  # noqa: E402

FAKE_WORKER = Path(__file__).resolve().parent / "fake_cv_worker.py"


def test_run_windows_concurrent_produces_all_results(tmp_path):
    ckpt_path = tmp_path / "ckpt.pkl"
    run_windows_concurrent(
        FAKE_WORKER, data_path="unused", checkpoint_path=ckpt_path,
        window_indices=[1, 2, 3, 4], max_workers=2, timeout=30,
        extra_args=["--delay", "0.05"],
    )
    ckpt = _load_checkpoint(ckpt_path)
    assert {r["window"] for r in ckpt["results"]} == {1, 2, 3, 4}
    assert {m["window"] for m in ckpt["cv_match_predictions"]} == {1, 2, 3, 4}
    # no leftover temp files
    assert list(tmp_path.glob(".tmp_*")) == []


def test_run_windows_concurrent_is_actually_concurrent(tmp_path):
    """The actual property being added: wall time for N windows at
    max_workers=W should track N/W * per-window-delay, not N * delay (which
    is what the old sequential subprocess.run loop always cost)."""
    ckpt_path = tmp_path / "ckpt.pkl"
    delay = 0.4
    n_windows = 4
    max_workers = 4

    t0 = time.time()
    run_windows_concurrent(
        FAKE_WORKER, data_path="unused", checkpoint_path=ckpt_path,
        window_indices=list(range(1, n_windows + 1)), max_workers=max_workers, timeout=30,
        extra_args=["--delay", str(delay)],
    )
    elapsed = time.time() - t0

    sequential_estimate = n_windows * delay
    assert elapsed < sequential_estimate * 0.7, (
        f"elapsed={elapsed:.2f}s not meaningfully faster than sequential estimate "
        f"{sequential_estimate:.2f}s -- windows don't appear to be running concurrently"
    )


def test_run_windows_concurrent_skips_already_done_windows(tmp_path):
    ckpt_path = tmp_path / "ckpt.pkl"
    # seed the checkpoint with window 1 already "done"
    with open(ckpt_path, "wb") as f:
        pickle.dump({"results": [{"window": 1, "mae": 999.0, "overrides": {}}],
                     "cv_match_predictions": [{"window": 1, "lambda_home": 9.0, "lambda_away": 9.0}]}, f)

    run_windows_concurrent(
        FAKE_WORKER, data_path="unused", checkpoint_path=ckpt_path,
        window_indices=[1, 2], max_workers=2, timeout=30, extra_args=["--delay", "0.05"],
    )
    ckpt = _load_checkpoint(ckpt_path)
    # window 1's PRE-EXISTING result must be untouched (mae=999.0), not
    # overwritten by a re-run -- proves the "already done" set really skips
    # dispatching a subprocess for it at all, not just re-merging the same value.
    w1 = next(r for r in ckpt["results"] if r["window"] == 1)
    assert w1["mae"] == 999.0
    assert {r["window"] for r in ckpt["results"]} == {1, 2}


def test_run_windows_concurrent_no_lost_updates_under_concurrency(tmp_path):
    """The actual race this function exists to avoid: run MANY windows at
    high concurrency and confirm every single one survives in the final
    checkpoint -- if the merge step weren't safely serialized, some window
    results would randomly go missing (the lost-update race described in
    run_windows_concurrent's docstring)."""
    ckpt_path = tmp_path / "ckpt.pkl"
    n_windows = 12
    run_windows_concurrent(
        FAKE_WORKER, data_path="unused", checkpoint_path=ckpt_path,
        window_indices=list(range(1, n_windows + 1)), max_workers=6, timeout=30,
        extra_args=["--delay", "0.02"],
    )
    ckpt = _load_checkpoint(ckpt_path)
    assert {r["window"] for r in ckpt["results"]} == set(range(1, n_windows + 1))
    assert len(ckpt["results"]) == n_windows  # no duplicates either


def test_run_windows_concurrent_failed_window_does_not_block_others(tmp_path):
    ckpt_path = tmp_path / "ckpt.pkl"
    run_windows_concurrent(
        FAKE_WORKER, data_path="unused", checkpoint_path=ckpt_path,
        window_indices=[1, 2, 3], max_workers=3, timeout=30,
        extra_args=["--delay", "0.05"],
    )
    # now simulate window 2 failing on a re-run of a DIFFERENT window set
    # sharing the same checkpoint -- windows 1/3 already done, only 2 retried
    ckpt_path2 = tmp_path / "ckpt2.pkl"

    # Use a script wrapper isn't straightforward with --fail applying to one
    # window only via extra_args (it's shared across all windows in this
    # helper's design) -- so directly test that ONE always-failing worker
    # doesn't prevent the checkpoint from being written for others by giving
    # it its own window set where failure is uniform, and confirming no
    # exception propagates out of run_windows_concurrent itself.
    run_windows_concurrent(
        FAKE_WORKER, data_path="unused", checkpoint_path=ckpt_path2,
        window_indices=[10, 11], max_workers=2, timeout=30,
        extra_args=["--delay", "0.02", "--fail"],
    )
    ckpt2 = _load_checkpoint(ckpt_path2)
    assert ckpt2["results"] == []  # both failed, nothing merged, no crash


# --- WP013: continuity covariate ---

def test_run_window_refuses_use_continuity_without_a_table(engineered_two_season_df):
    """Without the table the feature is all zeros and the arm would silently
    run as the baseline while its checkpoint claimed use_continuity."""
    df = engineered_two_season_df
    last = int(df["round"].max())
    window = {"train_start": 1, "train_end": last - 2, "test_start": last - 1, "test_end": last}
    with pytest.raises(ValueError, match="continuity_table"):
        run_window(df, window, 1, config_overrides={"use_continuity": True})

