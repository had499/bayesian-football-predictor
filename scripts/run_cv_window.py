"""Run one rolling-CV window's fit + evaluation as a standalone process.

Each CV window gets a fresh interpreter/process so nothing (JAX compiled
programs, thread-pool state, memory) can accumulate across windows the way
it can when many windows run sequentially inside one long-lived Jupyter
kernel — that accumulation is the suspected cause of the notebook's
mid-sweep stalls when all 35+ windows ran in-process.

Usage:
    python scripts/run_cv_window.py \
        --data-path <shared_data.pkl> \
        --checkpoint-path <checkpoint.pkl> \
        --window-index <1-based index into the windows list> \
        [--use-xg true|false] [--use-dc true|false] \
        [--config-json '{"init_scale": 0.35, "home_adv_sd": 0.05}']

`data-path` must be a pickle of {'df_cv': DataFrame, 'windows': list[dict]}.
`checkpoint-path` is read (if it exists) and rewritten with this window's
result appended/replaced — same schema the notebook already uses:
{'results': list[dict], 'cv_match_predictions': list[dict]}.
`--use-xg`/`--use-dc` default to true/true (matches WP001's config) —
override both to false for ablation runs (WP002).
`--config-json` is an optional JSON object of extra ModelConfig field
overrides applied on top of the base config — the WP005 prior-loosening
sweep uses it to vary init_scale / home_adv_sd / sigma_att / rho_att_alpha
etc. one knob at a time. The overrides are recorded verbatim in each
result row so a checkpoint always says which arm it belongs to.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import poisson

from dataclasses import fields

from football_model.data.prepare_model_data import prepare_model_data
from football_model.model.model import build_model
from football_model.model.predict import dixon_coles_log_correction, predict_rows
from football_model.types.model_data import ModelConfig


def _validate_config_overrides(overrides: dict) -> dict:
    """Reject unknown keys up front — a typo'd override would otherwise be
    silently dropped by the dataclass and the whole sweep arm would run the
    baseline config without anyone noticing."""
    valid = {f.name for f in fields(ModelConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"unknown ModelConfig override(s): {sorted(unknown)}")
    return overrides


def run_window(
    df_cv: pd.DataFrame,
    window: dict,
    window_num: int,
    use_xg: bool = True,
    use_dc: bool = True,
    config_overrides: dict | None = None,
):
    train_data = prepare_model_data(df_cv, max_round=window["train_end"])

    config_overrides = _validate_config_overrides(dict(config_overrides or {}))
    config = ModelConfig(
        clip_theta=5.0, center_team_strength=False, use_dixon_coles=use_dc, use_xG=use_xg,
        **config_overrides,
    )
    model_cv = build_model(train_data, config)

    with model_cv:
        trace_cv = pm.sample(
            2000,
            tune=2000,
            chains=3,
            target_accept=0.95,
            random_seed=42 + window_num,
            return_inferencedata=True,
            discard_tuned_samples=True,
            nuts_sampler="numpyro",
        )

    attack_mean_cv = trace_cv.posterior["attack"].mean(dim=["chain", "draw"]).values
    defense_mean_cv = trace_cv.posterior["defence"].mean(dim=["chain", "draw"]).values
    home_adv_mean_cv = trace_cv.posterior["home_adv"].mean(dim=["chain", "draw"]).values

    # If trained with xG, predictions must include it too — otherwise attack/
    # defense (fit assuming part of the signal is carried by beta_xG * log(xG))
    # get scored as if they were the whole story, systematically understating
    # the model. `full_data` (train+test rounds) gives us test rows' t_idx/
    # team_idx/opp_idx/xG/goals straight from prepare_model_data — the one
    # place that owns this bookkeeping (see football_model.model.predict's
    # module docstring for why nothing else may hand-rebuild it).
    beta_xg_mean_cv = None
    if use_xg:
        beta_xg_mean_cv = float(trace_cv.posterior["beta_xG"].mean(dim=["chain", "draw"]).values)

    # Same story for Dixon-Coles: dixon_coles_adjustment only ever shapes the
    # TRAINING posterior (a pm.Potential inside build_model) — nothing
    # evaluates the actual tau(goals; lambda, rho) correction it exists for
    # unless we do it explicitly here too, via the same predict.py functions
    # that mirror it (see that module's docstring).
    rho_dc_mean_cv = None
    if use_dc:
        rho_dc_mean_cv = float(trace_cv.posterior["rho_dc"].mean(dim=["chain", "draw"]).values)

    # WP006: with use_per_team_sigma, each team's own fitted innovation SD is
    # worth keeping — the whole point of that config is to check whether it
    # lands higher on teams you'd expect (promoted sides, a managerial
    # change) rather than just improving the score. Doesn't affect
    # prediction (attack/defense already encode whatever sigma produced
    # them); this is purely for that diagnostic.
    sigma_att_team_map = sigma_def_team_map = None
    if config.use_per_team_sigma:
        sigma_att_team_mean_cv = trace_cv.posterior["sigma_att_team"].mean(dim=["chain", "draw"]).values
        sigma_def_team_mean_cv = trace_cv.posterior["sigma_def_team"].mean(dim=["chain", "draw"]).values
        sigma_att_team_map = {t: float(sigma_att_team_mean_cv[i]) for t, i in train_data.team_mapping.items()}
        sigma_def_team_map = {t: float(sigma_def_team_mean_cv[i]) for t, i in train_data.team_mapping.items()}

    full_data = prepare_model_data(df_cv, max_round=window["test_end"])

    last_t_cv = attack_mean_cv.shape[0] - 1
    test_rows = np.where(
        (full_data.t_idx >= window["test_start"]) & (full_data.t_idx <= window["test_end"])
    )[0]

    # Same formula football_model.model.model.build_model trains with — see
    # football_model.model.predict for why this is a shared function instead
    # of hand-rederiving theta (and the team/time lookups it needs) here.
    # max_t freezes attack/defense at the last trained round: test rounds lie
    # beyond where the AR1 process was actually fit, so we hold latent state
    # at its final trained value rather than reading (nonexistent) later t.
    test_lambda_home_cv, test_lambda_away_cv, _ = predict_rows(
        full_data, test_rows,
        attack=attack_mean_cv, defense=defense_mean_cv, home_adv=home_adv_mean_cv,
        clip_theta=config.clip_theta,
        beta_xG=beta_xg_mean_cv,
        max_t=last_t_cv,
        # Currently always False/0.3 (config's own defaults) in this harness,
        # but pass them through explicitly rather than relying on predict_rows'
        # defaults matching config's — that's exactly the assumption that made
        # the Dixon-Coles gap silent for as long as it was.
        use_opponent_adjusted_xG=config.use_opponent_adjusted_xG,
        xG_adjustment_strength=config.xG_adjustment_strength,
    )
    test_goals_home_cv = full_data.goals_home[test_rows]
    test_goals_away_cv = full_data.goals_away[test_rows]

    match_predictions = [
        {
            "window": window_num,
            "lambda_home": lh,
            "lambda_away": la,
            "goals_home": int(gh),
            "goals_away": int(ga),
            "rho_dc": rho_dc_mean_cv,  # None unless use_dc — carried per match so
                                       # downstream RPS/calibration can apply the
                                       # same correction training actually used.
        }
        for lh, la, gh, ga in zip(
            test_lambda_home_cv, test_lambda_away_cv, test_goals_home_cv, test_goals_away_cv
        )
    ]

    mae_home_cv = np.mean(np.abs(test_lambda_home_cv - test_goals_home_cv))
    mae_away_cv = np.mean(np.abs(test_lambda_away_cv - test_goals_away_cv))
    mae_cv = (mae_home_cv + mae_away_cv) / 2

    ll_home_cv = poisson.logpmf(test_goals_home_cv.astype(int), test_lambda_home_cv).sum()
    ll_away_cv = poisson.logpmf(test_goals_away_cv.astype(int), test_lambda_away_cv).sum()
    ll_total_cv = ll_home_cv + ll_away_cv

    if use_dc:
        # Add the same log(tau) correction dixon_coles_adjustment adds during
        # training (as a pm.Potential), so held-out LL reflects the model
        # use_dixon_coles=True actually specifies, not just its independent-
        # Poisson piece. Naive baseline below has no DC concept — left as
        # plain independent Poisson, unaffected.
        ll_total_cv += dixon_coles_log_correction(
            test_lambda_home_cv, test_lambda_away_cv,
            test_goals_home_cv, test_goals_away_cv, rho_dc_mean_cv,
        ).sum()

    naive_lambda = df_cv[df_cv["round"] <= window["train_end"]]["goals_home"].mean()
    naive_ll_cv = (
        poisson.logpmf(test_goals_home_cv.astype(int), naive_lambda).sum()
        + poisson.logpmf(test_goals_away_cv.astype(int), naive_lambda).sum()
    )

    result = {
        "window": window_num,
        "train_rounds": f"{window['train_start']}-{window['train_end']}",
        "test_rounds": f"{window['test_start']}-{window['test_end']}",
        "n_train": len(train_data.goals_home),
        "n_test": len(test_goals_home_cv) + len(test_goals_away_cv),
        "mae": mae_cv,
        "log_likelihood": ll_total_cv,
        "ll_naive": naive_ll_cv,
        "ll_improvement": ll_total_cv - naive_ll_cv,
        "use_xg": use_xg,
        "use_dc": use_dc,
        "rho_dc": rho_dc_mean_cv,
        "config_overrides": config_overrides,
        "sigma_att_team": sigma_att_team_map,  # None unless use_per_team_sigma
        "sigma_def_team": sigma_def_team_map,
    }
    return result, match_predictions


def _str2bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--window-index", required=True, type=int, help="1-based index into the windows list"
    )
    parser.add_argument("--use-xg", type=_str2bool, default=True)
    parser.add_argument("--use-dc", type=_str2bool, default=True)
    parser.add_argument(
        "--config-json", default="{}",
        help='JSON object of extra ModelConfig overrides, e.g. \'{"init_scale": 0.35}\'',
    )
    args = parser.parse_args()

    config_overrides = _validate_config_overrides(json.loads(args.config_json))

    with open(args.data_path, "rb") as f:
        shared = pickle.load(f)
    df_cv = shared["df_cv"]
    windows = shared["windows"]
    window = windows[args.window_index - 1]

    print(
        f"[window {args.window_index}/{len(windows)}] "
        f"training rounds {window['train_start']}-{window['train_end']} "
        f"(use_xg={args.use_xg}, use_dc={args.use_dc}, overrides={config_overrides})"
    )
    result, match_predictions = run_window(
        df_cv, window, args.window_index, use_xg=args.use_xg, use_dc=args.use_dc,
        config_overrides=config_overrides,
    )
    print(
        f"[window {args.window_index}] MAE={result['mae']:.3f} "
        f"LL_improvement={result['ll_improvement']:.2f}"
    )

    checkpoint_path = Path(args.checkpoint_path)
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
    else:
        checkpoint = {"results": [], "cv_match_predictions": []}

    # Replace this window's entry if it exists (e.g. a retried window),
    # otherwise append.
    checkpoint["results"] = [
        r for r in checkpoint["results"] if r["window"] != args.window_index
    ]
    checkpoint["results"].append(result)
    checkpoint["cv_match_predictions"] = [
        m for m in checkpoint["cv_match_predictions"] if m["window"] != args.window_index
    ] + match_predictions

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"[window {args.window_index}] checkpoint updated: {checkpoint_path}")


if __name__ == "__main__":
    main()
