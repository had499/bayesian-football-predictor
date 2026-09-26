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
`--predict-mode plugin|posterior` (default `plugin`, the WP001-WP006
behaviour: score from posterior MEANS at the frozen last-trained latent
state). `posterior` (WP007) scores the posterior AVERAGE of per-draw
probabilities and rolls the latent state forward through the forecast
horizon (football_model.model.predict.posterior_lambdas). Both modes record
the same per-match schema, including home/draw/away probabilities, so
downstream analysis doesn't care which produced a checkpoint.
`--draws/--tune/--chains/--target-accept` override the sampler settings
(defaults 2000/2000/3/0.95 — unchanged).
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

import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import poisson

from dataclasses import fields

from football_model.data.prepare_model_data import prepare_model_data
from football_model.model.model import build_model
from football_model.model.predict import (
    dc_outcome_probs,
    dixon_coles_log_correction,
    posterior_goals_loglik,
    posterior_lambdas,
    posterior_outcome_probs,
    predict_rows,
    ranked_probability_score,
)
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
    predict_mode: str = "plugin",
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 3,
    target_accept: float = 0.95,
):
    if predict_mode not in ("plugin", "posterior"):
        raise ValueError(f"predict_mode must be 'plugin' or 'posterior', got {predict_mode!r}")
    t_start = time.time()
    train_data = prepare_model_data(df_cv, max_round=window["train_end"])

    config_overrides = _validate_config_overrides(dict(config_overrides or {}))
    # Overrides win over the harness defaults (previously a duplicate key
    # such as center_team_strength raised a TypeError instead).
    config = ModelConfig(**{
        "clip_theta": 5.0, "center_team_strength": False,
        "use_dixon_coles": use_dc, "use_xG": use_xg,
        **config_overrides,
    })
    model_cv = build_model(train_data, config)

    with model_cv:
        trace_cv = pm.sample(
            draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=42 + window_num,
            return_inferencedata=True,
            discard_tuned_samples=True,
            nuts_sampler="numpyro",
        )
    n_diverging = int(trace_cv.sample_stats["diverging"].sum())
    scalar_names = [v for v in ("sigma_att", "sigma_def", "home_mu", "intercept", "promo_att",
                                "promo_def", "xg_shape", "rho_dc", "beta_xG", "rho_att", "rho_def",
                                "beta_absence")
                    if v in trace_cv.posterior]
    max_rhat = float(max(
        (float(az.rhat(trace_cv, var_names=[v])[v].max()) for v in scalar_names), default=float("nan")
    ))
    # Posterior means of every scalar parameter the config actually has — lets
    # a work product report what the model LEARNED (e.g. the promotion shift,
    # the walk's sigma, xg_shape), not just how it scored.
    param_means = {
        v: float(trace_cv.posterior[v].mean())
        for v in scalar_names + ["home_sd", "xg_kappa"] if v in trace_cv.posterior
    }

    attack_mean_cv = trace_cv.posterior["attack"].mean(dim=["chain", "draw"]).values
    defense_mean_cv = trace_cv.posterior["defence"].mean(dim=["chain", "draw"]).values
    home_adv_mean_cv = trace_cv.posterior["home_adv"].mean(dim=["chain", "draw"]).values

    # If trained with xG (the covariate), predictions must include it too —
    # otherwise attack/defense get scored as if they were the whole story.
    # `full_data` (train+test rounds) gives test rows' t_idx/team_idx/
    # opp_idx/xG/goals straight from prepare_model_data — the one place that
    # owns this bookkeeping (see football_model.model.predict's docstring).
    full_data = prepare_model_data(df_cv, max_round=window["test_end"])
    last_t_cv = trace_cv.posterior["attack"].shape[2] - 1
    test_rows = np.where(
        (full_data.t_idx >= window["test_start"]) & (full_data.t_idx <= window["test_end"])
    )[0]
    test_goals_home_cv = full_data.goals_home[test_rows]
    test_goals_away_cv = full_data.goals_away[test_rows]
    outcome_cv = np.where(test_goals_home_cv > test_goals_away_cv, 0,
                          np.where(test_goals_home_cv == test_goals_away_cv, 1, 2))

    def _post_mean(name):
        return trace_cv.posterior[name].mean(dim=["chain", "draw"]).values

    rho_dc_mean_cv = float(_post_mean("rho_dc")) if use_dc else None

    # WP006: with use_per_team_sigma, each team's own fitted innovation SD is
    # worth keeping (diagnostic only — attack/defense already encode it).
    sigma_att_team_map = sigma_def_team_map = None
    if config.use_per_team_sigma:
        sigma_att_team_mean_cv, sigma_def_team_mean_cv = _post_mean("sigma_att_team"), _post_mean("sigma_def_team")
        sigma_att_team_map = {t: float(sigma_att_team_mean_cv[i]) for t, i in train_data.team_mapping.items()}
        sigma_def_team_map = {t: float(sigma_def_team_mean_cv[i]) for t, i in train_data.team_mapping.items()}

    if predict_mode == "plugin":
        # WP001-WP006 behaviour: one lambda per side from posterior means,
        # latent state frozen at the last trained round (max_t).
        beta_xg_mean_cv = float(_post_mean("beta_xG")) if config.use_xG else None
        test_lambda_home_cv, test_lambda_away_cv, _ = predict_rows(
            full_data, test_rows,
            attack=_post_mean("attack"), defense=_post_mean("defence"), home_adv=_post_mean("home_adv"),
            clip_theta=config.clip_theta,
            beta_xG=beta_xg_mean_cv,
            max_t=last_t_cv,
            use_opponent_adjusted_xG=config.use_opponent_adjusted_xG,
            xG_adjustment_strength=config.xG_adjustment_strength,
            intercept=float(_post_mean("intercept")) if config.use_intercept else 0.0,
            beta_absence=float(_post_mean("beta_absence")) if config.use_absence else None,
        )
        probs_cv = np.array([
            dc_outcome_probs(lh, la, rho=rho_dc_mean_cv)
            for lh, la in zip(test_lambda_home_cv, test_lambda_away_cv)
        ])
        ll_match_cv = (
            poisson.logpmf(test_goals_home_cv.astype(int), test_lambda_home_cv)
            + poisson.logpmf(test_goals_away_cv.astype(int), test_lambda_away_cv)
        )
        if use_dc:
            # same log(tau) correction dixon_coles_adjustment adds in training
            ll_match_cv = ll_match_cv + dixon_coles_log_correction(
                test_lambda_home_cv, test_lambda_away_cv,
                test_goals_home_cv, test_goals_away_cv, rho_dc_mean_cv,
            )
    else:
        # WP007: posterior-averaged, forecast-horizon-aware.
        wanted = ["attack", "defence", "attack_raw", "defence_raw", "home_adv", "sigma_att", "sigma_def",
                  "sigma_att_team", "sigma_def_team", "rho_att", "rho_def", "promo_att", "promo_def",
                  "att_0", "def_0", "intercept", "beta_xG", "rho_dc", "beta_absence"]
        post = {}
        for name in wanted:
            if name in trace_cv.posterior:
                v = trace_cv.posterior[name].values
                post[name] = v.reshape((v.shape[0] * v.shape[1],) + v.shape[2:])
        n_keep = min(400, next(iter(post.values())).shape[0])
        keep = np.linspace(0, next(iter(post.values())).shape[0] - 1, n_keep).astype(int)
        post = {k: v[keep] for k, v in post.items()}   # subsample ONCE so rho_dc stays aligned with the lambdas
        lam_h_draws, lam_a_draws = posterior_lambdas(
            full_data, test_rows, post, config.clip_theta,
            last_t=last_t_cv, season_start_sigma_mult=config.season_start_sigma_mult,
            n_draws=n_keep, seed=window_num, propagate=True,
            active_last=train_data.active_mask[last_t_cv],
        )
        rho_draws = post.get("rho_dc") if use_dc else None
        probs_cv = posterior_outcome_probs(lam_h_draws, lam_a_draws, rho_draws)
        ll_match_cv = posterior_goals_loglik(
            lam_h_draws, lam_a_draws, test_goals_home_cv, test_goals_away_cv, rho_draws
        )
        test_lambda_home_cv, test_lambda_away_cv = lam_h_draws.mean(0), lam_a_draws.mean(0)

    rps_cv = ranked_probability_score(probs_cv, outcome_cv)

    # Record who played and when, straight off ModelData (the one place that
    # owns the index bookkeeping), so downstream joins to odds/other data key
    # on real identities instead of relying on row order.
    code_of = {i: t for t, i in full_data.team_mapping.items()}
    home_codes = [code_of[int(i)] for i in full_data.team_idx[test_rows]]
    away_codes = [code_of[int(i)] for i in full_data.opp_idx[test_rows]]
    rounds_cv = full_data.t_idx[test_rows].astype(int)

    match_predictions = [
        {
            "window": window_num,
            "home": hc, "away": ac, "round": int(rd),
            "lambda_home": lh,
            "lambda_away": la,
            "goals_home": int(gh),
            "goals_away": int(ga),
            "rho_dc": rho_dc_mean_cv,  # None unless use_dc
            "p_home": float(p[0]), "p_draw": float(p[1]), "p_away": float(p[2]),
            "rps": float(r),
            "ll_goals": float(ll),
        }
        for hc, ac, rd, lh, la, gh, ga, p, r, ll in zip(
            home_codes, away_codes, rounds_cv,
            test_lambda_home_cv, test_lambda_away_cv, test_goals_home_cv, test_goals_away_cv,
            probs_cv, rps_cv, ll_match_cv,
        )
    ]

    mae_home_cv = np.mean(np.abs(test_lambda_home_cv - test_goals_home_cv))
    mae_away_cv = np.mean(np.abs(test_lambda_away_cv - test_goals_away_cv))
    mae_cv = (mae_home_cv + mae_away_cv) / 2
    ll_total_cv = float(ll_match_cv.sum())

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
        "predict_mode": predict_mode,
        "rps": float(rps_cv.mean()),
        "n_diverging": n_diverging,
        "max_rhat": max_rhat,
        "param_means": param_means,
        "seconds": time.time() - t_start,
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
    parser.add_argument("--predict-mode", choices=["plugin", "posterior"], default="plugin")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--target-accept", type=float, default=0.95)
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
        config_overrides=config_overrides, predict_mode=args.predict_mode,
        draws=args.draws, tune=args.tune, chains=args.chains, target_accept=args.target_accept,
    )
    print(
        f"[window {args.window_index}] MAE={result['mae']:.3f} "
        f"LL_improvement={result['ll_improvement']:.2f} RPS={result['rps']:.4f} "
        f"div={result['n_diverging']} rhat={result['max_rhat']:.3f} {result['seconds']:.0f}s"
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
