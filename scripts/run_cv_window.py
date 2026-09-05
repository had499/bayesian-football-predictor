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
        --window-index <1-based index into the windows list>

`data-path` must be a pickle of {'df_cv': DataFrame, 'windows': list[dict]}.
`checkpoint-path` is read (if it exists) and rewritten with this window's
result appended/replaced — same schema the notebook already uses:
{'results': list[dict], 'cv_match_predictions': list[dict]}.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import poisson

from football_model.data.prepare_model_data import prepare_model_data
from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


def run_window(df_cv: pd.DataFrame, window: dict, window_num: int):
    train_data = prepare_model_data(df_cv, max_round=window["train_end"])

    # Same config as the notebook's main fit — keep CV honest to what's deployed.
    config = ModelConfig(
        clip_theta=5.0, center_team_strength=False, use_dixon_coles=True, use_xG=True
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

    df_cv = df_cv.copy()
    test_mask_cv = (
        (df_cv["round"] >= window["test_start"])
        & (df_cv["round"] <= window["test_end"])
        & (df_cv["is_home"] == 1)
    ).values
    test_indices_cv = np.where(test_mask_cv)[0]

    teams_cv = pd.unique(df_cv[["team", "opp_team"]].values.ravel())
    team_idx_map_cv = {t: i for i, t in enumerate(teams_cv)}
    df_cv["team_id"] = df_cv["team"].map(team_idx_map_cv)
    df_cv["opp_id"] = df_cv["opp_team"].map(team_idx_map_cv)

    last_t_cv = attack_mean_cv.shape[0] - 1
    test_lambda_home_cv = []
    test_lambda_away_cv = []

    for idx in test_indices_cv:
        team_id = int(df_cv.iloc[idx]["team_id"])  # always home (is_home==1 filter above)
        opp_id = int(df_cv.iloc[idx]["opp_id"])  # always away

        theta_home = (
            attack_mean_cv[last_t_cv, team_id]
            - defense_mean_cv[last_t_cv, opp_id]
            + home_adv_mean_cv[team_id]
        )
        theta_away = attack_mean_cv[last_t_cv, opp_id] - defense_mean_cv[last_t_cv, team_id]

        test_lambda_home_cv.append(np.exp(np.clip(theta_home, -5, 5)))
        test_lambda_away_cv.append(np.exp(np.clip(theta_away, -5, 5)))

    test_lambda_home_cv = np.array(test_lambda_home_cv)
    test_lambda_away_cv = np.array(test_lambda_away_cv)
    test_goals_home_cv = df_cv[test_mask_cv]["goals_home"].values
    test_goals_away_cv = df_cv[test_mask_cv]["goals_away"].values

    match_predictions = [
        {
            "window": window_num,
            "lambda_home": lh,
            "lambda_away": la,
            "goals_home": int(gh),
            "goals_away": int(ga),
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
    }
    return result, match_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--window-index", required=True, type=int, help="1-based index into the windows list"
    )
    args = parser.parse_args()

    with open(args.data_path, "rb") as f:
        shared = pickle.load(f)
    df_cv = shared["df_cv"]
    windows = shared["windows"]
    window = windows[args.window_index - 1]

    print(
        f"[window {args.window_index}/{len(windows)}] "
        f"training rounds {window['train_start']}-{window['train_end']}"
    )
    result, match_predictions = run_window(df_cv, window, args.window_index)
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
