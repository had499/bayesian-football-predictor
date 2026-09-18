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
import concurrent.futures
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import poisson

from dataclasses import fields

from football_model.data.prepare_model_data import (
    prepare_model_data, prepare_multileague_data, max_round_for_cutoff_date,
)
from football_model.model.model import build_model, build_multileague_model
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
    lineup_dev_table: pd.DataFrame | None = None,
):
    train_data = prepare_model_data(
        df_cv, max_round=window["train_end"], lineup_dev_table=lineup_dev_table
    )

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

    # WP008: same story as beta_xG/rho_dc above — nothing about the lineup
    # covariate is applied at prediction time unless we do it explicitly
    # here, via the same predict.py functions that mirror it.
    beta_lineup_mean_cv = None
    if config.use_lineup_xg:
        beta_lineup_mean_cv = float(trace_cv.posterior["beta_lineup"].mean(dim=["chain", "draw"]).values)

    full_data = prepare_model_data(
        df_cv, max_round=window["test_end"], lineup_dev_table=lineup_dev_table
    )

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
        beta_lineup=beta_lineup_mean_cv,
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
        "beta_lineup": beta_lineup_mean_cv,  # None unless use_lineup_xg
    }
    return result, match_predictions


def run_window_multileague(
    dfs_by_league: dict,
    eval_league: str,
    window: dict,
    window_num: int,
    use_xg: bool = True,
    use_dc: bool = True,
    config_overrides: dict | None = None,
):
    """WP011: same shape of function as run_window, deliberately NOT sharing
    code with it (see build_multileague_model's own docstring for the same
    reasoning) — run_window is what every past WP's real CV numbers were
    produced by, and must not risk a regression from being refactored to
    share logic with a brand-new, much-less-battle-tested code path.

    `dfs_by_league`: {league_name: engineered_df}, one of which must be
    `eval_league` — the only one ever evaluated against test rounds (every
    other league is training-only context for the shared hyperparameters).
    `window`: the SAME dict shape as run_window's, with train_start/
    train_end/test_start/test_end expressed as `eval_league`'s own round
    numbers (e.g. straight from WP001's `windows` list) — other leagues'
    training cutoff is derived from the calendar DATE that round
    corresponds to in eval_league's own data (see max_round_for_cutoff_date
    for why a shared date, not a shared round number, is what's actually
    leakage-safe across competitions with different season structures).

    Does not support use_lineup_xg / use_per_team_sigma / xG opponent
    adjustment (build_multileague_model itself raises if asked) — WP011
    scope is the base hierarchical model only.
    """
    if eval_league not in dfs_by_league:
        raise ValueError(f"eval_league {eval_league!r} not in dfs_by_league keys {list(dfs_by_league)}")

    eval_df = dfs_by_league[eval_league]
    train_cutoff_date = eval_df.loc[eval_df["round"] <= window["train_end"], "datetime"].max()

    train_leagues = prepare_multileague_data(dfs_by_league, cutoff_date=train_cutoff_date)

    config_overrides = _validate_config_overrides(dict(config_overrides or {}))
    config = ModelConfig(
        clip_theta=5.0, center_team_strength=False, use_dixon_coles=use_dc, use_xG=use_xg,
        **config_overrides,
    )
    model_cv = build_multileague_model(train_leagues, config)

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

    attack_mean_cv = trace_cv.posterior[f"attack_{eval_league}"].mean(dim=["chain", "draw"]).values
    defense_mean_cv = trace_cv.posterior[f"defence_{eval_league}"].mean(dim=["chain", "draw"]).values
    home_adv_mean_cv = trace_cv.posterior[f"home_adv_{eval_league}"].mean(dim=["chain", "draw"]).values

    beta_xg_mean_cv = None
    if use_xg:
        beta_xg_mean_cv = float(trace_cv.posterior["beta_xG"].mean(dim=["chain", "draw"]).values)

    rho_dc_mean_cv = None
    if use_dc:
        rho_dc_mean_cv = float(trace_cv.posterior["rho_dc"].mean(dim=["chain", "draw"]).values)

    # Only eval_league is ever predicted/scored — exactly the same
    # single-league prepare_model_data + predict_rows call run_window
    # always used, because eval_league's trained posterior slice IS an
    # ordinary single-league ModelData's worth of attack/defence/home_adv
    # (see the WP011 cross-check test in tests/test_predict.py proving this
    # bit-for-bit). No other league's data is touched again after training.
    full_data = prepare_model_data(eval_df, max_round=window["test_end"])

    last_t_cv = attack_mean_cv.shape[0] - 1
    test_rows = np.where(
        (full_data.t_idx >= window["test_start"]) & (full_data.t_idx <= window["test_end"])
    )[0]

    test_lambda_home_cv, test_lambda_away_cv, _ = predict_rows(
        full_data, test_rows,
        attack=attack_mean_cv, defense=defense_mean_cv, home_adv=home_adv_mean_cv,
        clip_theta=config.clip_theta,
        beta_xG=beta_xg_mean_cv,
        max_t=last_t_cv,
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
            "rho_dc": rho_dc_mean_cv,
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
        ll_total_cv += dixon_coles_log_correction(
            test_lambda_home_cv, test_lambda_away_cv,
            test_goals_home_cv, test_goals_away_cv, rho_dc_mean_cv,
        ).sum()

    naive_lambda = eval_df[eval_df["round"] <= window["train_end"]]["goals_home"].mean()
    naive_ll_cv = (
        poisson.logpmf(test_goals_home_cv.astype(int), naive_lambda).sum()
        + poisson.logpmf(test_goals_away_cv.astype(int), naive_lambda).sum()
    )

    result = {
        "window": window_num,
        "train_rounds": f"{window['train_start']}-{window['train_end']}",
        "test_rounds": f"{window['test_start']}-{window['test_end']}",
        "n_train": int(sum(d.n_matches for d in train_leagues.values())),
        "n_test": len(test_goals_home_cv) + len(test_goals_away_cv),
        "mae": mae_cv,
        "log_likelihood": ll_total_cv,
        "ll_naive": naive_ll_cv,
        "ll_improvement": ll_total_cv - naive_ll_cv,
        "use_xg": use_xg,
        "use_dc": use_dc,
        "rho_dc": rho_dc_mean_cv,
        "config_overrides": config_overrides,
        "sigma_att_team": None,  # WP011 doesn't support use_per_team_sigma
        "sigma_def_team": None,
        "beta_lineup": None,     # WP011 doesn't support use_lineup_xg
        "eval_league": eval_league,
        "leagues_available": sorted(dfs_by_league),  # every league passed in
        "leagues_used": sorted(train_leagues),        # leagues ACTUALLY trained on this
                                                        # window (prepare_multileague_data
                                                        # may have skipped some — see its
                                                        # docstring — so this can be a
                                                        # strict subset of leagues_available,
                                                        # e.g. an early window whose cutoff
                                                        # predates a trimmed league's history)
        "train_cutoff_date": str(train_cutoff_date),
    }
    return result, match_predictions


def _load_checkpoint(path: Path) -> dict:
    return pickle.load(open(path, "rb")) if path.exists() else {"results": [], "cv_match_predictions": []}


def _merge_window_into_checkpoint(checkpoint_path: Path, window_checkpoint: dict, window_num: int):
    """Merge ONE window's result (from its own isolated temp checkpoint file)
    into the shared checkpoint at `checkpoint_path`. Must only ever be
    called from a single thread/process at a time — see
    run_windows_concurrent's docstring for why that's the actual
    correctness property this whole module needs, not the merge logic
    itself (which is a plain read-modify-write, same replace-or-append
    pattern main() has always used)."""
    main_ckpt = _load_checkpoint(checkpoint_path)
    main_ckpt["results"] = (
        [r for r in main_ckpt["results"] if r["window"] != window_num]
        + [r for r in window_checkpoint["results"] if r["window"] == window_num]
    )
    main_ckpt["cv_match_predictions"] = (
        [m for m in main_ckpt["cv_match_predictions"] if m["window"] != window_num]
        + [m for m in window_checkpoint["cv_match_predictions"] if m["window"] == window_num]
    )
    with open(checkpoint_path, "wb") as f:
        pickle.dump(main_ckpt, f)


def run_windows_concurrent(
    script, data_path, checkpoint_path, window_indices,
    config_overrides: dict | None = None, max_workers: int = 2, timeout: int = 1200,
    extra_args: list | None = None,
):
    """Run several CV windows concurrently instead of the sequential
    `for w in windows: subprocess.run(...)` loop every prior WP's notebook
    has used — each window is already an isolated subprocess by
    run_cv_window.py's own design (see this module's top docstring), so
    there's nothing unsafe about running several of them at once, EXCEPT
    one thing: `main()`'s own checkpoint read-modify-write (load the
    pickle, add/replace this window's entry, write it back) is NOT safe if
    two subprocesses point at the SAME checkpoint file concurrently — a
    classic lost-update race, where both read the same "before" state, both
    write their own "after", and whichever finishes last silently wins,
    dropping the other window's result with no error.

    Solved by giving each concurrent subprocess its OWN throwaway
    checkpoint file (so its read-modify-write only ever touches a file
    nothing else touches), then merging each one into the real
    `checkpoint_path` via `_merge_window_into_checkpoint` — deliberately
    only ever called here, in the main thread, as each future completes
    (concurrent.futures.as_completed + a plain for-loop), never inside a
    worker. That serializes the one step that actually needs it without
    serializing the expensive part (the sampling itself).

    `subprocess.run` releases the GIL while blocked on the child process
    (waiting on it is a syscall, not Python bytecode), so a
    ThreadPoolExecutor is the right tool here, not multiprocessing — the
    concurrency is in the OS processes already, threads just dispatch them.

    Skips windows already present in checkpoint_path's results, same
    resume behaviour as every prior WP's sequential loop. `max_workers`
    should stay conservative (2-3) unless you've checked your machine has
    the RAM for it — each window is a full NUTS run, and running too many
    at once risks swapping rather than saving time (see WP011's README).
    """
    checkpoint_path = Path(checkpoint_path)
    done = {r["window"] for r in _load_checkpoint(checkpoint_path)["results"]}
    todo = [w for w in window_indices if w not in done]
    if not todo:
        print("  nothing to do — every requested window is already in the checkpoint")
        return

    config_json = json.dumps(config_overrides or {})

    def _run_one(w):
        tmp_ckpt = checkpoint_path.parent / f".tmp_{checkpoint_path.stem}_w{w}.pkl"
        if tmp_ckpt.exists():
            tmp_ckpt.unlink()
        cmd = [
            sys.executable, str(script),
            "--data-path", str(data_path),
            "--checkpoint-path", str(tmp_ckpt),
            "--window-index", str(w),
            "--config-json", config_json,
        ] + list(extra_args or [])
        subprocess.run(cmd, timeout=timeout, check=True)
        return w, tmp_ckpt

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, w): w for w in todo}
        for fut in concurrent.futures.as_completed(futures):
            w = futures[fut]
            try:
                w_done, tmp_ckpt = fut.result()
                _merge_window_into_checkpoint(checkpoint_path, _load_checkpoint(tmp_ckpt), w_done)
                tmp_ckpt.unlink(missing_ok=True)
                print(f"  [window {w_done}] done and merged")
            except subprocess.TimeoutExpired:
                print(f"  [window {w}] TIMEOUT — skipped, re-run to retry")
            except subprocess.CalledProcessError as e:
                print(f"  [window {w}] FAILED (exit {e.returncode}) — skipped, re-run to retry")


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
    windows = shared["windows"]
    window = windows[args.window_index - 1]

    # WP011: optional, backward-compatible multi-league mode — a shared-data
    # pickle built for multi-league CV carries 'dfs_by_league'/'eval_league'
    # instead of (or alongside) the plain 'df_cv' every prior WP's pickle
    # has; every shared-data pickle before WP011 lacks 'dfs_by_league', so
    # .get() -> None routes it through the exact unchanged single-league
    # path below, same as always.
    dfs_by_league = shared.get("dfs_by_league")

    if dfs_by_league is not None:
        eval_league = shared["eval_league"]
        print(
            f"[window {args.window_index}/{len(windows)}] multi-league "
            f"({sorted(dfs_by_league)}, eval={eval_league}) "
            f"training rounds {window['train_start']}-{window['train_end']} "
            f"(use_xg={args.use_xg}, use_dc={args.use_dc}, overrides={config_overrides})"
        )
        result, match_predictions = run_window_multileague(
            dfs_by_league, eval_league, window, args.window_index,
            use_xg=args.use_xg, use_dc=args.use_dc, config_overrides=config_overrides,
        )
    else:
        df_cv = shared["df_cv"]
        # WP008: optional, backward-compatible — every shared-data pickle
        # before WP008 lacks this key, and get() -> None disables the
        # lineup covariate exactly like not passing it at all
        # (prepare_model_data's own default).
        lineup_dev_table = shared.get("lineup_dev_table")

        print(
            f"[window {args.window_index}/{len(windows)}] "
            f"training rounds {window['train_start']}-{window['train_end']} "
            f"(use_xg={args.use_xg}, use_dc={args.use_dc}, overrides={config_overrides})"
        )
        result, match_predictions = run_window(
            df_cv, window, args.window_index, use_xg=args.use_xg, use_dc=args.use_dc,
            config_overrides=config_overrides, lineup_dev_table=lineup_dev_table,
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
