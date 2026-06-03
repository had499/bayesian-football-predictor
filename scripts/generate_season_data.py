"""
Generate predictions and actuals for every completed gameweek in the season.

For each round N:
  - Trains model on rounds 1..N-1
  - Saves predictions for round N  -> data/gameweeks/round_{N}/predictions.json
  - Saves actuals for round N       -> data/gameweeks/round_{N}/actuals.json

The trace is discarded after each round to keep storage low.

Usage:
    python scripts/generate_season_data.py
    python scripts/generate_season_data.py --leagues EPL --years 2024 --start-round 5
"""

import sys
import os
import json
import argparse
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pymc as pm
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from football_model.data.get_data import get_understat_data
from football_model.features.add_metadata import add_rounds_to_data, add_home_away_goals_xg, add_match_ids
from football_model.data.prepare_model_data import prepare_model_data
from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


DEFAULT_SAMPLES = 5000
DEFAULT_TUNE = 2000
DEFAULT_CHAINS = 4
DEFAULT_TARGET_ACCEPT = 0.97


def get_completed_rounds(df: pd.DataFrame) -> list[int]:
    """Return rounds where every match has already been played."""
    now = datetime.today()
    round_status = df.groupby("round")["datetime"].max()
    return sorted(r for r, last_dt in round_status.items() if last_dt <= now)


def extract_actuals(df: pd.DataFrame, round_num: int) -> list[dict]:
    """Extract actual results for a given round (one row per match)."""
    round_df = df[(df["round"] == round_num) & (df["is_home"] == 1)].copy()
    actuals = []
    for _, row in round_df.iterrows():
        actuals.append({
            "home_team": row["team"],
            "away_team": row["opp_team"],
            "goals_home": int(row["goals"]),
            "goals_away": int(row["goals_against"]),
        })
    return actuals


def extract_team_strengths(trace, model_data) -> dict:
    """Extract attack, defence, and home_adv posterior summaries for each team over time."""
    attack_samples = trace.posterior["attack"].values      # (chains, draws, time, teams)
    defense_samples = trace.posterior["defence"].values

    n_chains, n_draws = attack_samples.shape[:2]
    attack_flat = attack_samples.reshape(n_chains * n_draws, *attack_samples.shape[2:])   # (samples, time, teams)
    defense_flat = defense_samples.reshape(n_chains * n_draws, *defense_samples.shape[2:])

    has_home_adv = "home_adv" in trace.posterior
    if has_home_adv:
        home_adv_flat = trace.posterior["home_adv"].values.reshape(
            n_chains * n_draws, trace.posterior["home_adv"].values.shape[2]
        )

    idx_to_team = {v: k for k, v in model_data.team_mapping.items()}
    n_time = attack_flat.shape[1]

    teams = {}
    for team_idx, team_name in idx_to_team.items():
        attack_over_time = []
        defence_over_time = []
        for t in range(n_time):
            a = attack_flat[:, t, team_idx]
            d = defense_flat[:, t, team_idx]
            attack_over_time.append({
                "t": t,
                "mean": round(float(a.mean()), 4),
                "p5": round(float(np.percentile(a, 5)), 4),
                "p95": round(float(np.percentile(a, 95)), 4),
            })
            defence_over_time.append({
                "t": t,
                "mean": round(float(d.mean()), 4),
                "p5": round(float(np.percentile(d, 5)), 4),
                "p95": round(float(np.percentile(d, 95)), 4),
            })

        entry = {"attack": attack_over_time, "defence": defence_over_time}

        if has_home_adv:
            h = home_adv_flat[:, team_idx]
            entry["home_adv"] = {
                "mean": round(float(h.mean()), 4),
                "p5": round(float(np.percentile(h, 5)), 4),
                "p95": round(float(np.percentile(h, 95)), 4),
            }

        teams[team_name] = entry

    return teams


def compute_predictions(trace, train_model_data, pred_model_data, round_num: int) -> list[dict]:
    """Compute match predictions for round_num using the posterior trace."""
    attack_samples = trace.posterior["attack"].values        # (chains, draws, time, teams)
    defense_samples = trace.posterior["defence"].values

    n_chains, n_draws = attack_samples.shape[:2]
    attack_flat = attack_samples.reshape(n_chains * n_draws, *attack_samples.shape[2:])
    defense_flat = defense_samples.reshape(n_chains * n_draws, *defense_samples.shape[2:])

    if "home_adv" in trace.posterior:
        home_adv_flat = trace.posterior["home_adv"].values.reshape(
            n_chains * n_draws, trace.posterior["home_adv"].values.shape[2]
        )
    else:
        home_adv_flat = None

    idx_to_team = {v: k for k, v in pred_model_data.team_mapping.items()}
    pred_mask = pred_model_data.t_idx == round_num

    if not pred_mask.any():
        return []

    predictions = []
    seen_matches = set()

    for i in np.where(pred_mask)[0]:
        team_idx = pred_model_data.team_idx[i]
        opp_idx = pred_model_data.opp_idx[i]
        is_home = pred_model_data.home[i]

        home_idx, away_idx = (team_idx, opp_idx) if is_home == 1 else (opp_idx, team_idx)

        match_key = tuple(sorted([home_idx, away_idx]))
        if match_key in seen_matches:
            continue
        seen_matches.add(match_key)

        home_team = idx_to_team[home_idx]
        away_team = idx_to_team[away_idx]

        t = min(round_num - 1, attack_flat.shape[1] - 1)

        home_attack = attack_flat[:, t, home_idx]
        away_attack = attack_flat[:, t, away_idx]
        home_defense = defense_flat[:, t, home_idx]
        away_defense = defense_flat[:, t, away_idx]

        home_adv = home_adv_flat[:, home_idx] if home_adv_flat is not None else 0.0

        lambda_home = np.clip(np.exp(home_attack - away_defense + home_adv), 0.01, 10)
        lambda_away = np.clip(np.exp(away_attack - home_defense), 0.01, 10)

        n_samples = min(1000, len(lambda_home))
        idx = np.random.choice(len(lambda_home), n_samples, replace=False)

        goals_home = np.random.poisson(lambda_home[idx])
        goals_away = np.random.poisson(lambda_away[idx])

        home_wins = float((goals_home > goals_away).mean())
        draws = float((goals_home == goals_away).mean())
        away_wins = float((goals_home < goals_away).mean())

        score_counts: dict = {}
        for h, a in zip(goals_home, goals_away):
            score = (int(h), int(a))
            score_counts[score] = score_counts.get(score, 0) + 1

        top_scores = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        score_probs = [
            {"score": f"{s[0]}-{s[1]}", "probability": round(count / len(goals_home), 4)}
            for s, count in top_scores
        ]

        predictions.append({
            "home_team": home_team,
            "away_team": away_team,
            "expected_goals_home": {
                "mean": round(float(lambda_home.mean()), 2),
                "median": round(float(np.median(lambda_home)), 2),
                "std": round(float(lambda_home.std()), 2),
                "percentile_5": round(float(np.percentile(lambda_home, 5)), 2),
                "percentile_95": round(float(np.percentile(lambda_home, 95)), 2),
            },
            "expected_goals_away": {
                "mean": round(float(lambda_away.mean()), 2),
                "median": round(float(np.median(lambda_away)), 2),
                "std": round(float(lambda_away.std()), 2),
                "percentile_5": round(float(np.percentile(lambda_away, 5)), 2),
                "percentile_95": round(float(np.percentile(lambda_away, 95)), 2),
            },
            "outcome_probabilities": {
                "home_win": round(home_wins, 4),
                "draw": round(draws, 4),
                "away_win": round(away_wins, 4),
            },
            "most_likely_scores": score_probs,
        })

    return predictions


def train(df, train_up_to_round: int, config: ModelConfig, samples: int, tune: int, chains: int, target_accept: float):
    input_model_data = prepare_model_data(df, max_round=train_up_to_round)
    model = build_model(input_model_data, config)

    with model:
        started_at = time.time()
        stop_event = threading.Event()

        def _heartbeat():
            while not stop_event.wait(30):
                print(f"  [sampling] {time.time() - started_at:.0f}s elapsed...")

        threading.Thread(target=_heartbeat, daemon=True).start()
        try:
            trace = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=42,
                idata_kwargs={"log_likelihood": False},
                return_inferencedata=True,
                discard_tuned_samples=True,
            )
        finally:
            stop_event.set()

    return trace, input_model_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["EPL"])
    parser.add_argument("--years", nargs="+", default=["2024"])
    parser.add_argument("--start-round", type=int, default=2, help="First round to generate predictions for (need at least 1 round of training data)")
    parser.add_argument("--end-round", type=int, default=None, help="Last round (inclusive). Defaults to all completed rounds.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--target-accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    parser.add_argument("--output-dir", type=str, default="data/gameweeks")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rounds that already have output files")
    parser.add_argument("--skip-rounds", nargs="+", type=int, default=[], metavar="ROUND", help="Round numbers to skip")
    parser.add_argument("--step", type=int, default=1, help="Process every Nth round (e.g. 2 = every other round)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ModelConfig(
        soft_center_team_strength=True,
        soft_center_sd=1.0,
        center_team_strength=False,
    )

    print(f"Fetching data for leagues={args.leagues} years={args.years}...")
    df = get_understat_data(leagues=args.leagues, years=args.years)
    df = add_rounds_to_data(df)
    df = add_match_ids(df)
    df = add_home_away_goals_xg(df)

    completed_rounds = get_completed_rounds(df)
    print(f"Found {len(completed_rounds)} completed rounds: {completed_rounds[0]}..{completed_rounds[-1]}")

    filtered_rounds = [
        r for r in completed_rounds
        if r >= args.start_round and (args.end_round is None or r <= args.end_round)
        and r not in args.skip_rounds
    ]
    rounds_to_process = filtered_rounds[::args.step]

    print(f"Processing {len(rounds_to_process)} rounds...\n")

    for round_num in rounds_to_process:
        round_dir = output_dir / f"round_{round_num}"
        predictions_path = round_dir / "predictions.json"
        actuals_path = round_dir / "actuals.json"
        strengths_path = round_dir / "team_strengths.json"

        if args.skip_existing and predictions_path.exists() and actuals_path.exists() and strengths_path.exists():
            print(f"Round {round_num}: skipping (already exists)")
            continue

        round_dir.mkdir(parents=True, exist_ok=True)

        print(f"Round {round_num}: training on rounds 1..{round_num - 1}...")
        t0 = time.time()

        try:
            trace, train_model_data = train(
                df,
                train_up_to_round=round_num - 1,
                config=config,
                samples=args.samples,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
            )

            pred_model_data = prepare_model_data(df, max_round=round_num)

            predictions = compute_predictions(trace, train_model_data, pred_model_data, round_num)
            actuals = extract_actuals(df, round_num)
            team_strengths = extract_team_strengths(trace, train_model_data)

            with open(predictions_path, "w") as f:
                json.dump({"round": round_num, "predictions": predictions}, f, indent=2)

            with open(actuals_path, "w") as f:
                json.dump({"round": round_num, "actuals": actuals}, f, indent=2)

            with open(strengths_path, "w") as f:
                json.dump({"round": round_num, "teams": team_strengths}, f, indent=2)

            elapsed = time.time() - t0
            print(f"Round {round_num}: done in {elapsed:.0f}s — {len(predictions)} predictions saved\n")

            del trace

        except Exception as e:
            import traceback
            print(f"Round {round_num}: FAILED — {e}")
            traceback.print_exc()
            print()
            continue

    print("Done.")


if __name__ == "__main__":
    main()
