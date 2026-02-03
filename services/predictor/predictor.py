import sys
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import pymc as pm
import arviz as az
from datetime import datetime
import pickle
from pathlib import Path
import numpy as np
import time
from collections import defaultdict

from football_model.data.get_data import get_understat_data
from football_model.features.add_metadata import add_rounds_to_data, add_home_away_goals_xg, add_match_ids
from football_model.data.prepare_model_data import prepare_model_data
from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig

app = FastAPI(title="Football Predictor API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
TRACE_PATH = DATA_DIR / "trace.pkl"
MODEL_DATA_PATH = DATA_DIR / "model_data.pkl"
DATAFRAME_PATH = DATA_DIR / "dataframe.pkl"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 10
request_counts = defaultdict(list)


def check_rate_limit(client_id: str):
    """Check if client has exceeded rate limit"""
    now = time.time()
    # Remove old requests outside the window
    request_counts[client_id] = [req_time for req_time in request_counts[client_id] 
                                   if now - req_time < RATE_LIMIT_WINDOW]
    
    if len(request_counts[client_id]) >= MAX_REQUESTS_PER_WINDOW:
        return False
    
    request_counts[client_id].append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests"""
    client_id = request.client.host
    
    if not check_rate_limit(client_id):
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Rate limit exceeded. Maximum {MAX_REQUESTS_PER_WINDOW} requests per {RATE_LIMIT_WINDOW} seconds."
            }
        )
    
    response = await call_next(request)
    return response


class TrainRequest(BaseModel):
    leagues: List[str] = ["EPL"]
    years: List[str] = ["2025"]
    samples: int = 5000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.97
    clip_theta: float = 5.0
    center_team_strength: bool = False


class PredictionResponse(BaseModel):
    round: int
    predictions: List[Dict]  # Each dict contains team names and goal distributions


@app.get("/")
async def root():
    return {"message": "Football Predictor API", "version": "1.0.0"}


@app.post("/train", response_model=Dict)
async def train_model(request: TrainRequest):
    """
    Train the football prediction model and save the trace.
    """
    try:
        # Get data
        df = get_understat_data(
            leagues=request.leagues,
            years=request.years
        )
        df = add_rounds_to_data(df)
        df = add_match_ids(df)
        df = add_home_away_goals_xg(df)
        
        past_matches = df[df['datetime'] <= datetime.today()]
        future_matches = df[df['datetime'] > datetime.today()]

        cur_round = past_matches['round'].max()

        if cur_round in future_matches['round'].values:
            round_over = False
        else:
            round_over = True
        
        # If we're in the middle of a round, that is round we want to predict next
        if round_over:
            next_round = cur_round + 1
        # If the current round is ongoing, we want to predict this round
        else:
            next_round = cur_round
            
        
        # Prepare model data
        input_model_data = prepare_model_data(df, max_round=next_round - 1)
        
        # Create model configuration
        config = ModelConfig(
            clip_theta=request.clip_theta,
            center_team_strength=request.center_team_strength,
        )
        
        # Build the model
        model = build_model(input_model_data, config)
        
        # Sample
        with model:
            trace = pm.sample(
                draws=request.samples,
                tune=request.tune,
                chains=request.chains,
                target_accept=request.target_accept,
                random_seed=42,
                idata_kwargs={"log_likelihood": True},
                return_inferencedata=True,
                discard_tuned_samples=True
            )
        
        # Save trace, model data, and dataframe
        with open(TRACE_PATH, 'wb') as f:
            pickle.dump(trace, f)
        
        with open(MODEL_DATA_PATH, 'wb') as f:
            pickle.dump(input_model_data, f)
            
        with open(DATAFRAME_PATH, 'wb') as f:
            pickle.dump(df, f)
        
        return {
            "status": "success",
            "message": f"Model trained up to round {next_round - 1} and trace saved",
            "next_gameweek": int(next_round),
            "n_teams": input_model_data.n_teams,
            "n_matches": len(input_model_data.goals_home)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/predict", response_model=PredictionResponse)
async def get_predictions():
    """
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")g /train endpoint."
            )
        
        with open(TRACE_PATH, 'rb') as f:
            trace = pickle.load(f)
        
        with open(MODEL_DATA_PATH, 'rb') as f:
            train_model_data = pickle.load(f)
        
        # Load the saved dataframe from training instead of fetching fresh
        with open(DATAFRAME_PATH, 'rb') as f:
            df = pickle.load(f)
        

        past_matches = df[df['datetime'] <= datetime.today()]
        future_matches = df[df['datetime'] > datetime.today()]

        cur_round = past_matches['round'].max()

        if cur_round in future_matches['round'].values:
            round_over = False
        else:
            round_over = True
        
        # If we're in the middle of a round, that is round we want to predict next
        if round_over:
            next_round = cur_round + 1
        # If the current round is ongoing, we want to predict this round
        else:
            next_round = cur_round
        
        # Prepare model data including the prediction round
        pred_model_data = prepare_model_data(df, max_round=next_round)
        
        # Filter to just matches in the prediction round
        pred_mask = pred_model_data.t_idx == next_round
        
        if not pred_mask.any():
            raise HTTPException(status_code=404, detail=f"No matches found for round {next_round}")
        
        # Extract posterior samples (not just means)
        attack_samples = trace.posterior['attack'].values  # shape: (chains, draws, time, teams)
        defense_samples = trace.posterior['defence'].values
        
        if 'home_adv' in trace.posterior:
            home_adv_samples = trace.posterior['home_adv'].values  # shape: (chains, draws, teams)
        else:
            home_adv_samples = None
        
        # Flatten chain and draw dimensions
        n_chains, n_draws = attack_samples.shape[0], attack_samples.shape[1]
        attack_flat = attack_samples.reshape(n_chains * n_draws, *attack_samples.shape[2:])  # (samples, time, teams)
        defense_flat = defense_samples.reshape(n_chains * n_draws, *defense_samples.shape[2:])
        
        if home_adv_samples is not None:
            home_adv_flat = home_adv_samples.reshape(n_chains * n_draws, home_adv_samples.shape[2])
        
        # Reverse team mapping
        idx_to_team = {v: k for k, v in pred_model_data.team_mapping.items()}
        
        predictions = []
        seen_matches = set()  # Track matches we've already processed
        
        # Iterate through prediction round matches
        for i in np.where(pred_mask)[0]:
            team_idx = pred_model_data.team_idx[i]
            opp_idx = pred_model_data.opp_idx[i]
            is_home = pred_model_data.home[i]
            
            # Determine home/away teams
            if is_home == 1:
                home_idx, away_idx = team_idx, opp_idx
            else:
                home_idx, away_idx = opp_idx, team_idx
            
            # Create a unique match identifier (sorted tuple to avoid duplicates)
            match_key = tuple(sorted([home_idx, away_idx]))
            if match_key in seen_matches:
                continue
            seen_matches.add(match_key)
            
            home_team = idx_to_team[home_idx]
            away_team = idx_to_team[away_idx]
            
            # Use last time index from training (next_round - 1)
            t = min(next_round - 1, attack_flat.shape[1] - 1)
            
            # Get samples for this match
            home_attack_samples = attack_flat[:, t, home_idx]
            away_attack_samples = attack_flat[:, t, away_idx]
            home_defense_samples = defense_flat[:, t, home_idx]
            away_defense_samples = defense_flat[:, t, away_idx]
            
            # Home advantage samples
            if home_adv_samples is not None:
                home_advantage_samples = home_adv_flat[:, home_idx]
            else:
                home_advantage_samples = 0.0
            
            # Calculate lambda (expected goals) for each sample
            lambda_home_samples = np.exp(home_attack_samples - away_defense_samples + home_advantage_samples)
            lambda_away_samples = np.exp(away_attack_samples - home_defense_samples)
            
            # Sample goals from Poisson distributions
            goals_home_samples = np.random.poisson(lambda_home_samples)
            goals_away_samples = np.random.poisson(lambda_away_samples)
            
            # Calculate outcome probabilities
            home_wins = (goals_home_samples > goals_away_samples).mean()
            draws = (goals_home_samples == goals_away_samples).mean()
            away_wins = (goals_home_samples < goals_away_samples).mean()
            
            # Calculate score distribution (most likely scores)
            score_counts = {}
            for h, a in zip(goals_home_samples, goals_away_samples):
                score = (int(h), int(a))
                score_counts[score] = score_counts.get(score, 0) + 1
            
            # Get top 10 most likely scores
            top_scores = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            score_probs = [{"score": f"{s[0]}-{s[1]}", "probability": round(count / len(goals_home_samples), 4)} 
                          for s, count in top_scores]
            
            predictions.append({
                "home_team": home_team,
                "away_team": away_team,
                "expected_goals_home": {
                    "mean": round(float(lambda_home_samples.mean()), 2),
                    "median": round(float(np.median(lambda_home_samples)), 2),
                    "std": round(float(lambda_home_samples.std()), 2),
                    "percentile_5": round(float(np.percentile(lambda_home_samples, 5)), 2),
                    "percentile_95": round(float(np.percentile(lambda_home_samples, 95)), 2),
                },
                "expected_goals_away": {
                    "mean": round(float(lambda_away_samples.mean()), 2),
                    "median": round(float(np.median(lambda_away_samples)), 2),
                    "std": round(float(lambda_away_samples.std()), 2),
                    "percentile_5": round(float(np.percentile(lambda_away_samples, 5)), 2),
                    "percentile_95": round(float(np.percentile(lambda_away_samples, 95)), 2),
                },
                "outcome_probabilities": {
                    "home_win": round(float(home_wins), 4),
                    "draw": round(float(draws), 4),
                    "away_win": round(float(away_wins), 4),
                },
                "most_likely_scores": score_probs
            })
        
        return PredictionResponse(
            round=int(next_round),
            predictions=predictions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/status")
async def get_status():
    """
    Check if a trained model is available.
    """
    return {
        "model_trained": TRACE_PATH.exists(),
        "trace_path": str(TRACE_PATH),
        "last_modified": datetime.fromtimestamp(TRACE_PATH.stat().st_mtime).isoformat() if TRACE_PATH.exists() else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8956)