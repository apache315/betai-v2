#!/usr/bin/env python3
"""
Generate predictions with CALIBRATED model for backtest
Applies Platt scaling to residuals
"""

import json
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import sys

def generate_calibrated_predictions(
    model_path='d:\\BetAI\\v2\\ml\\models\\residual_model.pkl',
    calibrator_path='d:\\BetAI\\v2\\ml\\models\\platt_calibrator.pkl',
    features_path='d:\\BetAI\\v2\\data\\processed\\features.json',
    output_path='d:\\BetAI\\v2\\backtest\\predictions_calibrated.json',
    min_edge=0.08
):
    """
    Generate predictions using calibrated model
    """
    print("Loading model and calibrator...")
    
    # Load model
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python ml/train_residual_optimized.py")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load calibrator (optional - if doesn't exist, use uncalibrated)
    calibrator = None
    if Path(calibrator_path).exists():
        with open(calibrator_path, 'rb') as f:
            calibrator = pickle.load(f)
        print("✅ Calibrator loaded")
    else:
        print("⚠️  Calibrator not found, using raw predictions")
    
    # Load features
    print("Loading features...")
    with open(features_path, 'r') as f:
        features = json.load(f)
    
    df = pd.DataFrame(features)
    
    # Flatten nested features
    if 'features' in df.columns and isinstance(df['features'].iloc[0], dict):
        features_expanded = pd.json_normalize(df['features'])
        df = pd.concat([df.drop('features', axis=1), features_expanded], axis=1)
    
    # Prepare target
    result_map = {'H': 1.0, 'D': 0.5, 'A': 0.0}
    df['target'] = df['result'].map(result_map)
    df['market_prob'] = 1 / df.get('marketOddsHome', 2.0).fillna(2.0)
    df = df.dropna(subset=['target', 'market_prob'])
    
    # Get features
    exclude = {'id', 'date', 'homeTeam', 'awayTeam', 'result', 'target', 
               'residual', 'features', 'matchId', 'league', 'season',
               'market_prob'}
    feature_cols = [c for c in df.columns if c not in exclude and 'odd' not in c.lower()]
    
    X = df[feature_cols].fillna(0)
    
    print(f"Generating predictions for {len(df)} matches...")
    
    # Get raw residuals
    residuals = model.predict(X)
    
    # Apply calibration if available
    if calibrator:
        calibrated_probs = calibrator.predict_proba(residuals.reshape(-1, 1))[:, 1]
    else:
        # Raw probability from residuals
        market_probs = df['market_prob'].values
        calibrated_probs = np.clip(market_probs + residuals, 0.01, 0.99)
    
    # Build predictions dict
    predictions = {}
    for idx, row in df.iterrows():
        match_id = row['matchId']
        
        # Home probability is calibrated_prob
        home_prob = calibrated_probs[idx]
        
        # Draw and Away: distribute remaining probability
        # Simple approach: market ratios
        market_draw = 1 / row.get('marketOddsDraw', 3.5)
        market_away = 1 / row.get('marketOddsAway', 3.0)
        market_sum = 1/row.get('marketOddsHome', 2.0) + market_draw + market_away
        
        draw_prob = (market_draw / market_sum) * (1 - home_prob)
        away_prob = (market_away / market_sum) * (1 - home_prob)
        
        predictions[match_id] = {
            'home': float(home_prob),
            'draw': float(draw_prob),
            'away': float(away_prob),
            'marketOdds': {
                'home': float(row.get('marketOddsHome', 2.0)),
                'draw': float(row.get('marketOddsDraw', 3.5)),
                'away': float(row.get('marketOddsAway', 3.0))
            }
        }
    
    # Save predictions
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n✅ Predictions saved: {output_file}")
    print(f"   Total predictions: {len(predictions)}")

if __name__ == '__main__':
    generate_calibrated_predictions()
