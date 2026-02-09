#!/usr/bin/env python3
"""
Calibrate model with Platt Scaling
Fixes systematic probability miscalibration without retraining
"""

import json
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from pathlib import Path

def calibrate_with_platt_scaling(model_path, features_path, output_path):
    """
    Apply Platt scaling to calibrate probability predictions
    """
    print("Loading model and features...")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load features
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
    df['residual'] = df['target'] - df['market_prob']
    df = df.dropna(subset=['target', 'market_prob', 'residual'])
    
    # Get features
    exclude = {'id', 'date', 'homeTeam', 'awayTeam', 'result', 'target', 
               'residual', 'features', 'matchId', 'league', 'season',
               'market_prob', 'marketOdds'}
    feature_cols = [c for c in df.columns if c not in exclude and 'odd' not in c.lower()]
    
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    print(f"Training Platt scaling on {len(X)} matches...")
    
    # Get raw predictions (residuals from model)
    residuals = model.predict(X)
    calibrated_probs = 1 / df['marketOddsHome'].fillna(2.0).values + residuals
    calibrated_probs = np.clip(calibrated_probs, 0.01, 0.99)
    
    # Fit logistic regression as calibrator
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(residuals.reshape(-1, 1), y)
    
    # Test calibration
    calibrated_predictions = calibrator.predict_proba(residuals.reshape(-1, 1))[:, 1]
    
    brier_before = np.mean((calibrated_probs - y) ** 2)
    brier_after = np.mean((calibrated_predictions - y) ** 2)
    
    print(f"\nCalibration Results:")
    print(f"  Brier before: {brier_before:.6f}")
    print(f"  Brier after:  {brier_after:.6f}")
    print(f"  Improvement:  {(brier_before - brier_after):.6f} ({100*(brier_before-brier_after)/brier_before:.1f}%)")
    
    # Save calibrator
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(calibrator, f)
    
    print(f"\n✅ Calibrator saved: {output_file}")
    
    return calibrator

if __name__ == '__main__':
    calibrate_with_platt_scaling(
        'd:\\BetAI\\v2\\ml\\models\\residual_model.pkl',
        'd:\\BetAI\\v2\\data\\processed\\features.json',
        'd:\\BetAI\\v2\\ml\\models\\platt_calibrator.pkl'
    )
