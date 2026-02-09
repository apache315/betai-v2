#!/usr/bin/env python3
"""
Train model WITHOUT market odds as features.
This tests if edge comes from real insight or just market data leak.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import pickle

def prepare_features_no_market_data(df):
    """
    Prepare features EXCLUDING market odds.
    This prevents data leak and tests true generalization.
    """
    # Features to exclude: any market/odds related
    market_keywords = ['odd', 'market', 'betting', 'implied', 'clv', 'quote', 'closing']
    
    exclude = {'id', 'date', 'homeTeam', 'awayTeam', 'result', 'target', 
               'residual', 'features', 'matchId', 'league', 'season', 
               'totalGoals', 'btts'}
    
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if any(kw.lower() in col.lower() for kw in market_keywords):
            continue
        feature_cols.append(col)
    
    return feature_cols

def train_no_market_data(features_path, output_path):
    """
    Train residual model WITHOUT market odds.
    """
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

    # Extract closing odds (nested dict: {home, draw, away})
    if 'closingOdds' in df.columns:
        closing = df['closingOdds'].apply(lambda x: x.get('home') if isinstance(x, dict) else None)
        df['market_prob'] = 1 / closing
    elif 'odds_home_raw' in df.columns:
        df['market_prob'] = 1 / df['odds_home_raw']
    else:
        df['market_prob'] = 1 / 2.0  # fallback

    df['market_prob'] = df['market_prob'].fillna(1 / 2.0)
    df['residual'] = df['target'] - df['market_prob']
    df = df.dropna(subset=['target', 'market_prob', 'residual'])
    
    # Get features WITHOUT market data
    feature_cols = prepare_features_no_market_data(df)
    
    print(f"\nModel WITHOUT market odds:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Sample features: {feature_cols[:10]}")
    
    X = df[feature_cols].fillna(0)
    y = df['residual']
    
    # Time series CV
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        
        model.fit(X_train, y_train, verbose=False)
        score = model.score(X_test, y_test)
        scores.append(score)
        print(f"  Fold {fold+1}/5: R² = {score:.4f}")
    
    print(f"\nMean CV R²: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    
    if np.mean(scores) < 0.01:
        print("[WARNING] Model without market data has very low R2")
        print("   This suggests edge comes from market data, not real insight")
    
    # Train final model
    final_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(final_model, f)
    
    metadata = {
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'cv_r2_mean': float(np.mean(scores)),
        'cv_r2_std': float(np.std(scores)),
        'has_market_data': False,
        'model_type': 'residual_no_market',
    }
    
    with open(output_file.parent / 'metadata_no_market.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved: {output_file}")

if __name__ == '__main__':
    train_no_market_data(
        'd:\\BetAI\\v2\\data\\processed\\features.json',
        'd:\\BetAI\\v2\\ml\\models\\residual_no_market.pkl'
    )
