#!/usr/bin/env python3
"""
Train residual XGBoost model with optimized memory handling.
Processes features in chunks to avoid memory issues.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import pickle
import sys

def load_features_chunked(filepath, chunk_size=10000):
    """Load large JSON features file in chunks to avoid memory issues."""
    with open(filepath, 'r') as f:
        features = json.load(f)
    
    # Convert to DataFrame chunks
    print(f"Loaded {len(features)} matches")
    for i in range(0, len(features), chunk_size):
        chunk = features[i:i+chunk_size]
        df_chunk = pd.DataFrame(chunk)
        yield df_chunk

def prepare_residual_data(df):
    """Prepare data for residual model training."""
    # Features sono nested - estrai dalla colonna 'features'
    if 'features' in df.columns and isinstance(df['features'].iloc[0], dict):
        features_expanded = pd.json_normalize(df['features'])
        df = pd.concat([df.drop('features', axis=1), features_expanded], axis=1)
    
    # Target: actual result (1 = H win, 0.5 = Draw, 0 = A win)
    result_map = {'H': 1.0, 'D': 0.5, 'A': 0.0}
    df['target'] = df['result'].map(result_map)
    
    # Market prediction: implied probability from odds (usa marketOdds se closing non disponibile)
    if 'closingOddsHome' not in df.columns:
        df['closingOddsHome'] = df.get('marketOddsHome', 2.0)
    
    df['market_prob'] = 1 / df['closingOddsHome'].fillna(2.0)
    
    # Residual: what the model should learn
    df['residual'] = df['target'] - df['market_prob']
    
    # Drop rows with NaN
    df = df.dropna(subset=['target', 'market_prob', 'residual'])
    
    return df

def get_feature_columns(df):
    """Identify feature columns (exclude metadata)."""
    exclude = {'id', 'date', 'homeTeam', 'awayTeam', 'result', 'target', 
               'market_prob', 'residual', 'features', 'closingOdds', 'matchId',
               'league', 'season', 'totalGoals', 'btts',
               'marketOdds', 'closingOddsHome', 'closingOddsDraw', 'closingOddsAway',
               'marketOddsHome', 'marketOddsDraw', 'marketOddsAway'}
    
    feature_cols = [col for col in df.columns if col not in exclude and not col.startswith('_')]
    return feature_cols

def train_residual_model(features_path, output_path):
    """Train residual XGBoost model."""
    print("Loading features...")
    
    # Load all features (we'll optimize if memory is an issue)
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
        df = pd.DataFrame(features)
    except Exception as e:
        print(f"Error loading features: {e}")
        return
    
    print(f"Loaded {len(df)} matches")
    print(f"Columns: {df.columns.tolist()[:10]}...")
    
    # Prepare data
    print("Preparing data...")
    df = prepare_residual_data(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features")
    
    if len(feature_cols) == 0:
        print("ERROR: No feature columns found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Handle missing values
    X = df[feature_cols].fillna(0)
    y = df['residual']
    
    print(f"X shape: {X.shape}")
    print(f"y stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Time series split (no look-ahead bias)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold+1}/5...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        scores.append(test_score)
        
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²:  {test_score:.4f}")
    
    # Final model on all data
    print(f"\nMean CV R²: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    print("\nTraining final model on all data...")
    
    final_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )
    
    final_model.fit(X, y, verbose=False)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(final_model, f)
    
    # Save metadata
    metadata = {
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'cv_r2_mean': float(np.mean(scores)),
        'cv_r2_std': float(np.std(scores)),
    }
    
    with open(output_file.parent / 'residual_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to: {output_file}")
    print(f"✅ Metadata saved to: {output_file.parent / 'residual_metadata.json'}")

if __name__ == '__main__':
    features_path = 'd:\\BetAI\\v2\\data\\processed\\features.json'
    output_path = 'd:\\BetAI\\v2\\ml\\models\\residual_model.pkl'
    
    train_residual_model(features_path, output_path)
