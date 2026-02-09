#!/usr/bin/env python3
"""
XGBoost Training Script for BetAI v2

Usage:
    python train.py --data features.json --output model.json

Trains a multi-class classifier (H/D/A) with:
- Optuna hyperparameter optimization
- Platt scaling calibration
- Walk-forward validation
- CLV tracking (Closing Line Value)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
import optuna
from optuna.samplers import TPESampler

# Suppress Optuna logs during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_features(path: str) -> pd.DataFrame:
    """Load features JSON and convert to DataFrame. Uses streaming for large files."""
    import os
    file_size = os.path.getsize(path)

    if file_size > 100_000_000:  # > 100MB
        print(f"  Large file ({file_size / 1e6:.0f} MB), streaming...")
        return _load_features_streaming(path, include_extra=True)

    with open(path, 'r') as f:
        data = json.load(f)

    rows = []
    for match in data:
        row = {
            'matchId': match['matchId'],
            'date': match['date'],
            'result': match['result'],
            'totalGoals': match['totalGoals'],
            'btts': match['btts'],
            **match['features']
        }
        # Add closing odds if available
        if match.get('closingOdds'):
            row['closing_odds_home'] = match['closingOdds']['home']
            row['closing_odds_draw'] = match['closingOdds']['draw']
            row['closing_odds_away'] = match['closingOdds']['away']
        rows.append(row)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _load_features_streaming(path: str, include_extra: bool = False) -> pd.DataFrame:
    """Stream-parse large JSON arrays line by line."""
    rows = []
    count = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if not line or line in ('[]', '[', ']'):
                continue
            try:
                match = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = {
                'matchId': match['matchId'],
                'date': match['date'],
                'result': match['result'],
                **match['features']
            }
            if include_extra:
                row['totalGoals'] = match.get('totalGoals', 0)
                row['btts'] = match.get('btts', False)
            if match.get('closingOdds'):
                row['closing_odds_home'] = match['closingOdds']['home']
                row['closing_odds_draw'] = match['closingOdds']['draw']
                row['closing_odds_away'] = match['closingOdds']['away']
            rows.append(row)
            count += 1
            if count % 50000 == 0:
                print(f"    Loaded {count} matches...")

    print(f"    Total: {count} matches")
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def prepare_data(df: pd.DataFrame, feature_cols: list):
    """Prepare X, y for training."""
    # Encode result: H=0, D=1, A=2
    label_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['result'].map(label_map).values
    X = df[feature_cols].values
    # Clean NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def print_data_summary(df: pd.DataFrame, feature_cols: list):
    """Print data understanding summary before training."""
    print("\n=== Data Summary ===")
    print(f"  Total matches: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Features: {len(feature_cols)}")

    # Class distribution
    counts = df['result'].value_counts()
    total = len(df)
    print(f"\n  Class distribution:")
    for label in ['H', 'D', 'A']:
        c = counts.get(label, 0)
        print(f"    {label}: {c} ({c/total:.1%})")

    # NaN/inf check
    X = df[feature_cols].values
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"\n  Data quality:")
        print(f"    NaN values: {nan_count} ({nan_count / X.size:.2%} of cells)")
        print(f"    Inf values: {inf_count}")
        # Per-feature NaN rates
        nan_per_col = np.isnan(X).sum(axis=0)
        bad_cols = [(feature_cols[i], nan_per_col[i]) for i in range(len(feature_cols)) if nan_per_col[i] > 0]
        if bad_cols:
            print(f"    Features with NaN (top 10):")
            for name, cnt in sorted(bad_cols, key=lambda x: -x[1])[:10]:
                print(f"      {name}: {cnt} ({cnt/len(df):.1%})")
    else:
        print(f"\n  Data quality: clean (no NaN/inf)")

    # Feature variance check
    variances = np.nanvar(X, axis=0)
    zero_var = [feature_cols[i] for i in range(len(feature_cols)) if variances[i] == 0]
    if zero_var:
        print(f"\n  Zero-variance features ({len(zero_var)}): {zero_var[:5]}...")
    print()


class EnsembleCalibrator:
    """
    Ensemble calibration: average of Platt (sigmoid) and Isotonic.

    Platt is more stable (parametric), Isotonic is more flexible (non-parametric).
    Averaging reduces overfitting risk from Isotonic while keeping flexibility.
    """

    def __init__(self):
        self.platt_model = None
        self.isotonic_model = None

    def fit(self, base_model, X_val, y_val):
        """Fit both calibrators on validation data."""
        self.platt_model = CalibratedClassifierCV(
            base_model, method='sigmoid', cv='prefit'
        )
        self.platt_model.fit(X_val, y_val)

        self.isotonic_model = CalibratedClassifierCV(
            base_model, method='isotonic', cv='prefit'
        )
        self.isotonic_model.fit(X_val, y_val)

    def predict_proba(self, X):
        """Average predictions from both calibrators."""
        p_platt = self.platt_model.predict_proba(X)
        p_isotonic = self.isotonic_model.predict_proba(X)
        # Weighted average: 40% Platt (stable) + 60% Isotonic (flexible)
        p_avg = 0.4 * p_platt + 0.6 * p_isotonic
        # Renormalize to ensure sum = 1
        row_sums = p_avg.sum(axis=1, keepdims=True)
        p_avg = p_avg / row_sums
        return p_avg


def compute_brier_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute multi-class Brier score."""
    n_classes = y_prob.shape[1]
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1
    return np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1))


def compute_clv(df: pd.DataFrame, y_prob: np.ndarray) -> dict:
    """
    Compute Closing Line Value.
    CLV = (played_odds / closing_odds - 1) * 100

    Positive CLV = beating the market.
    """
    if 'closing_odds_home' not in df.columns:
        return {'clv_mean': None, 'clv_positive_rate': None}

    clv_values = []
    for i, row in df.iterrows():
        if pd.isna(row.get('closing_odds_home')):
            continue

        # Find best value bet (highest edge)
        implied_probs = [
            1 / row['closing_odds_home'],
            1 / row['closing_odds_draw'],
            1 / row['closing_odds_away']
        ]
        closing_odds = [row['closing_odds_home'], row['closing_odds_draw'], row['closing_odds_away']]

        model_probs = y_prob[i]

        # For each outcome, compute edge
        for j in range(3):
            if implied_probs[j] > 0:
                edge = model_probs[j] - implied_probs[j]
                if edge > 0.05:  # Only consider value bets (>5% edge)
                    # Assume we bet at a typical early odds (5% worse than closing)
                    played_odds = closing_odds[j] * 1.05
                    clv = (played_odds / closing_odds[j] - 1) * 100
                    clv_values.append(clv)

    if not clv_values:
        return {'clv_mean': 0.0, 'clv_positive_rate': 0.0}

    return {
        'clv_mean': float(np.mean(clv_values)),
        'clv_positive_rate': float(np.mean([1 if v > 0 else 0 for v in clv_values]))
    }


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for hyperparameter optimization."""
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'seed': 42,
        # Hyperparameters to optimize
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = xgb.XGBClassifier(**params, verbosity=0, early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_prob = model.predict_proba(X_val)
    brier = compute_brier_multiclass(y_val, y_prob)

    return brier  # Minimize Brier score


def train_with_optuna(X_train, y_train, X_val, y_val, n_trials=50):
    """Hyperparameter optimization with Optuna."""
    print(f"  Optuna: optimizing with {n_trials} trials...")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"  Best Brier: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


def walk_forward_validation(df: pd.DataFrame, feature_cols: list, n_splits=5):
    """
    Walk-forward (time series) cross-validation.

    Each fold trains on all data before a cutoff and tests on data after.
    """
    print(f"\n=== Walk-Forward Validation ({n_splits} folds) ===")

    X, y = prepare_data(df, feature_cols)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Split train into train/val for Optuna
        val_size = int(len(X_train) * 0.15)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        # Optimize hyperparameters
        best_params = train_with_optuna(X_tr, y_tr, X_val, y_val, n_trials=30)

        # Train final model with best params + early stopping
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'tree_method': 'hist',
            'seed': 42,
            'verbosity': 0,
            'early_stopping_rounds': 30,
            **best_params
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Ensemble calibration (Platt + Isotonic average)
        calibrator = EnsembleCalibrator()
        calibrator.fit(model, X_val, y_val)

        # Evaluate
        y_prob = calibrator.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)

        brier = compute_brier_multiclass(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_prob)

        # CLV on test set
        test_df = df.iloc[test_idx].reset_index(drop=True)
        clv = compute_clv(test_df, y_prob)

        test_start = df.iloc[test_idx[0]]['date'].strftime('%Y-%m')
        test_end = df.iloc[test_idx[-1]]['date'].strftime('%Y-%m')

        print(f"\nFold {fold + 1}: {test_start} to {test_end}")
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"  Brier: {brier:.4f}, Accuracy: {accuracy:.2%}, LogLoss: {logloss:.4f}")
        print(f"  CLV: mean={clv['clv_mean']:.2f}%, positive_rate={clv['clv_positive_rate']:.2%}")

        results.append({
            'fold': fold + 1,
            'test_period': f"{test_start} to {test_end}",
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'brier': brier,
            'accuracy': accuracy,
            'logloss': logloss,
            'clv_mean': clv['clv_mean'],
            'clv_positive_rate': clv['clv_positive_rate'],
            'best_params': best_params,
        })

    # Aggregate results
    avg_brier = np.mean([r['brier'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_clv = np.mean([r['clv_mean'] for r in results if r['clv_mean'] is not None])

    print(f"\n=== Aggregate Results ===")
    print(f"  Mean Brier: {avg_brier:.4f}")
    print(f"  Mean Accuracy: {avg_accuracy:.2%}")
    print(f"  Mean CLV: {avg_clv:.2f}%")

    return results


def train_final_model(df: pd.DataFrame, feature_cols: list, output_path: str):
    """Train final model on all data and save."""
    print("\n=== Training Final Model ===")

    X, y = prepare_data(df, feature_cols)

    # Use last 15% for validation/calibration
    val_size = int(len(X) * 0.15)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    # Optimize
    best_params = train_with_optuna(X_train, y_train, X_val, y_val, n_trials=50)

    # Train on full training set with early stopping
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'tree_method': 'hist',
        'seed': 42,
        'verbosity': 0,
        'early_stopping_rounds': 30,
        **best_params
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Ensemble calibration
    calibrator = EnsembleCalibrator()
    calibrator.fit(model, X_val, y_val)

    # Evaluate on validation
    y_prob = calibrator.predict_proba(X_val)
    brier = compute_brier_multiclass(y_val, y_prob)
    print(f"  Final validation Brier: {brier:.4f}")

    # Save model
    model.save_model(output_path)

    # Save metadata
    meta_path = output_path.replace('.json', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'feature_cols': feature_cols,
            'best_params': best_params,
            'validation_brier': brier,
            'train_size': len(X_train),
            'trained_at': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"  Model saved to: {output_path}")
    print(f"  Metadata saved to: {meta_path}")

    return model, best_params


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model for BetAI')
    parser.add_argument('--data', required=True, help='Path to features JSON')
    parser.add_argument('--output', default='model.json', help='Output model path')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation, skip final training')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--selected-features', default=None, help='Path to selected_features.json from feature selection')
    args = parser.parse_args()

    print("=== BetAI v2 - XGBoost Training ===\n")

    # Load data
    print(f"Loading features from: {args.data}")
    df = load_features(args.data)
    print(f"  Loaded {len(df)} matches")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Get feature columns
    exclude_cols = {'matchId', 'date', 'result', 'totalGoals', 'btts',
                    'closing_odds_home', 'closing_odds_draw', 'closing_odds_away'}

    if args.selected_features:
        # Use pre-selected features from feature_selection.py
        with open(args.selected_features, 'r') as f:
            sel_data = json.load(f)
        feature_cols = sel_data['selected_features']
        # Ensure all columns exist in df
        feature_cols = [c for c in feature_cols if c in df.columns]
        print(f"  Features: {len(feature_cols)} (from selection, was {sel_data['n_original']})")
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        print(f"  Features: {len(feature_cols)}")

    # Data understanding
    print_data_summary(df, feature_cols)

    # Walk-forward validation
    results = walk_forward_validation(df, feature_cols, n_splits=args.n_splits)

    if args.validate_only:
        print("\n[validate-only mode, skipping final training]")
        return

    # Train final model
    train_final_model(df, feature_cols, args.output)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
