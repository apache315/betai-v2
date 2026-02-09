#!/usr/bin/env python3
"""
Bayesian Ensemble for BetAI v2

Combines three probability sources:
1. Market Prior (Pinnacle odds → implied probabilities)
2. XGBoost (tabular features → calibrated probabilities)
3. GNN (graph embeddings → probabilities)

Ensemble formula:
    P_final = α × P_market + β × P_xgboost + γ × P_gnn

Where α + β + γ = 1, optimized on validation data to minimize Brier score.

The key insight: the market is already very efficient (~95% accuracy on
implied probabilities). Our edge comes from the 5% where our models
disagree with the market. The Bayesian framework properly weights
each source based on historical reliability.

Usage:
    python bayesian_ensemble.py --features features.json --gnn-embeddings gnn_embeddings.json --selected-features selected_features.json --output ensemble_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from scipy.optimize import minimize


def load_features(path: str) -> pd.DataFrame:
    import os
    file_size = os.path.getsize(path)

    if file_size > 100_000_000:  # > 100MB: stream line by line
        print(f"  Large file ({file_size / 1e6:.0f} MB), streaming...")
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

    with open(path, 'r') as f:
        data = json.load(f)

    rows = []
    for match in data:
        row = {
            'matchId': match['matchId'],
            'date': match['date'],
            'result': match['result'],
            **match['features']
        }
        if match.get('closingOdds'):
            row['closing_odds_home'] = match['closingOdds']['home']
            row['closing_odds_draw'] = match['closingOdds']['draw']
            row['closing_odds_away'] = match['closingOdds']['away']
        rows.append(row)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def compute_brier(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_oh = np.zeros((len(y_true), n_classes))
    for i, l in enumerate(y_true):
        y_oh[i, l] = 1
    return np.mean(np.sum((y_prob - y_oh) ** 2, axis=1))


# ──────────────────────────────────────────────
# SOURCE 1: MARKET PRIOR
# ──────────────────────────────────────────────

def get_market_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Extract market implied probabilities from closing odds.
    Reads from closingOdds (separate from model features) and removes overround.
    """
    probs = np.zeros((len(df), 3))

    for i, (_, row) in enumerate(df.iterrows()):
        # Read from closing_odds fields (loaded separately, not model features)
        oh = row.get('closing_odds_home', 0)
        od = row.get('closing_odds_draw', 0)
        oa = row.get('closing_odds_away', 0)

        if oh > 1 and od > 1 and oa > 1:
            # Convert decimal odds to implied probabilities, remove overround
            raw_h = 1.0 / oh
            raw_d = 1.0 / od
            raw_a = 1.0 / oa
            total = raw_h + raw_d + raw_a
            probs[i] = [raw_h / total, raw_d / total, raw_a / total]
        else:
            probs[i] = [0.4, 0.27, 0.33]  # Fallback when no odds

    return probs


# ──────────────────────────────────────────────
# SOURCE 2: XGBOOST
# ──────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with calibration and early stopping."""
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=300, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=7, reg_alpha=0.05, reg_lambda=0.01,
        tree_method='hist', seed=42, verbosity=0,
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Calibrate with both methods and average
    platt = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    platt.fit(X_val, y_val)

    isotonic = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    isotonic.fit(X_val, y_val)

    return model, platt, isotonic


def predict_xgboost(platt, isotonic, X):
    """Ensemble calibrated prediction."""
    p1 = platt.predict_proba(X)
    p2 = isotonic.predict_proba(X)
    p = 0.4 * p1 + 0.6 * p2
    return p / p.sum(axis=1, keepdims=True)


# ──────────────────────────────────────────────
# SOURCE 3: GNN PREDICTIONS
# ──────────────────────────────────────────────

def get_gnn_probs(df: pd.DataFrame, gnn_embeddings: dict) -> np.ndarray:
    """Extract GNN predictions for each match."""
    probs = np.zeros((len(df), 3))

    for i, (_, row) in enumerate(df.iterrows()):
        match_id = row['matchId']
        if match_id in gnn_embeddings:
            emb = gnn_embeddings[match_id]
            probs[i] = [emb['gnn_prob_home'], emb['gnn_prob_draw'], emb['gnn_prob_away']]
        else:
            # Fallback: use closing odds as market prior
            oh = row.get('closing_odds_home', 0)
            od = row.get('closing_odds_draw', 0)
            oa = row.get('closing_odds_away', 0)
            if oh > 1 and od > 1 and oa > 1:
                raw_h, raw_d, raw_a = 1/oh, 1/od, 1/oa
                total = raw_h + raw_d + raw_a
                probs[i] = [raw_h/total, raw_d/total, raw_a/total]
            else:
                probs[i] = [0.4, 0.27, 0.33]

    return probs


# ──────────────────────────────────────────────
# BAYESIAN ENSEMBLE OPTIMIZATION
# ──────────────────────────────────────────────

def optimize_weights(p_market, p_xgb, p_gnn, y_true):
    """
    Find optimal weights α, β, γ that minimize Brier score.
    Constraint: α + β + γ = 1, all >= 0.05 (minimum 5% for each source)
    """
    def objective(weights):
        w = weights / weights.sum()  # Normalize
        p_ensemble = w[0] * p_market + w[1] * p_xgb + w[2] * p_gnn
        # Renormalize
        p_ensemble = p_ensemble / p_ensemble.sum(axis=1, keepdims=True)
        return compute_brier(y_true, p_ensemble)

    # Start with equal weights
    x0 = np.array([0.33, 0.33, 0.34])

    # Bounds: each weight between 5% and 80%
    bounds = [(0.05, 0.80), (0.05, 0.80), (0.05, 0.80)]

    # Constraint: sum = 1
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x / result.x.sum()
    return optimal_weights, result.fun


def ensemble_predict(p_market, p_xgb, p_gnn, weights):
    """Apply ensemble weights."""
    p = weights[0] * p_market + weights[1] * p_xgb + weights[2] * p_gnn
    return p / p.sum(axis=1, keepdims=True)


# ──────────────────────────────────────────────
# WALK-FORWARD EVALUATION
# ──────────────────────────────────────────────

def walk_forward_ensemble(df, feature_cols, gnn_embeddings, n_splits=3):
    """
    Walk-forward validation of the full ensemble.
    """
    print(f"\n=== Bayesian Ensemble Walk-Forward ({n_splits} folds) ===")

    label_map = {'H': 0, 'D': 1, 'A': 2}
    y_all = df['result'].map(label_map).values
    X_all = df[feature_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        test_start = test_df['date'].min().strftime('%Y-%m')
        test_end = test_df['date'].max().strftime('%Y-%m')
        print(f"\nFold {fold + 1}: {test_start} to {test_end} (train={len(train_idx)}, test={len(test_idx)})")

        # Split train into train/val
        val_size = int(len(X_train) * 0.15)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        val_df = train_df.iloc[-val_size:]

        # SOURCE 1: Market
        p_market_val = get_market_probs(val_df)
        p_market_test = get_market_probs(test_df)

        # SOURCE 2: XGBoost
        _, platt, isotonic = train_xgboost(X_tr, y_tr, X_val, y_val)
        p_xgb_val = predict_xgboost(platt, isotonic, X_val)
        p_xgb_test = predict_xgboost(platt, isotonic, X_test)

        # SOURCE 3: GNN
        p_gnn_val = get_gnn_probs(val_df, gnn_embeddings)
        p_gnn_test = get_gnn_probs(test_df, gnn_embeddings)

        # Optimize weights on validation set
        weights, val_brier = optimize_weights(p_market_val, p_xgb_val, p_gnn_val, y_val)
        print(f"  Optimal weights: market={weights[0]:.2%}, xgb={weights[1]:.2%}, gnn={weights[2]:.2%}")
        print(f"  Validation Brier: {val_brier:.4f}")

        # Predict test set
        p_ensemble = ensemble_predict(p_market_test, p_xgb_test, p_gnn_test, weights)

        # Evaluate all sources individually
        brier_market = compute_brier(y_test, p_market_test)
        brier_xgb = compute_brier(y_test, p_xgb_test)
        brier_gnn = compute_brier(y_test, p_gnn_test)
        brier_ensemble = compute_brier(y_test, p_ensemble)

        acc_market = accuracy_score(y_test, p_market_test.argmax(axis=1))
        acc_xgb = accuracy_score(y_test, p_xgb_test.argmax(axis=1))
        acc_gnn = accuracy_score(y_test, p_gnn_test.argmax(axis=1))
        acc_ensemble = accuracy_score(y_test, p_ensemble.argmax(axis=1))

        print(f"\n  Results (Brier / Accuracy):")
        print(f"    Market only:   {brier_market:.4f} / {acc_market:.2%}")
        print(f"    XGBoost only:  {brier_xgb:.4f} / {acc_xgb:.2%}")
        print(f"    GNN only:      {brier_gnn:.4f} / {acc_gnn:.2%}")
        print(f"    ENSEMBLE:      {brier_ensemble:.4f} / {acc_ensemble:.2%}")

        improvement = brier_market - brier_ensemble
        print(f"    Improvement over market: {improvement:+.4f}")

        results.append({
            'fold': fold + 1,
            'test_period': f"{test_start} to {test_end}",
            'weights': {'market': float(weights[0]), 'xgboost': float(weights[1]), 'gnn': float(weights[2])},
            'brier': {
                'market': float(brier_market),
                'xgboost': float(brier_xgb),
                'gnn': float(brier_gnn),
                'ensemble': float(brier_ensemble),
            },
            'accuracy': {
                'market': float(acc_market),
                'xgboost': float(acc_xgb),
                'gnn': float(acc_gnn),
                'ensemble': float(acc_ensemble),
            },
            'improvement_over_market': float(improvement),
        })

    # Aggregate
    print(f"\n=== Aggregate Results ===")
    for source in ['market', 'xgboost', 'gnn', 'ensemble']:
        avg_brier = np.mean([r['brier'][source] for r in results])
        avg_acc = np.mean([r['accuracy'][source] for r in results])
        marker = " <<<" if source == 'ensemble' else ""
        print(f"  {source:12s}: Brier={avg_brier:.4f}, Acc={avg_acc:.2%}{marker}")

    avg_improvement = np.mean([r['improvement_over_market'] for r in results])
    print(f"\n  Mean improvement over market: {avg_improvement:+.4f}")

    avg_weights = {
        'market': np.mean([r['weights']['market'] for r in results]),
        'xgboost': np.mean([r['weights']['xgboost'] for r in results]),
        'gnn': np.mean([r['weights']['gnn'] for r in results]),
    }
    print(f"  Mean weights: market={avg_weights['market']:.2%}, xgb={avg_weights['xgboost']:.2%}, gnn={avg_weights['gnn']:.2%}")

    return results, avg_weights


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Bayesian Ensemble for BetAI')
    parser.add_argument('--features', required=True, help='Path to features JSON')
    parser.add_argument('--gnn-embeddings', required=True, help='Path to GNN embeddings JSON')
    parser.add_argument('--selected-features', default=None, help='Path to selected features JSON')
    parser.add_argument('--output', default='ensemble_results.json', help='Output path')
    args = parser.parse_args()

    print("=== BetAI v2 - Bayesian Ensemble ===\n")

    # Load data
    df = load_features(args.features)
    print(f"Loaded {len(df)} matches")

    # Load GNN embeddings
    with open(args.gnn_embeddings, 'r') as f:
        gnn_embeddings = json.load(f)
    print(f"Loaded {len(gnn_embeddings)} GNN embeddings")

    # Get feature columns
    exclude = {'matchId', 'date', 'result', 'totalGoals', 'btts',
               'closing_odds_home', 'closing_odds_draw', 'closing_odds_away'}

    if args.selected_features:
        with open(args.selected_features, 'r') as f:
            sel_data = json.load(f)
        feature_cols = [c for c in sel_data['selected_features'] if c in df.columns]
        print(f"Using {len(feature_cols)} selected features")
    else:
        feature_cols = [c for c in df.columns if c not in exclude]
        print(f"Using {len(feature_cols)} features")

    # Run ensemble evaluation
    results, avg_weights = walk_forward_ensemble(df, feature_cols, gnn_embeddings)

    # Save results
    output = {
        'results': results,
        'avg_weights': avg_weights,
        'config': {
            'n_features': len(feature_cols),
            'n_matches': len(df),
            'n_gnn_embeddings': len(gnn_embeddings),
        },
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {args.output}")
    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()
