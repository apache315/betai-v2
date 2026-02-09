#!/usr/bin/env python3
"""
Feature Selection for BetAI v2

Two methods:
1. Boruta: finds ALL relevant features (wrapper around Random Forest)
2. Feature Ablation: removes one feature at a time, measures Brier delta

Usage:
    python feature_selection.py --data features.json --output selected_features.json
    python feature_selection.py --data features.json --method boruta
    python feature_selection.py --data features.json --method ablation
    python feature_selection.py --data features.json --method both
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import xgboost as xgb

warnings.filterwarnings('ignore')


def load_features(path: str) -> pd.DataFrame:
    """Load features from JSON, using streaming parser for large files."""
    import os
    file_size = os.path.getsize(path)

    if file_size > 100_000_000:  # > 100MB: use streaming ijson if available, else chunked
        print(f"  Large file ({file_size / 1e6:.0f} MB), using streaming loader...")
        return _load_features_streaming(path)

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


def _load_features_streaming(path: str) -> pd.DataFrame:
    """Stream-parse large JSON arrays line by line to avoid full memory load."""
    import re
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

    print(f"    Total: {count} matches loaded")
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {'matchId', 'date', 'result', 'totalGoals', 'btts',
               'closing_odds_home', 'closing_odds_draw', 'closing_odds_away'}
    return [c for c in df.columns if c not in exclude]


def compute_brier(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_oh = np.zeros((len(y_true), n_classes))
    for i, l in enumerate(y_true):
        y_oh[i, l] = 1
    return np.mean(np.sum((y_prob - y_oh) ** 2, axis=1))


# ──────────────────────────────────────────────
# BORUTA
# ──────────────────────────────────────────────

def run_boruta(df: pd.DataFrame, feature_cols: list, n_iterations: int = 50, max_samples: int = 50000) -> dict:
    """
    Boruta feature selection.

    Creates shadow features (random permutations), trains RF,
    compares real feature importance to shadow max.
    Features that consistently beat shadows are "confirmed".
    """
    print("\n=== BORUTA Feature Selection ===")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Iterations: {n_iterations}")

    label_map = {'H': 0, 'D': 1, 'A': 2}

    # Sample if dataset is too large (RF doesn't need all rows for importance)
    if len(df) > max_samples:
        print(f"  Sampling {max_samples} from {len(df)} matches (using latest data)")
        df_boruta = df.tail(max_samples).reset_index(drop=True)
    else:
        df_boruta = df

    y = df_boruta['result'].map(label_map).values
    X = df_boruta[feature_cols].values
    print(f"  Training samples: {len(y)}")

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = X.shape[1]

    # Track hits: how many times each feature beats shadow max
    hits = np.zeros(n_features, dtype=int)

    for iteration in range(n_iterations):
        # Create shadow features (random permutation of each column)
        X_shadow = np.copy(X)
        for col in range(n_features):
            np.random.shuffle(X_shadow[:, col])

        # Combine real + shadow
        X_combined = np.hstack([X, X_shadow])

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            n_jobs=-1,
            random_state=iteration,
        )
        rf.fit(X_combined, y)

        # Feature importances
        importances = rf.feature_importances_
        real_imp = importances[:n_features]
        shadow_imp = importances[n_features:]
        shadow_max = np.max(shadow_imp)

        # Count hits (real feature > shadow max)
        hits += (real_imp > shadow_max).astype(int)

        if (iteration + 1) % 10 == 0:
            confirmed = np.sum(hits > iteration * 0.6)
            print(f"  Iteration {iteration + 1}/{n_iterations}: {confirmed} features confirmed so far")

    # Statistical test: binomial with p=0.5
    # Feature is "confirmed" if hits > threshold (e.g., 60% of iterations)
    # Feature is "rejected" if hits < lower threshold (e.g., 40%)
    # Otherwise "tentative"

    confirm_threshold = int(n_iterations * 0.6)
    reject_threshold = int(n_iterations * 0.4)

    confirmed = []
    tentative = []
    rejected = []

    for i, col in enumerate(feature_cols):
        if hits[i] >= confirm_threshold:
            confirmed.append({'feature': col, 'hits': int(hits[i]), 'status': 'confirmed'})
        elif hits[i] <= reject_threshold:
            rejected.append({'feature': col, 'hits': int(hits[i]), 'status': 'rejected'})
        else:
            tentative.append({'feature': col, 'hits': int(hits[i]), 'status': 'tentative'})

    # Sort by hits
    confirmed.sort(key=lambda x: -x['hits'])
    tentative.sort(key=lambda x: -x['hits'])
    rejected.sort(key=lambda x: -x['hits'])

    print(f"\n  Results:")
    print(f"    Confirmed: {len(confirmed)} features")
    print(f"    Tentative: {len(tentative)} features")
    print(f"    Rejected:  {len(rejected)} features")

    if confirmed:
        print(f"\n  Top confirmed features:")
        for f in confirmed[:15]:
            print(f"    {f['feature']}: {f['hits']}/{n_iterations} hits")

    return {
        'confirmed': confirmed,
        'tentative': tentative,
        'rejected': rejected,
        'n_iterations': n_iterations,
    }


# ──────────────────────────────────────────────
# FEATURE ABLATION
# ──────────────────────────────────────────────

def run_ablation(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Feature ablation: remove one feature at a time, measure Brier delta.

    Uses time-series split to avoid look-ahead.
    Features that INCREASE Brier when removed are important.
    Features that DECREASE Brier when removed are harmful (overfitting).
    """
    print("\n=== Feature Ablation ===")
    print(f"  Features to test: {len(feature_cols)}")

    label_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['result'].map(label_map).values
    X_full = df[feature_cols].values
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Use last fold of time series split for evaluation
    tscv = TimeSeriesSplit(n_splits=3)
    folds = list(tscv.split(X_full))
    train_idx, test_idx = folds[-1]  # Use last (largest) fold

    X_train_full = X_full[train_idx]
    X_test_full = X_full[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Baseline Brier with all features
    base_model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=200, max_depth=5, learning_rate=0.05,
        tree_method='hist', seed=42, verbosity=0,
    )
    base_model.fit(X_train_full, y_train)
    base_probs = base_model.predict_proba(X_test_full)
    base_brier = compute_brier(y_test, base_probs)

    print(f"  Baseline Brier (all {len(feature_cols)} features): {base_brier:.4f}")

    # Ablation: remove each feature one at a time
    results = []

    for i, col in enumerate(feature_cols):
        # Remove feature i
        mask = list(range(len(feature_cols)))
        mask.remove(i)

        X_train_abl = X_train_full[:, mask]
        X_test_abl = X_test_full[:, mask]

        model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            tree_method='hist', seed=42, verbosity=0,
        )
        model.fit(X_train_abl, y_train)
        probs = model.predict_proba(X_test_abl)
        brier = compute_brier(y_test, probs)

        delta = brier - base_brier  # Positive = removal hurts (feature is useful)

        results.append({
            'feature': col,
            'brier_without': float(brier),
            'delta': float(delta),  # >0 means feature is important
        })

        if (i + 1) % 20 == 0:
            print(f"  Tested {i + 1}/{len(feature_cols)} features...")

    # Sort by delta (most important first = highest positive delta)
    results.sort(key=lambda x: -x['delta'])

    important = [r for r in results if r['delta'] > 0.001]    # Removal hurts
    neutral = [r for r in results if -0.001 <= r['delta'] <= 0.001]
    harmful = [r for r in results if r['delta'] < -0.001]     # Removal helps

    print(f"\n  Results:")
    print(f"    Important (removal hurts): {len(important)} features")
    print(f"    Neutral:                   {len(neutral)} features")
    print(f"    Harmful  (removal helps):  {len(harmful)} features")

    if important:
        print(f"\n  Top important features:")
        for r in important[:15]:
            print(f"    {r['feature']}: delta={r['delta']:+.4f}")

    if harmful:
        print(f"\n  Harmful features (should remove):")
        for r in harmful[:10]:
            print(f"    {r['feature']}: delta={r['delta']:+.4f}")

    return {
        'baseline_brier': float(base_brier),
        'important': important,
        'neutral': neutral,
        'harmful': harmful,
        'all_results': results,
    }


# ──────────────────────────────────────────────
# COMBINE METHODS
# ──────────────────────────────────────────────

def combine_selections(boruta_result: dict, ablation_result: dict, feature_cols: list) -> list:
    """
    Combine Boruta + Ablation to get final feature set.

    Rules:
    1. Confirmed by Boruta AND important in ablation → KEEP
    2. Confirmed by Boruta OR important in ablation → KEEP (lenient)
    3. Rejected by Boruta AND harmful in ablation → REMOVE
    4. Everything else → KEEP if Boruta tentative
    """
    print("\n=== Combining Boruta + Ablation ===")

    boruta_confirmed = {f['feature'] for f in boruta_result['confirmed']}
    boruta_tentative = {f['feature'] for f in boruta_result['tentative']}
    boruta_rejected = {f['feature'] for f in boruta_result['rejected']}

    ablation_harmful = {r['feature'] for r in ablation_result['harmful']}
    ablation_important = {r['feature'] for r in ablation_result['important']}

    selected = []
    removed = []

    for col in feature_cols:
        # Rule 3: Both say remove → REMOVE
        if col in boruta_rejected and col in ablation_harmful:
            removed.append({'feature': col, 'reason': 'rejected_by_both'})
            continue

        # Rule 1 & 2: Either says keep → KEEP
        if col in boruta_confirmed or col in ablation_important:
            selected.append(col)
            continue

        # Rule 4: Boruta tentative → KEEP
        if col in boruta_tentative:
            selected.append(col)
            continue

        # Ablation neutral + Boruta rejected → REMOVE
        if col in boruta_rejected:
            removed.append({'feature': col, 'reason': 'boruta_rejected'})
            continue

        # Default: keep
        selected.append(col)

    print(f"  Selected: {len(selected)} features (from {len(feature_cols)})")
    print(f"  Removed:  {len(removed)} features")

    if removed:
        print(f"\n  Removed features:")
        for r in removed:
            print(f"    {r['feature']} ({r['reason']})")

    return selected


def main():
    parser = argparse.ArgumentParser(description='Feature selection for BetAI')
    parser.add_argument('--data', required=True, help='Path to features JSON')
    parser.add_argument('--output', default='selected_features.json', help='Output path')
    parser.add_argument('--method', default='both', choices=['boruta', 'ablation', 'both'])
    parser.add_argument('--boruta-iterations', type=int, default=50)
    args = parser.parse_args()

    print("=== BetAI v2 - Feature Selection ===\n")

    df = load_features(args.data)
    feature_cols = get_feature_cols(df)
    print(f"Loaded {len(df)} matches, {len(feature_cols)} features")

    boruta_result = None
    ablation_result = None
    selected = feature_cols  # Default: all

    if args.method in ('boruta', 'both'):
        boruta_result = run_boruta(df, feature_cols, n_iterations=args.boruta_iterations)

    if args.method in ('ablation', 'both'):
        ablation_result = run_ablation(df, feature_cols)

    if boruta_result and ablation_result:
        selected = combine_selections(boruta_result, ablation_result, feature_cols)
    elif boruta_result:
        selected = [f['feature'] for f in boruta_result['confirmed']] + \
                   [f['feature'] for f in boruta_result['tentative']]
    elif ablation_result:
        # Remove harmful, keep everything else
        harmful_set = {r['feature'] for r in ablation_result['harmful']}
        selected = [c for c in feature_cols if c not in harmful_set]

    # Validate: retrain with selected features
    print(f"\n=== Validation: {len(selected)} selected features ===")

    label_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['result'].map(label_map).values
    X_sel = df[selected].values
    X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

    tscv = TimeSeriesSplit(n_splits=3)
    briers = []
    for train_idx, test_idx in tscv.split(X_sel):
        model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            tree_method='hist', seed=42, verbosity=0,
        )
        model.fit(X_sel[train_idx], y[train_idx])
        probs = model.predict_proba(X_sel[test_idx])
        briers.append(compute_brier(y[test_idx], probs))

    mean_brier = np.mean(briers)
    print(f"  Mean Brier (selected): {mean_brier:.4f}")

    # Compare with all features
    X_all = df[feature_cols].values
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    briers_all = []
    for train_idx, test_idx in tscv.split(X_all):
        model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            tree_method='hist', seed=42, verbosity=0,
        )
        model.fit(X_all[train_idx], y[train_idx])
        probs = model.predict_proba(X_all[test_idx])
        briers_all.append(compute_brier(y[test_idx], probs))

    mean_brier_all = np.mean(briers_all)
    print(f"  Mean Brier (all {len(feature_cols)}):    {mean_brier_all:.4f}")
    print(f"  Improvement: {mean_brier_all - mean_brier:+.4f}")

    # Save results
    output = {
        'selected_features': selected,
        'n_original': len(feature_cols),
        'n_selected': len(selected),
        'brier_selected': float(mean_brier),
        'brier_all': float(mean_brier_all),
        'improvement': float(mean_brier_all - mean_brier),
    }

    if boruta_result:
        output['boruta'] = boruta_result
    if ablation_result:
        output['ablation'] = {
            'baseline_brier': ablation_result['baseline_brier'],
            'important_count': len(ablation_result['important']),
            'harmful_count': len(ablation_result['harmful']),
            'harmful_features': [r['feature'] for r in ablation_result['harmful']],
        }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {args.output}")
    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()
