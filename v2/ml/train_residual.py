"""
Residual Model Training for BetAI v2.

Instead of predicting H/D/A from scratch, this model predicts the RESIDUAL
(correction) relative to market-implied probabilities.

The key insight: market odds already encode most available information.
Our statistical features can only add value where the market has systematic biases
(e.g., over-betting favorites, under-valuing promoted teams, fatigue effects).

Architecture:
  1. Market implied probs = baseline (from closing odds)
  2. XGBoost regression predicts residual = P(true) - P(market) per outcome
  3. Final prob = market + residual (clipped & normalized)
  4. Bet only when |residual| > threshold (model detects market inefficiency)

This avoids the "copying odds" problem: the model CANNOT copy odds because
odds are the baseline, not a feature. It can only learn corrections.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_features(path: str) -> pd.DataFrame:
    """Load features from JSON, streaming for large files."""
    file_size = os.path.getsize(path)

    if file_size > 100_000_000:
        print(f"  Large file ({file_size / 1e6:.0f} MB), streaming...")
        return _load_streaming(path)

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


def _load_streaming(path: str) -> pd.DataFrame:
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


# ──────────────────────────────────────────────
# MARKET PROBABILITIES
# ──────────────────────────────────────────────

def compute_market_probs(df: pd.DataFrame) -> np.ndarray:
    """Convert closing odds to implied probs (overround removed)."""
    probs = np.full((len(df), 3), np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        oh = row.get('closing_odds_home', 0)
        od = row.get('closing_odds_draw', 0)
        oa = row.get('closing_odds_away', 0)

        if oh > 1 and od > 1 and oa > 1:
            raw_h = 1.0 / oh
            raw_d = 1.0 / od
            raw_a = 1.0 / oa
            total = raw_h + raw_d + raw_a
            probs[i] = [raw_h / total, raw_d / total, raw_a / total]

    return probs


def compute_residual_targets(y_labels: np.ndarray, market_probs: np.ndarray) -> np.ndarray:
    """
    Compute residual targets: one_hot(result) - market_probs.

    For a Home win with market probs [0.45, 0.28, 0.27]:
      residuals = [1, 0, 0] - [0.45, 0.28, 0.27] = [0.55, -0.28, -0.27]

    The model learns to predict these corrections.
    """
    n = len(y_labels)
    one_hot = np.zeros((n, 3))
    for i, label in enumerate(y_labels):
        one_hot[i, label] = 1.0

    return one_hot - market_probs


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def compute_brier_multiclass(y_true, y_prob):
    """Multiclass Brier score (lower is better)."""
    n = len(y_true)
    y_oh = np.zeros((n, 3))
    for i, l in enumerate(y_true):
        y_oh[i, l] = 1
    return np.mean(np.sum((y_prob - y_oh) ** 2, axis=1))


def compute_clv(y_true, model_probs, market_probs):
    """
    Closing Line Value: measures if model probs are better than market.

    For each match, CLV = model_prob[actual_outcome] - market_prob[actual_outcome].
    Positive CLV means model assigned higher probability to the actual result.
    """
    clv_values = []
    for i in range(len(y_true)):
        actual = y_true[i]
        if not np.isnan(market_probs[i]).any():
            clv = model_probs[i, actual] - market_probs[i, actual]
            clv_values.append(clv)

    if not clv_values:
        return {'mean': 0, 'positive_rate': 0, 'n': 0}

    clv_arr = np.array(clv_values)
    return {
        'mean': float(np.mean(clv_arr)),
        'positive_rate': float(np.mean(clv_arr > 0)),
        'n': len(clv_arr),
    }


def compute_value_bets(model_probs, market_probs, threshold=0.03):
    """
    Find value bets: matches where model_prob > market_prob + threshold.

    Returns mask and details for each outcome.
    """
    n = len(model_probs)
    value_bets = []

    for i in range(n):
        if np.isnan(market_probs[i]).any():
            continue
        for outcome in range(3):
            edge = model_probs[i, outcome] - market_probs[i, outcome]
            if edge > threshold:
                value_bets.append({
                    'match_idx': i,
                    'outcome': outcome,  # 0=H, 1=D, 2=A
                    'edge': edge,
                    'model_prob': model_probs[i, outcome],
                    'market_prob': market_probs[i, outcome],
                })

    return value_bets


def compute_flat_yield(y_true, model_probs, market_probs, closing_odds_df=None, threshold=0.03):
    """
    Compute flat-stake yield for value bets.

    For each match, if model_prob > market_prob + threshold for any outcome,
    simulate a 1-unit flat bet. Yield = total_profit / total_staked * 100.

    Also returns CLV-based yield if closing_odds are available.
    """
    value_bets = compute_value_bets(model_probs, market_probs, threshold)

    if len(value_bets) == 0:
        return {'yield': 0.0, 'n_bets': 0, 'profit': 0.0, 'clv_yield': 0.0}

    total_profit = 0.0
    total_staked = 0
    clv_sum = 0.0

    for vb in value_bets:
        idx = vb['match_idx']
        outcome = vb['outcome']
        actual = y_true[idx]
        market_p = market_probs[idx, outcome]

        # Implied odds from market prob (overround-adjusted)
        if market_p > 0:
            fair_odds = 1.0 / market_p
        else:
            continue

        won = (actual == outcome)
        total_staked += 1

        if won:
            total_profit += fair_odds - 1
        else:
            total_profit -= 1

        # CLV: how much better is model prob vs market prob for actual outcome
        clv_sum += model_probs[idx, actual] - market_probs[idx, actual]

    flat_yield = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    clv_yield = (clv_sum / total_staked * 100) if total_staked > 0 else 0.0

    return {
        'yield': flat_yield,
        'n_bets': total_staked,
        'profit': total_profit,
        'clv_yield': clv_yield,
    }


# ──────────────────────────────────────────────
# RESIDUAL MODEL
# ──────────────────────────────────────────────

class ResidualModel:
    """
    Trains 3 XGBoost regressors to predict residuals for H, D, A.

    Each regressor predicts: P(outcome) - market_P(outcome)
    using only our statistical features (no odds).
    """

    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }
        self.models = [None, None, None]  # H, D, A

    def fit(self, X_train, residuals_train, X_val=None, residuals_val=None):
        """Train 3 regressors for H/D/A residuals."""
        for outcome in range(3):
            y_train = residuals_train[:, outcome]

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                seed=42,
                verbosity=0,
                early_stopping_rounds=20,
                **self.params
            )

            if X_val is not None and residuals_val is not None:
                y_val = residuals_val[:, outcome]
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)

            self.models[outcome] = model

    def predict_residuals(self, X):
        """Predict residuals for all 3 outcomes."""
        residuals = np.zeros((len(X), 3))
        for outcome in range(3):
            residuals[:, outcome] = self.models[outcome].predict(X)
        return residuals

    def predict_probs(self, X, market_probs):
        """
        Final prediction: market_probs + predicted_residuals.
        Clipped to [0.01, 0.99] and normalized to sum to 1.
        """
        residuals = self.predict_residuals(X)
        raw_probs = market_probs + residuals

        # Clip to valid range
        raw_probs = np.clip(raw_probs, 0.01, 0.99)

        # Normalize to sum to 1
        row_sums = raw_probs.sum(axis=1, keepdims=True)
        probs = raw_probs / row_sums

        return probs


# ──────────────────────────────────────────────
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ──────────────────────────────────────────────

def optimize_residual(X_train, residuals_train, X_val, residuals_val,
                      market_probs_val, y_val, n_trials=30,
                      optimize_for='clv_yield'):
    """
    Optimize residual model hyperparameters using Optuna.

    optimize_for options:
      - 'brier': minimize Brier score (original, calibration-focused)
      - 'clv_yield': maximize composite of CLV + flat yield (betting-focused)
      - 'clv': maximize mean CLV (probability accuracy on actual outcomes)

    The CLV/yield objective is preferred for betting because:
    - Brier rewards overall calibration including non-bet scenarios
    - CLV/yield directly measures profitability where we bet
    - Brier is kept as a constraint (must not exceed market Brier by >0.01)
    """
    market_brier_baseline = compute_brier_multiclass(y_val, market_probs_val)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        }

        model = ResidualModel(params)
        model.fit(X_train, residuals_train, X_val, residuals_val)
        pred_probs = model.predict_probs(X_val, market_probs_val)

        if optimize_for == 'brier':
            return compute_brier_multiclass(y_val, pred_probs)

        # Brier constraint: don't degrade calibration too much
        brier = compute_brier_multiclass(y_val, pred_probs)
        if brier > market_brier_baseline + 0.01:
            # Penalize heavily: model is worse than market
            return -100.0

        # Compute betting metrics
        clv = compute_clv(y_val, pred_probs, market_probs_val)
        yield_data = compute_flat_yield(y_val, pred_probs, market_probs_val, threshold=0.03)

        if optimize_for == 'clv':
            return clv['mean']  # maximize

        # 'clv_yield': composite score
        # Weight CLV more than yield because yield is noisy on small samples
        # CLV is the strongest predictor of long-term profitability
        clv_score = clv['mean'] * 100        # e.g. 0.006 → 0.6
        yield_score = yield_data['yield']     # already in %
        bet_volume = yield_data['n_bets']

        # Penalize if too few bets (model is too conservative)
        volume_penalty = 0 if bet_volume >= 50 else -5.0

        composite = 0.7 * clv_score + 0.3 * yield_score + volume_penalty
        return composite

    direction = 'minimize' if optimize_for == 'brier' else 'maximize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    label = 'Brier' if optimize_for == 'brier' else 'CLV/Yield composite'
    print(f"  Best {label}: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


# ──────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ──────────────────────────────────────────────

def walk_forward_validation(df, feature_cols, n_splits=5, n_trials=30):
    """
    Walk-forward validation of the residual model.

    Only evaluates on matches WITH closing odds (needed for residuals).
    """
    print(f"\n=== Walk-Forward Validation ({n_splits} folds) ===")

    # Filter to matches with closing odds
    has_odds = df['closing_odds_home'].notna() & (df['closing_odds_home'] > 1)
    df_odds = df[has_odds].reset_index(drop=True)
    n_no_odds = len(df) - len(df_odds)
    print(f"  Matches with odds: {len(df_odds)} (excluded {n_no_odds} without odds)")

    label_map = {'H': 0, 'D': 1, 'A': 2}
    y_all = df_odds['result'].map(label_map).values
    X_all = df_odds[feature_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    market_probs_all = compute_market_probs(df_odds)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        market_train = market_probs_all[train_idx]
        market_test = market_probs_all[test_idx]

        # Compute residual targets for training
        residuals_train = compute_residual_targets(y_train, market_train)

        # Split train into train/val for Optuna
        val_size = int(len(X_train) * 0.15)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        res_tr = residuals_train[:-val_size]
        res_val = residuals_train[-val_size:]
        market_val = market_train[-val_size:]

        # Optimize hyperparameters
        best_params = optimize_residual(
            X_tr, res_tr, X_val, res_val,
            market_val, y_val, n_trials=n_trials
        )

        # Train final model on full training set
        model = ResidualModel(best_params)
        model.fit(X_train, residuals_train, X_val, res_val)

        # Predict on test
        pred_probs = model.predict_probs(X_test, market_test)

        # Also compute market-only Brier for comparison
        market_brier = compute_brier_multiclass(y_test, market_test)

        # Metrics
        brier = compute_brier_multiclass(y_test, pred_probs)
        accuracy = np.mean(np.argmax(pred_probs, axis=1) == y_test)
        clv = compute_clv(y_test, pred_probs, market_test)

        # Value bets analysis
        value_bets = compute_value_bets(pred_probs, market_test, threshold=0.03)
        n_value = len(value_bets)
        if n_value > 0:
            value_correct = sum(
                1 for vb in value_bets
                if y_test[vb['match_idx']] == vb['outcome']
            )
            value_roi = value_correct / n_value if n_value > 0 else 0
            avg_edge = np.mean([vb['edge'] for vb in value_bets])
        else:
            value_correct = 0
            value_roi = 0
            avg_edge = 0

        # Flat-stake yield (the betting bottom line)
        yield_data = compute_flat_yield(y_test, pred_probs, market_test, threshold=0.03)

        test_start = df_odds.iloc[test_idx[0]]['date']
        test_end = df_odds.iloc[test_idx[-1]]['date']

        result = {
            'fold': fold + 1,
            'period': f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
            'train_size': len(X_train),
            'test_size': len(X_test),
            'brier': brier,
            'market_brier': market_brier,
            'brier_improvement': market_brier - brier,
            'accuracy': accuracy,
            'clv_mean': clv['mean'],
            'clv_positive_rate': clv['positive_rate'],
            'value_bets': n_value,
            'value_correct': value_correct,
            'value_hit_rate': value_roi,
            'avg_edge': avg_edge,
            'flat_yield': yield_data['yield'],
            'flat_profit': yield_data['profit'],
            'clv_yield': yield_data['clv_yield'],
        }
        results.append(result)

        print(f"\nFold {fold + 1}: {result['period']}")
        print(f"  Train: {result['train_size']}, Test: {result['test_size']}")
        print(f"  Model Brier:  {brier:.4f}")
        print(f"  Market Brier: {market_brier:.4f}")
        print(f"  Improvement:  {result['brier_improvement']:+.4f} {'(better)' if result['brier_improvement'] > 0 else '(worse)'}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  CLV: mean={clv['mean']:+.2%}, positive_rate={clv['positive_rate']:.1%}")
        print(f"  Value bets (>3% edge): {n_value} ({n_value/len(X_test):.1%} of matches)")
        if n_value > 0:
            print(f"    Hit rate: {value_roi:.1%}, Avg edge: {avg_edge:.2%}")
        print(f"  Flat-stake yield: {yield_data['yield']:+.2f}% ({yield_data['n_bets']} bets)")

    # Summary
    print(f"\n{'='*50}")
    print(f"=== SUMMARY ({n_splits} folds) ===")
    print(f"{'='*50}")

    avg_brier = np.mean([r['brier'] for r in results])
    avg_market = np.mean([r['market_brier'] for r in results])
    avg_improvement = np.mean([r['brier_improvement'] for r in results])
    avg_clv = np.mean([r['clv_mean'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    total_value = sum(r['value_bets'] for r in results)
    total_value_correct = sum(r['value_correct'] for r in results)
    avg_yield = np.mean([r['flat_yield'] for r in results])
    total_profit = sum(r['flat_profit'] for r in results)

    print(f"  Model Brier (avg):  {avg_brier:.4f}")
    print(f"  Market Brier (avg): {avg_market:.4f}")
    print(f"  Improvement (avg):  {avg_improvement:+.4f}")
    print(f"  Accuracy (avg):     {avg_accuracy:.1%}")
    print(f"  CLV (avg):          {avg_clv:+.2%}")
    print(f"  Flat yield (avg):   {avg_yield:+.2f}%")
    print(f"  Total flat profit:  {total_profit:+.1f} units")
    print(f"  Total value bets:   {total_value}")
    if total_value > 0:
        print(f"  Value hit rate:     {total_value_correct/total_value:.1%}")

    if avg_yield > 0:
        print(f"\n  PROFITABLE: {avg_yield:+.2f}% avg yield, {total_profit:+.1f} units total")
    elif avg_improvement > 0:
        print(f"\n  Model BEATS market Brier by {avg_improvement:.4f} but yield is negative")
        print(f"  Edge exists but not yet converted to profit — check thresholds")
    else:
        print(f"\n  Model LOSES to market by {-avg_improvement:.4f} Brier points")

    return results


# ──────────────────────────────────────────────
# FINAL MODEL TRAINING
# ──────────────────────────────────────────────

def train_final_model(df, feature_cols, output_path, n_trials=50):
    """Train final residual model on all data and save."""
    print(f"\n=== Training Final Residual Model ===")

    # Filter to matches with odds
    has_odds = df['closing_odds_home'].notna() & (df['closing_odds_home'] > 1)
    df_odds = df[has_odds].reset_index(drop=True)

    label_map = {'H': 0, 'D': 1, 'A': 2}
    y = df_odds['result'].map(label_map).values
    X = df_odds[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    market_probs = compute_market_probs(df_odds)
    residuals = compute_residual_targets(y, market_probs)

    # Split for validation
    val_size = int(len(X) * 0.15)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    res_train, res_val = residuals[:-val_size], residuals[-val_size:]
    market_val = market_probs[-val_size:]

    # Optimize
    best_params = optimize_residual(
        X_train, res_train, X_val, res_val,
        market_val, y_val, n_trials=n_trials
    )

    # Train on full training set
    model = ResidualModel(best_params)
    model.fit(X_train, res_train, X_val, res_val)

    # Evaluate on validation
    pred_probs = model.predict_probs(X_val, market_val)
    brier = compute_brier_multiclass(y_val, pred_probs)
    market_brier = compute_brier_multiclass(y_val, market_val)
    clv = compute_clv(y_val, pred_probs, market_val)

    print(f"  Validation Brier:  {brier:.4f}")
    print(f"  Market Brier:      {market_brier:.4f}")
    print(f"  Improvement:       {market_brier - brier:+.4f}")
    print(f"  CLV: {clv['mean']:+.2%}")

    # Save model configs (XGBoost models saved separately)
    output = {
        'type': 'residual_model',
        'created': datetime.now().isoformat(),
        'features': feature_cols,
        'n_features': len(feature_cols),
        'best_params': best_params,
        'validation': {
            'brier': float(brier),
            'market_brier': float(market_brier),
            'improvement': float(market_brier - brier),
            'clv_mean': clv['mean'],
            'clv_positive_rate': clv['positive_rate'],
        },
        'training_matches': len(df_odds),
    }

    # Save XGBoost models
    model_dir = os.path.dirname(output_path)
    for outcome, name in enumerate(['home', 'draw', 'away']):
        model_path = os.path.join(model_dir, f'residual_{name}.json')
        model.models[outcome].save_model(model_path)
        output[f'model_{name}_path'] = model_path

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved to: {output_path}")


# ──────────────────────────────────────────────
# DATA SUMMARY
# ──────────────────────────────────────────────

def print_data_summary(df, feature_cols):
    """Print data understanding summary."""
    print(f"\n=== Data Summary ===")
    print(f"  Total matches: {len(df)}")

    has_odds = df['closing_odds_home'].notna() & (df['closing_odds_home'] > 1)
    n_with_odds = has_odds.sum()
    print(f"  Matches with closing odds: {n_with_odds} ({n_with_odds/len(df):.1%})")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Features: {len(feature_cols)}")

    # Class distribution
    counts = df['result'].value_counts()
    total = len(df)
    print(f"\n  Class distribution:")
    for label in ['H', 'D', 'A']:
        c = counts.get(label, 0)
        print(f"    {label}: {c} ({c/total:.1%})")

    # Market calibration check (how good are the odds?)
    df_odds = df[has_odds].copy()
    if len(df_odds) > 1000:
        market_probs = compute_market_probs(df_odds)
        label_map = {'H': 0, 'D': 1, 'A': 2}
        y = df_odds['result'].map(label_map).values
        market_brier = compute_brier_multiclass(y, market_probs)
        print(f"\n  Market baseline Brier: {market_brier:.4f}")
        print(f"  (this is the score to beat)")

    # Feature quality
    X = df[feature_cols].values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"\n  NaN values: {nan_count} ({nan_count / X.size:.2%})")
    else:
        print(f"\n  Data quality: clean (no NaN/inf)")
    print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train residual model')
    parser.add_argument('--data', required=True, help='Path to features.json')
    parser.add_argument('--output', default='residual_model.json', help='Output path')
    parser.add_argument('--selected-features', help='Path to selected_features.json')
    parser.add_argument('--n-splits', type=int, default=5, help='Walk-forward folds')
    parser.add_argument('--n-trials', type=int, default=30, help='Optuna trials per fold')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, no final training')
    args = parser.parse_args()

    print("=== BetAI v2 - Residual Model Training ===\n")

    # Load data
    print(f"Loading features from: {args.data}")
    df = load_features(args.data)
    print(f"  Loaded {len(df)} matches")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Determine feature columns
    exclude_cols = {'matchId', 'date', 'result', 'totalGoals', 'btts',
                    'closing_odds_home', 'closing_odds_draw', 'closing_odds_away'}

    if args.selected_features and os.path.exists(args.selected_features):
        with open(args.selected_features) as f:
            sel_data = json.load(f)
        feature_cols = sel_data['selected_features']
        feature_cols = [c for c in feature_cols if c in df.columns]
        print(f"  Features: {len(feature_cols)} (from selection, was {sel_data['n_original']})")
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        print(f"  Features: {len(feature_cols)}")

    # Data summary
    print_data_summary(df, feature_cols)

    # Walk-forward validation
    results = walk_forward_validation(df, feature_cols,
                                       n_splits=args.n_splits,
                                       n_trials=args.n_trials)

    if args.validate_only:
        print("\n[validate-only mode, skipping final training]")
        return

    # Train final model
    train_final_model(df, feature_cols, args.output, n_trials=50)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
