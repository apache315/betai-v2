"""
Training Model Over 2.5

FEATURES VALIDATE (no data leakage):
- xG predictions (pre-match)
- ELO ratings (pre-match)
- Form indicators (pre-match)
- Rolling stats con shift

CALIBRAZIONE: Isotonic per correggere overconfidence
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.append(str(Path(__file__).parent.parent))
from common.utils import load_dataset, temporal_split

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# Features SICURE - verificate senza leakage
FEATURES = [
    # xG core (pre-match predictions)
    'xG Home', 'xG Away', 'Gross xG', 'Net xG',
    'xG 1H Home', 'xG 1H Away',

    # ELO (pre-match)
    'ELO', 'ELO.1',

    # Rankings (pre-match)
    'Home Overall Rank', 'Away Overall Rank',
    'Home Pos', 'Away Pos',

    # Form (pre-match)
    'Games without Win', 'Games without Loss',
    'Games without Win.1', 'Games without Loss.1',
    'Home Team Game No.', 'Away Team Game No.',

    # Rolling con shift (verificato)
    'Home_xG_roll5', 'Away_xG_roll5',
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features specifiche per Over 2.5."""
    df = df.copy()

    # Derived features
    df['xG_total'] = df['xG Home'] + df['xG Away']
    df['xG_diff_abs'] = np.abs(df['Net xG'])
    df['ELO_diff'] = df['ELO'] - df['ELO.1']
    df['ELO_avg'] = (df['ELO'] + df['ELO.1']) / 2
    df['ELO_min'] = np.minimum(df['ELO'], df['ELO.1'])
    df['Rank_diff'] = df['Home Overall Rank'] - df['Away Overall Rank']
    df['xG_1H_total'] = df['xG 1H Home'] + df['xG 1H Away']

    return df


def train_model(save: bool = True):
    """
    Pipeline training completa con calibrazione.
    """
    print("="*70)
    print("   TRAINING OVER 2.5 MODEL - CALIBRATO")
    print("="*70)

    # 1. Load data
    print("\n1. Caricamento dati...")
    df = load_dataset()
    df = create_features(df)

    # 2. Prepare features
    all_features = FEATURES + ['xG_total', 'xG_diff_abs', 'ELO_diff', 'ELO_avg', 'ELO_min', 'Rank_diff', 'xG_1H_total']
    available = [f for f in all_features if f in df.columns]

    print(f"   Features disponibili: {len(available)}")

    # 3. Remove NaN
    df_valid = df.dropna(subset=['Over25'] + available)
    print(f"   Righe valide: {len(df_valid):,}")

    # 4. Temporal split
    print("\n2. Split temporale...")
    train_df, test_df = temporal_split(df_valid, test_months=6)
    print(f"   Train: {len(train_df):,} ({train_df['Date'].min().date()} - {train_df['Date'].max().date()})")
    print(f"   Test:  {len(test_df):,} ({test_df['Date'].min().date()} - {test_df['Date'].max().date()})")

    X_train = train_df[available].values
    y_train = train_df['Over25'].values.astype(int)
    X_test = test_df[available].values
    y_test = test_df['Over25'].values.astype(int)

    # 5. Train base models
    print("\n3. Training modelli base...")

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(X_train, y_train)

    # 6. Stacking
    print("\n4. Stacking ensemble...")
    xgb_proba_train = xgb_model.predict_proba(X_train)[:, 1]
    lgb_proba_train = lgb_model.predict_proba(X_train)[:, 1]
    xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]
    lgb_proba_test = lgb_model.predict_proba(X_test)[:, 1]

    stack_train = np.column_stack([xgb_proba_train, lgb_proba_train])
    stack_test = np.column_stack([xgb_proba_test, lgb_proba_test])

    lr = LogisticRegression(C=0.5, max_iter=1000)
    lr.fit(stack_train, y_train)

    proba_train_raw = lr.predict_proba(stack_train)[:, 1]
    proba_test_raw = lr.predict_proba(stack_test)[:, 1]

    # 7. Calibrazione con Isotonic Regression
    print("\n5. Calibrazione probabilita...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(proba_train_raw, y_train)

    proba_test_calibrated = calibrator.predict(proba_test_raw)

    # 8. Metriche
    print("\n6. Metriche sul TEST SET:")

    auc_raw = roc_auc_score(y_test, proba_test_raw)
    auc_cal = roc_auc_score(y_test, proba_test_calibrated)
    brier_raw = brier_score_loss(y_test, proba_test_raw)
    brier_cal = brier_score_loss(y_test, proba_test_calibrated)

    print(f"   AUC (raw):        {auc_raw:.4f}")
    print(f"   AUC (calibrated): {auc_cal:.4f}")
    print(f"   Brier (raw):      {brier_raw:.4f}")
    print(f"   Brier (calib):    {brier_cal:.4f}")

    # 9. Verifica calibrazione
    print("\n7. Verifica calibrazione:")
    test_df_eval = test_df.copy()
    test_df_eval['P_raw'] = proba_test_raw
    test_df_eval['P_cal'] = proba_test_calibrated

    for low, high in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]:
        mask = (test_df_eval['P_cal'] >= low) & (test_df_eval['P_cal'] < high)
        if mask.sum() > 50:
            p_real = test_df_eval[mask]['Over25'].mean()
            expected = (low + high) / 2
            gap = (p_real - expected) * 100
            print(f"   P=[{low:.0%}-{high:.0%}]: reale={p_real:.1%}, expected={expected:.0%}, gap={gap:+.1f}pp (n={mask.sum()})")

    # 10. Save
    if save:
        print("\n8. Salvataggio modello...")
        model_data = {
            'xgb': xgb_model,
            'lgb': lgb_model,
            'lr': lr,
            'calibrator': calibrator,
            'features': available,
            'metrics': {
                'auc': auc_cal,
                'brier': brier_cal,
            },
            'trained_at': datetime.now().isoformat(),
            'train_period': f"{train_df['Date'].min().date()} - {train_df['Date'].max().date()}",
            'test_period': f"{test_df['Date'].min().date()} - {test_df['Date'].max().date()}",
        }
        joblib.dump(model_data, MODEL_DIR / "model_over25_calibrated.joblib")
        print(f"   Salvato: {MODEL_DIR / 'model_over25_calibrated.joblib'}")

    return model_data


if __name__ == "__main__":
    train_model()
