"""
Training Model BTTS (Both Teams To Score)

NOTA: Questo modello ha performance marginali (AUC ~0.55)
NON RACCOMANDATO per betting reale.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.append(str(Path(__file__).parent.parent))
from common.utils import load_dataset, temporal_split

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


FEATURES = [
    'xG Home', 'xG Away', 'Gross xG', 'Net xG',
    'xG 1H Home', 'xG 1H Away',
    'ELO', 'ELO.1',
    'Home Overall Rank', 'Away Overall Rank',
    'Home Pos', 'Away Pos',
    'Games without Win', 'Games without Loss',
    'Games without Win.1', 'Games without Loss.1',
    'Home Team Game No.', 'Away Team Game No.',
    'Home_xG_roll5', 'Away_xG_roll5',
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features specifiche per BTTS."""
    df = df.copy()
    df['xG_min'] = np.minimum(df['xG Home'], df['xG Away'])
    df['xG_product'] = df['xG Home'] * df['xG Away']
    df['both_xG_above_05'] = ((df['xG Home'] > 0.5) & (df['xG Away'] > 0.5)).astype(int)
    df['both_xG_above_10'] = ((df['xG Home'] > 1.0) & (df['xG Away'] > 1.0)).astype(int)
    df['ELO_diff'] = df['ELO'] - df['ELO.1']
    df['ELO_min'] = np.minimum(df['ELO'], df['ELO.1'])
    return df


def train_model(save: bool = True):
    """Pipeline training BTTS."""
    print("="*70)
    print("   TRAINING BTTS MODEL")
    print("   NOTA: Performance marginali - NON RACCOMANDATO")
    print("="*70)

    df = load_dataset()
    df = create_features(df)

    all_features = FEATURES + ['xG_min', 'xG_product', 'both_xG_above_05',
                               'both_xG_above_10', 'ELO_diff', 'ELO_min']
    available = [f for f in all_features if f in df.columns]

    df_valid = df.dropna(subset=['BTTS'] + available)
    print(f"\nRighe valide: {len(df_valid):,}")

    train_df, test_df = temporal_split(df_valid, test_months=6)

    X_train = train_df[available].values
    y_train = train_df['BTTS'].values.astype(int)
    X_test = test_df[available].values
    y_test = test_df['BTTS'].values.astype(int)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
        reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbosity=-1,
    )
    lgb_model.fit(X_train, y_train)

    # Stacking
    xgb_p_train = xgb_model.predict_proba(X_train)[:, 1]
    lgb_p_train = lgb_model.predict_proba(X_train)[:, 1]
    xgb_p_test = xgb_model.predict_proba(X_test)[:, 1]
    lgb_p_test = lgb_model.predict_proba(X_test)[:, 1]

    stack_train = np.column_stack([xgb_p_train, lgb_p_train])
    stack_test = np.column_stack([xgb_p_test, lgb_p_test])

    lr = LogisticRegression(C=0.5, max_iter=1000)
    lr.fit(stack_train, y_train)

    proba_train = lr.predict_proba(stack_train)[:, 1]
    proba_test = lr.predict_proba(stack_test)[:, 1]

    # Calibrazione
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(proba_train, y_train)
    proba_test_cal = calibrator.predict(proba_test)

    # Metriche
    auc = roc_auc_score(y_test, proba_test_cal)
    brier = brier_score_loss(y_test, proba_test_cal)

    print(f"\nMetriche TEST SET:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Brier: {brier:.4f}")
    print(f"   Baseline BTTS rate: {y_train.mean():.1%}")

    if save:
        model_data = {
            'xgb': xgb_model, 'lgb': lgb_model, 'lr': lr,
            'calibrator': calibrator, 'features': available,
            'metrics': {'auc': auc, 'brier': brier},
            'trained_at': datetime.now().isoformat(),
            'warning': 'PERFORMANCE MARGINALI - NON RACCOMANDATO PER BETTING'
        }
        joblib.dump(model_data, MODEL_DIR / "model_btts_calibrated.joblib")
        print(f"\nSalvato: {MODEL_DIR / 'model_btts_calibrated.joblib'}")

    return model_data


if __name__ == "__main__":
    train_model()
