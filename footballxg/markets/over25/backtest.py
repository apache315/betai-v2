"""
Backtest Over 2.5 - Versione Corretta

CORREZIONI APPLICATE:
1. Quote empiriche (non Poisson)
2. Solo test set
3. Modello calibrato
4. Metriche realistiche
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from common.utils import (
    load_dataset, temporal_split, calculate_roi,
    calculate_edge, kelly_criterion, flat_stake,
    calculate_drawdown, sharpe_ratio, empirical_probability
)

MODEL_DIR = Path(__file__).parent / "models"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features per Over 2.5."""
    df = df.copy()
    df['xG_total'] = df['xG Home'] + df['xG Away']
    df['xG_diff_abs'] = np.abs(df['Net xG'])
    df['ELO_diff'] = df['ELO'] - df['ELO.1']
    df['ELO_avg'] = (df['ELO'] + df['ELO.1']) / 2
    df['ELO_min'] = np.minimum(df['ELO'], df['ELO.1'])
    df['Rank_diff'] = df['Home Overall Rank'] - df['Away Overall Rank']
    df['xG_1H_total'] = df['xG 1H Home'] + df['xG 1H Away']
    return df


def run_backtest(
    prob_threshold: float = 0.55,
    min_edge: float = 0.03,
    min_odds: float = 1.5,
    max_odds: float = 3.5,
    stake_mode: str = 'flat',
    stake_pct: float = 0.02,
    kelly_fraction: float = 0.15,
    initial_bankroll: float = 1000,
    test_months: int = 6,
):
    """
    Backtest con quote empiriche e modello calibrato.
    """
    print("="*70)
    print("   BACKTEST OVER 2.5 - CORRETTO")
    print("="*70)

    # Load data
    df = load_dataset()
    df = create_features(df)

    # Split
    train_df, test_df = temporal_split(df, test_months)
    print(f"\nPeriodo test: {test_df['Date'].min().date()} -> {test_df['Date'].max().date()}")
    print(f"Partite: {len(test_df):,}")

    # Load model
    model_path = MODEL_DIR / "model_over25_calibrated.joblib"
    if not model_path.exists():
        print(f"\nERRORE: Modello non trovato. Esegui prima train.py")
        return None

    model = joblib.load(model_path)

    # Predictions
    features = [f for f in model['features'] if f in test_df.columns]
    X = test_df[features].values

    xgb_p = model['xgb'].predict_proba(X)[:, 1]
    lgb_p = model['lgb'].predict_proba(X)[:, 1]
    stack = np.column_stack([xgb_p, lgb_p])
    proba_raw = model['lr'].predict_proba(stack)[:, 1]
    test_df['P_model'] = model['calibrator'].predict(proba_raw)

    # Empirical odds from training data
    def get_empirical_odds(gross_xg):
        p = empirical_probability(train_df, 'Gross xG', 'Over25', gross_xg, window=0.3, min_samples=50)
        if p < 0.1:
            return 10.0
        return (1 / p) * 0.95  # 5% margin

    test_df['odds_empirical'] = test_df['Gross xG'].apply(get_empirical_odds)
    test_df['edge'] = test_df.apply(
        lambda r: calculate_edge(r['P_model'], r['odds_empirical']), axis=1
    )

    # Simulation
    print(f"\nParametri:")
    print(f"   P threshold: {prob_threshold:.0%}")
    print(f"   Min edge: {min_edge:.0%}")
    print(f"   Quote range: {min_odds} - {max_odds}")
    print(f"   Stake mode: {stake_mode}")

    bankroll = initial_bankroll
    bets = []

    for _, row in test_df.iterrows():
        prob = row['P_model']
        odds = row['odds_empirical']
        edge = row['edge']
        result = row['Over25']

        # Filters
        if prob < prob_threshold:
            continue
        if edge < min_edge:
            continue
        if odds < min_odds or odds > max_odds:
            continue
        if pd.isna(result):
            continue

        # Stake
        if stake_mode == 'kelly':
            stake = kelly_criterion(prob, odds, kelly_fraction) * bankroll
        else:
            stake = flat_stake(bankroll, stake_pct)

        stake = max(5, min(stake, bankroll * 0.05))

        if stake > bankroll or bankroll < 100:
            continue

        # Result
        won = result == 1
        profit = stake * (odds - 1) if won else -stake
        bankroll += profit

        bets.append({
            'date': row['Date'],
            'match': f"{row.get('Home Team', '?')} vs {row.get('Away Team', '?')}",
            'prob': prob,
            'odds': odds,
            'edge': edge,
            'stake': stake,
            'won': won,
            'profit': profit,
            'bankroll': bankroll,
        })

    if not bets:
        print("\nNessuna scommessa con questi filtri!")
        return None

    bets_df = pd.DataFrame(bets)

    # Metrics
    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    win_rate = wins / total_bets
    total_profit = bets_df['profit'].sum()
    total_staked = bets_df['stake'].sum()
    roi = calculate_roi(win_rate, bets_df['odds'].mean())
    roi_real = (total_profit / total_staked) * 100

    bankroll_history = [initial_bankroll] + bets_df['bankroll'].tolist()
    max_dd = calculate_drawdown(bankroll_history)

    daily_profits = bets_df.groupby(bets_df['date'].dt.date)['profit'].sum()
    sharpe = sharpe_ratio(daily_profits)

    # Results
    print("\n" + "="*70)
    print("   RISULTATI")
    print("="*70)

    print(f"\nPerformance:")
    print(f"   Scommesse: {total_bets}")
    print(f"   Vinte: {wins} ({win_rate:.1%})")
    print(f"   Profit: EUR {total_profit:,.2f}")
    print(f"   ROI: {roi_real:.2f}%")
    print(f"   Bankroll: EUR {initial_bankroll:,} -> EUR {bankroll:,.2f} ({(bankroll/initial_bankroll-1)*100:+.1f}%)")

    print(f"\nRischio:")
    print(f"   Max Drawdown: {max_dd:.1%}")
    print(f"   Sharpe Ratio: {sharpe:.2f}")

    print(f"\nStatistiche:")
    print(f"   Prob media: {bets_df['prob'].mean():.1%}")
    print(f"   Quota media: {bets_df['odds'].mean():.2f}")
    print(f"   Edge medio: {bets_df['edge'].mean():.1%}")
    print(f"   Stake medio: EUR {bets_df['stake'].mean():.2f}")

    # Per odds bucket
    bets_df['odds_bucket'] = pd.cut(
        bets_df['odds'],
        bins=[1.0, 1.7, 2.0, 2.5, 3.0, 5.0],
        labels=['1.0-1.7', '1.7-2.0', '2.0-2.5', '2.5-3.0', '3.0+']
    )

    print(f"\nPer fascia quota:")
    for bucket in ['1.0-1.7', '1.7-2.0', '2.0-2.5', '2.5-3.0', '3.0+']:
        subset = bets_df[bets_df['odds_bucket'] == bucket]
        if len(subset) > 10:
            wr = subset['won'].mean()
            r = (subset['profit'].sum() / subset['stake'].sum()) * 100 if subset['stake'].sum() > 0 else 0
            print(f"   {bucket}: {len(subset)} bet, WR {wr:.0%}, ROI {r:+.1f}%")

    return {
        'bets_df': bets_df,
        'total_bets': total_bets,
        'win_rate': win_rate,
        'roi': roi_real,
        'profit': total_profit,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'final_bankroll': bankroll,
    }


if __name__ == "__main__":
    # Test con parametri conservativi
    print("\n>>> TEST CONSERVATIVO (P>55%, Edge>3%)")
    run_backtest(prob_threshold=0.55, min_edge=0.03)

    print("\n\n>>> TEST AGGRESSIVO (P>50%, Edge>5%)")
    run_backtest(prob_threshold=0.50, min_edge=0.05)
