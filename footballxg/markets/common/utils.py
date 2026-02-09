"""
Utility comuni per tutti i mercati betting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"


def load_dataset() -> pd.DataFrame:
    """Carica il dataset principale."""
    return pd.read_parquet(DATA_DIR / "xg_full_dataset.parquet")


def temporal_split(df: pd.DataFrame, test_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporale per evitare data leakage.

    Args:
        df: DataFrame con colonna 'Date'
        test_months: Mesi da usare come test set

    Returns:
        (train_df, test_df)
    """
    df = df.sort_values('Date')
    cutoff = df['Date'].max() - pd.DateOffset(months=test_months)

    train = df[df['Date'] < cutoff].copy()
    test = df[df['Date'] >= cutoff].copy()

    return train, test


def calculate_roi(win_rate: float, odds: float) -> float:
    """
    Calcola ROI teorico.

    ROI = (win_rate * odds - 1) * 100
    """
    return (win_rate * odds - 1) * 100


def calculate_edge(prob: float, odds: float) -> float:
    """
    Calcola edge (expected value).

    Edge = prob * odds - 1
    Se Edge > 0, abbiamo valore.
    """
    return prob * odds - 1


def break_even_winrate(odds: float) -> float:
    """
    Calcola win rate necessario per break-even.

    BE = 1 / odds
    """
    return 1 / odds


def kelly_criterion(prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Kelly Criterion per stake sizing.

    Args:
        prob: Probabilita stimata di vincere
        odds: Quota decimale
        fraction: Frazione di Kelly (0.25 = quarter Kelly)

    Returns:
        Percentuale del bankroll da scommettere
    """
    b = odds - 1
    q = 1 - prob

    kelly = (prob * b - q) / b

    if kelly <= 0:
        return 0

    return kelly * fraction


def flat_stake(bankroll: float, pct: float = 0.02) -> float:
    """Stake fisso come percentuale del bankroll."""
    return bankroll * pct


def calculate_drawdown(bankroll_history: list) -> float:
    """Calcola max drawdown da storia bankroll."""
    peak = bankroll_history[0]
    max_dd = 0

    for b in bankroll_history:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def sharpe_ratio(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calcola Sharpe Ratio.

    Args:
        returns: Serie di rendimenti giornalieri
        annualize: Se True, annualizza il ratio

    Returns:
        Sharpe Ratio
    """
    if returns.std() == 0:
        return 0

    ratio = returns.mean() / returns.std()

    if annualize:
        ratio *= np.sqrt(252)

    return ratio


def empirical_probability(df: pd.DataFrame, feature: str, target: str,
                          value: float, window: float = 0.3, min_samples: int = 50) -> float:
    """
    Calcola probabilita empirica da dati storici.

    Args:
        df: DataFrame con dati storici
        feature: Nome colonna feature
        target: Nome colonna target
        value: Valore della feature per cui calcolare prob
        window: Finestra +/- intorno al valore
        min_samples: Minimo campioni richiesti

    Returns:
        Probabilita empirica
    """
    mask = (df[feature] >= value - window) & (df[feature] < value + window)

    if mask.sum() < min_samples:
        return 0.5

    return df[mask][target].mean()
