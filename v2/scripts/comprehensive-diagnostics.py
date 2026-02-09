#!/usr/bin/env python3
"""
COMPREHENSIVE DIAGNOSTIC SUITE
Tests overfitting, calibration, and true edge detection
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

def load_backtest_results(results_path='d:\\BetAI\\v2\\backtest\\results.json'):
    """Load backtest results"""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Results not found: {results_path}")
        sys.exit(1)

def load_features(features_path='d:\\BetAI\\v2\\data\\processed\\features.json'):
    """Load features to get ground truth"""
    with open(features_path, 'r') as f:
        features = json.load(f)
    return pd.DataFrame(features)

def calculate_brier_score(y_true, y_prob):
    """Calculate Brier score (lower is better)"""
    return np.mean((y_prob - y_true) ** 2)

def diagnose_overfitting(results):
    """
    TEST 1: Overfitting Diagnostic
    Compare Brier on ENTIRE dataset vs value bets only
    """
    print("\n" + "="*70)
    print("TEST 1: OVERFITTING DETECTION")
    print("="*70)
    
    bets = results['allBets']
    
    # Get all bets
    all_predictions = []
    all_results = []
    all_clv = []
    
    for bet in bets:
        pred = bet.get('predictedProb', bet.get('pred_prob'))
        if pred is None:
            continue
        result = 1.0 if bet['result'] in ['H', 'D', 'A'] else 0.0
        clv = bet.get('clv', 0)
        
        all_predictions.append(pred)
        all_results.append(result)
        all_clv.append(clv)
    
    all_predictions = np.array(all_predictions)
    all_results = np.array(all_results)
    all_clv = np.array(all_clv)
    
    # Overall Brier
    overall_brier = calculate_brier_score(all_results, all_predictions)
    print(f"\n📊 Overall Dataset:")
    print(f"  Total bets: {len(bets)}")
    print(f"  Brier score: {overall_brier:.6f}")
    
    # Market Brier (approx from odds)
    market_predictions = []
    for bet in bets:
        market_prob = 1.0 / bet.get('marketOdds', 2.0)
        market_predictions.append(min(max(market_prob, 0.01), 0.99))
    market_brier = calculate_brier_score(all_results, np.array(market_predictions))
    print(f"  Market Brier: {market_brier:.6f}")
    print(f"  Model better?: {'✅ YES' if overall_brier < market_brier else '❌ NO'} ({overall_brier - market_brier:+.6f})")
    
    # Test different edge thresholds
    print(f"\n🎯 Value Bets Analysis (edge thresholds):")
    for min_edge in [0.03, 0.05, 0.08, 0.10, 0.12]:
        value_bets_idx = np.where(all_clv >= min_edge/100)[0]
        
        if len(value_bets_idx) == 0:
            continue
        
        vb_predictions = all_predictions[value_bets_idx]
        vb_results = all_results[value_bets_idx]
        vb_brier = calculate_brier_score(vb_results, vb_predictions)
        
        print(f"\n  Edge >= {min_edge*100:.1f}%:")
        print(f"    Count: {len(value_bets_idx)}/{len(bets)} ({100*len(value_bets_idx)/len(bets):.1f}%)")
        print(f"    Brier: {vb_brier:.6f}")
        print(f"    vs Market: {vb_brier - market_brier:+.6f} {'✅ Better' if vb_brier < market_brier else '❌ Worse'}")

def diagnose_segmentation(results, features_df):
    """
    TEST 2: Segmentation Analysis
    Where does the model work? (leagues, quote ranges, bet types)
    """
    print("\n" + "="*70)
    print("TEST 2: SEGMENTATION ANALYSIS")
    print("="*70)
    
    bets = results['allBets']
    
    # Match bets with features
    match_map = {(f['homeTeam'], f['awayTeam'], f['date']): f 
                 for f in features_df.to_dict('records')}
    
    segments = defaultdict(lambda: {'pred': [], 'result': [], 'clv': []})
    
    for bet in bets:
        key = (bet.get('homeTeam'), bet.get('awayTeam'), bet.get('date'))
        feature = match_map.get(key, {})
        league = feature.get('league', 'unknown')
        
        pred = bet.get('predictedProb', 0)
        result = 1.0 if bet['result'] == 'H' else 0.0
        clv = bet.get('clv', 0)
        market_odds = bet.get('marketOdds', 2.0)
        
        # Segment by league
        segments[f"league_{league}"]['pred'].append(pred)
        segments[f"league_{league}"]['result'].append(result)
        segments[f"league_{league}"]['clv'].append(clv)
        
        # Segment by odds range
        if market_odds < 2.0:
            odds_seg = "favorites"
        elif market_odds < 3.5:
            odds_seg = "mid"
        else:
            odds_seg = "longshots"
        
        segments[f"odds_{odds_seg}"]['pred'].append(pred)
        segments[f"odds_{odds_seg}"]['result'].append(result)
        segments[f"odds_{odds_seg}"]['clv'].append(clv)
    
    print(f"\n⚙️  Results by League:")
    for seg_name in sorted(segments.keys()):
        if 'league' not in seg_name:
            continue
        preds = np.array(segments[seg_name]['pred'])
        results = np.array(segments[seg_name]['result'])
        clv = np.array(segments[seg_name]['clv'])
        
        if len(preds) == 0:
            continue
        
        brier = calculate_brier_score(results, preds)
        print(f"\n  {seg_name}:")
        print(f"    Count: {len(preds)}")
        print(f"    Brier: {brier:.6f}")
        print(f"    Avg CLV: {np.mean(clv)*100:.2f}%")
        print(f"    Hit rate: {np.mean(results)*100:.1f}%")
    
    print(f"\n⚙️  Results by Odds Range:")
    for seg_name in sorted(segments.keys()):
        if 'odds' not in seg_name:
            continue
        preds = np.array(segments[seg_name]['pred'])
        results = np.array(segments[seg_name]['result'])
        
        brier = calculate_brier_score(results, preds)
        print(f"\n  {seg_name}:")
        print(f"    Count: {len(preds)}")
        print(f"    Brier: {brier:.6f}")

def diagnose_calibration(results):
    """
    TEST 3: Calibration Check
    Are predicted probabilities well-calibrated?
    """
    print("\n" + "="*70)
    print("TEST 3: CALIBRATION ANALYSIS")
    print("="*70)
    
    bets = results['allBets']
    
    predictions = []
    outcomes = []
    
    for bet in bets:
        pred = bet.get('predictedProb')
        if pred is None:
            continue
        
        result = 1.0 if bet['result'] == 'H' else 0.0
        predictions.append(pred)
        outcomes.append(result)
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    # Bin analysis
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    print(f"\n📈 Calibration Curve (predicted vs actual):")
    print(f"{'Pred Range':<15} {'Count':<8} {'Actual %':<12} {'Expected %':<12} {'Gap':<8}")
    print("-" * 60)
    
    for i in range(len(bins)-1):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if np.sum(mask) == 0:
            continue
        
        actual = np.mean(outcomes[mask])
        expected = (bins[i] + bins[i+1]) / 2
        gap = actual - expected
        
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}        {np.sum(mask):<8} {actual*100:<12.1f} {expected*100:<12.1f} {gap:+.2f}")
    
    # ECE (Expected Calibration Error)
    ece = 0
    for i in range(len(bins)-1):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if np.sum(mask) > 0:
            actual = np.mean(outcomes[mask])
            expected = (bins[i] + bins[i+1]) / 2
            weight = np.sum(mask) / len(predictions)
            ece += weight * abs(actual - expected)
    
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    if ece > 0.05:
        print("  ⚠️  HIGH - Model needs calibration (Platt scaling recommended)")
    elif ece > 0.02:
        print("  ⚠️  MODERATE - Consider calibration")
    else:
        print("  ✅ GOOD - Model is reasonably calibrated")

def run_all_diagnostics():
    """Run complete diagnostic suite"""
    print("\n" + "#"*70)
    print("# COMPREHENSIVE BACKTEST DIAGNOSTICS")
    print("# Testing: Overfitting | Calibration | Edge Reality")
    print("#"*70)
    
    results = load_backtest_results()
    features_df = load_features()
    
    diagnose_overfitting(results)
    diagnose_segmentation(results, features_df)
    diagnose_calibration(results)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
If Brier on HIGH EDGE BETS < Market Brier:
  ✅ Real edge exists → Proceed with live betting
  
If Brier on HIGH EDGE BETS > Market Brier:
  ❌ Model miscalibrated → Need:
     1. Calibration (Platt scaling)
     2. Data improvements (real xG)
     3. Feature engineering (injuries, lineups)
  
If Brier overall >> Market Brier but improves on high edge:
  ⚠️  Overfitting pattern - reduce bet count significantly
    """)

if __name__ == '__main__':
    run_all_diagnostics()
