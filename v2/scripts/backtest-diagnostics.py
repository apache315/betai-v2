#!/usr/bin/env python3
"""
Advanced backtest diagnostics.
Tests: edge calibration, Brier on value bets only, segmentation, closing line test.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binom

def load_backtest_results(backtest_log_path):
    """Load backtest results."""
    with open(backtest_log_path, 'r') as f:
        return json.load(f)

def analyze_value_bets_only(results, min_edge=0.08):
    """
    Analyze Brier ONLY on value bets (edge >= min_edge).
    This is the real test.
    """
    print(f"\n{'='*60}")
    print(f"VALUE BETS ANALYSIS (edge >= {min_edge*100}%)")
    print(f"{'='*60}")
    
    value_bets = [b for b in results['all_bets'] if b['edge'] >= min_edge]
    
    if not value_bets:
        print(f"❌ No value bets found with edge >= {min_edge*100}%")
        return
    
    print(f"\nTotal bets: {len(results['all_bets'])}")
    print(f"Value bets (edge >= {min_edge*100}%): {len(value_bets)} ({len(value_bets)/len(results['all_bets'])*100:.1f}%)")
    
    # Brier on value bets only
    won = sum(1 for b in value_bets if b['won'])
    brier_value = np.mean([
        (b['probability'] - (1 if b['won'] else 0))**2 
        for b in value_bets
    ])
    
    # Market Brier on same bets
    market_brier = np.mean([
        (1/b['marketOdds'] - (1 if b['won'] else 0))**2 
        for b in value_bets
    ])
    
    # Accuracy
    accuracy = won / len(value_bets)
    
    print(f"\nWon: {won}/{len(value_bets)} ({accuracy*100:.1f}%)")
    print(f"Model Brier (value bets): {brier_value:.4f}")
    print(f"Market Brier (same bets): {market_brier:.4f}")
    print(f"Improvement: {(market_brier - brier_value):.4f} {'✅' if market_brier > brier_value else '❌'}")
    
    # Expected wins under null hypothesis (50%)
    expected_wins = len(value_bets) * 0.5
    wins_std = np.sqrt(len(value_bets) * 0.5 * 0.5)
    z_score = (won - expected_wins) / wins_std
    
    print(f"\nNull hypothesis (50% hit rate):")
    print(f"  Expected wins: {expected_wins:.0f}")
    print(f"  Actual wins: {won}")
    print(f"  Z-score: {z_score:.2f} {'✅ Significant' if abs(z_score) > 2 else '❌ Not significant'}")
    
    # CLV on value bets
    clv_mean = np.mean([b['clv'] for b in value_bets])
    clv_positive = sum(1 for b in value_bets if b['clv'] > 0)
    
    print(f"\nCLV on value bets:")
    print(f"  Mean CLV: {clv_mean:.4f}%")
    print(f"  Positive: {clv_positive}/{len(value_bets)} ({clv_positive/len(value_bets)*100:.1f}%)")

def segment_analysis(results):
    """Segment analysis by league, quote range, bet type."""
    print(f"\n{'='*60}")
    print(f"SEGMENTATION ANALYSIS")
    print(f"{'='*60}")
    
    bets = results['all_bets']
    
    # By league
    print(f"\nBy League:")
    for league in set(b['league'] for b in bets):
        league_bets = [b for b in bets if b['league'] == league]
        won = sum(1 for b in league_bets if b['won'])
        clv = np.mean([b['clv'] for b in league_bets])
        print(f"  {league:6s}: {len(league_bets):5d} bets, {won/len(league_bets)*100:5.1f}% hit, CLV {clv:+.2f}%")
    
    # By quote range
    print(f"\nBy Quote Range:")
    quote_ranges = [
        (1.0, 1.5, "1.00-1.50"),
        (1.5, 2.0, "1.50-2.00"),
        (2.0, 3.0, "2.00-3.00"),
        (3.0, 5.0, "3.00-5.00"),
        (5.0, 100, "5.00+"),
    ]
    for min_q, max_q, label in quote_ranges:
        range_bets = [b for b in bets if min_q <= b['marketOdds'] < max_q]
        if range_bets:
            won = sum(1 for b in range_bets if b['won'])
            clv = np.mean([b['clv'] for b in range_bets])
            edge = np.mean([b['edge'] for b in range_bets])
            print(f"  {label:10s}: {len(range_bets):5d} bets, {won/len(range_bets)*100:5.1f}% hit, edge {edge:+.2f}%, CLV {clv:+.2f}%")
    
    # By edge tier
    print(f"\nBy Edge Tier:")
    edge_ranges = [
        (0.0, 0.02, "0-2%"),
        (0.02, 0.05, "2-5%"),
        (0.05, 0.10, "5-10%"),
        (0.10, 1.0, "10%+"),
    ]
    for min_e, max_e, label in edge_ranges:
        edge_bets = [b for b in bets if min_e <= b['edge'] < max_e]
        if edge_bets:
            won = sum(1 for b in edge_bets if b['won'])
            clv = np.mean([b['clv'] for b in edge_bets])
            roi = np.mean([b['profit']/b['stake']*100 if b['stake'] > 0 else 0 for b in edge_bets])
            print(f"  {label:6s}: {len(edge_bets):5d} bets, {won/len(edge_bets)*100:5.1f}% hit, CLV {clv:+.2f}%, ROI {roi:+.1f}%")

def closing_line_test(results):
    """
    Test: does model beat closing line?
    If model uses market odds as feature, this is unfair comparison.
    But if it doesn't use market data, this tests real generalization.
    """
    print(f"\n{'='*60}")
    print(f"CLOSING LINE TEST")
    print(f"{'='*60}")
    
    print(f"\n⚠️  IMPORTANT: This test is valid ONLY if:")
    print(f"    - Closing odds are NOT used as features")
    print(f"    - Model sees only historical/team data")
    print(f"\n❓ Check: are market odds in feature list?")
    print(f"    If YES → this test is invalid (data leak)")
    
    bets = results['all_bets']
    
    # CLV vs closing
    wins = sum(1 for b in bets if b['won'])
    roi_vs_closing = np.mean([
        (b['won'] * (b.get('closingOdds', b['marketOdds']) - 1) - (1 if not b['won'] else 0))
        / (1 if b.get('closingOdds', b['marketOdds']) > 1 else 1)
        for b in bets
    ])
    
    print(f"\nResults vs closing odds:")
    print(f"  Total bets: {len(bets)}")
    print(f"  Wins: {wins}/{len(bets)} ({wins/len(bets)*100:.1f}%)")
    print(f"  ROI vs closing: {roi_vs_closing:.2f}%")

def main(backtest_file='d:\\BetAI\\v2\\backtest\\results\\latest.json'):
    """Main diagnostics."""
    
    if not Path(backtest_file).exists():
        print(f"❌ Backtest file not found: {backtest_file}")
        return
    
    print("\n🔍 ADVANCED BACKTEST DIAGNOSTICS")
    print("="*60)
    
    results = load_backtest_results(backtest_file)
    
    # 1. Value bets analysis (CRITICAL)
    analyze_value_bets_only(results, min_edge=0.08)
    analyze_value_bets_only(results, min_edge=0.10)
    
    # 2. Segmentation
    segment_analysis(results)
    
    # 3. Closing line
    closing_line_test(results)
    
    print(f"\n{'='*60}")
    print(f"CONCLUSIONS")
    print(f"{'='*60}")
    print(f"""
IF value bets (edge >= 8%) BEAT market Brier:
  → Modello ha edge reale, vale la pena sviluppare

IF value bets (edge >= 8%) LOSE vs market Brier:
  → Modello non calibrato, non pronto per live

IF ROI vs closing > +2%:
  → Potenziale edge, ma testare prima live

IF ROI vs closing < +1%:
  → Probabilmente rumore, non rischiare denaro vero
    """)

if __name__ == '__main__':
    # Assume latest backtest results saved
    main()
