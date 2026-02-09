/**
 * Kelly Criterion for Optimal Bet Sizing
 *
 * Full Kelly: f* = (p * b - q) / b
 * where:
 *   p = estimated probability of winning
 *   q = 1 - p (probability of losing)
 *   b = decimal odds - 1 (net profit per unit staked)
 *   f* = fraction of bankroll to bet
 *
 * Full Kelly is too volatile for real betting. We use Fractional Kelly
 * (typically 1/4 or 1/3) to reduce variance and protect against model errors.
 *
 * Reference: "Fortune's Formula" by William Poundstone
 */

import { BETTING_CONFIG } from '../src/config.js';
import type { LeagueTier } from '../src/config.js';

export interface KellyResult {
  fraction: number;      // Optimal fraction (full Kelly)
  adjustedFraction: number; // After applying fractional Kelly
  stake: number;         // Recommended stake (as % of bankroll)
  edge: number;          // p - impliedProb
  expectedValue: number; // (p * odds) - 1
  shouldBet: boolean;
}

/**
 * Adaptive Kelly multiplier based on confidence signals.
 *
 * Instead of a fixed 1/4 Kelly, the fraction scales with:
 * - Model confidence: how peaked is the probability distribution
 * - Segment reliability: historically profitable segments get higher fraction
 * - Edge magnitude: larger edges get proportionally more
 *
 * Range: [0.10, 0.40] of full Kelly (never below 1/10, never above 2/5)
 */
export function getAdaptiveKellyFraction(options: {
  confidence?: number;       // 0-1, how peaked the model distribution is
  odds?: number;             // decimal odds of the bet
  leagueTier?: LeagueTier;   // 1, 2, or 3
} = {}): number {
  const { confidence = 0.5, odds = 2.0, leagueTier = 1 } = options;

  let fraction = BETTING_CONFIG.kellyFraction; // baseline 0.25

  // Confidence adjustment: higher confidence → more aggressive
  // confidence 0.0 → ×0.6, confidence 0.5 → ×1.0, confidence 1.0 → ×1.4
  const confMultiplier = 0.6 + 0.8 * confidence;
  fraction *= confMultiplier;

  // Segment adjustment based on bias analysis findings:
  // - Favourites (1.00-1.50): strong positive bias → slightly more aggressive
  // - Tier 2 favourites: strongest bias → most aggressive
  // - Longshots (3.50+): negative bias → conservative
  if (odds >= 1.0 && odds < 1.5) {
    fraction *= leagueTier === 2 ? 1.3 : 1.15;
  } else if (odds >= 1.5 && odds < 2.0) {
    fraction *= 1.05;
  } else if (odds >= 3.5 && odds < 5.0) {
    fraction *= 0.8;
  } else if (odds >= 5.0) {
    fraction *= 0.6;
  }

  // Clamp to [0.10, 0.40]
  return Math.max(0.10, Math.min(0.40, fraction));
}

/**
 * Calculate Kelly stake for a single bet.
 *
 * @param modelProb - Model's estimated probability of winning
 * @param decimalOdds - Market decimal odds (e.g., 2.50)
 * @param kellyFraction - Fraction of Kelly to use (default 0.25 = quarter Kelly)
 * @param minEdge - Minimum edge required to bet (default 0.05 = 5%)
 */
export function calculateKelly(
  modelProb: number,
  decimalOdds: number,
  kellyFraction: number = BETTING_CONFIG.kellyFraction,
  minEdge: number = BETTING_CONFIG.minEdge,
): KellyResult {
  // Implied probability from market odds
  const impliedProb = 1 / decimalOdds;

  // Edge: how much we think we're ahead of the market
  const edge = modelProb - impliedProb;

  // Expected value: (prob * odds) - 1
  const expectedValue = (modelProb * decimalOdds) - 1;

  // No edge = no bet
  if (edge < minEdge || modelProb <= impliedProb) {
    return {
      fraction: 0,
      adjustedFraction: 0,
      stake: 0,
      edge,
      expectedValue,
      shouldBet: false,
    };
  }

  // Kelly formula: f* = (p * b - q) / b
  // where b = odds - 1, q = 1 - p
  const b = decimalOdds - 1;
  const q = 1 - modelProb;
  const fullKelly = (modelProb * b - q) / b;

  // Apply fractional Kelly
  const adjustedKelly = fullKelly * kellyFraction;

  // Cap at max stake
  const stake = Math.min(adjustedKelly, BETTING_CONFIG.maxStakePercent);

  return {
    fraction: Math.max(0, fullKelly),
    adjustedFraction: Math.max(0, adjustedKelly),
    stake: Math.max(0, stake),
    edge,
    expectedValue,
    shouldBet: stake > 0,
  };
}

/**
 * Calculate Kelly for multiple independent bets.
 * Sums the individual stakes (simple approach).
 *
 * For correlated bets (e.g., same match), use only the best value bet.
 */
export function calculateMultiKelly(
  bets: { modelProb: number; decimalOdds: number }[],
  kellyFraction: number = BETTING_CONFIG.kellyFraction,
): KellyResult[] {
  return bets.map(bet => calculateKelly(bet.modelProb, bet.decimalOdds, kellyFraction));
}

/**
 * Simulate Kelly growth over a series of bets.
 * Useful for backtesting.
 *
 * @param bets - Array of { won: boolean, stake: number (fraction), odds: number }
 * @param initialBankroll - Starting bankroll
 */
export function simulateKellyGrowth(
  bets: { won: boolean; stake: number; odds: number }[],
  initialBankroll: number = BETTING_CONFIG.initialBankroll,
): {
  finalBankroll: number;
  maxDrawdown: number;
  history: number[];
} {
  let bankroll = initialBankroll;
  let peak = initialBankroll;
  let maxDrawdown = 0;
  const history: number[] = [initialBankroll];

  for (const bet of bets) {
    const betAmount = bankroll * bet.stake;

    if (bet.won) {
      bankroll += betAmount * (bet.odds - 1);
    } else {
      bankroll -= betAmount;
    }

    history.push(bankroll);

    // Track drawdown
    if (bankroll > peak) {
      peak = bankroll;
    }
    const drawdown = (peak - bankroll) / peak;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  return {
    finalBankroll: bankroll,
    maxDrawdown,
    history,
  };
}

/**
 * Calculate Sharpe Ratio from bet returns.
 * Sharpe = mean(returns) / std(returns) * sqrt(n_bets_per_year)
 */
export function calculateSharpe(returns: number[], betsPerYear: number = 500): number {
  if (returns.length < 2) return 0;

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / (returns.length - 1);
  const std = Math.sqrt(variance);

  if (std === 0) return 0;
  return (mean / std) * Math.sqrt(betsPerYear);
}
