/**
 * Value Bet Detection
 *
 * A value bet exists when:
 *   model_probability > implied_probability (from odds)
 *
 * We express this as:
 *   edge = model_prob - implied_prob
 *   value = (model_prob * odds) - 1
 *
 * Only bet when edge exceeds threshold (default 5%).
 *
 * CLV (Closing Line Value) is the definitive validation metric:
 * If your played odds are consistently better than closing odds,
 * you're beating the market and profits will follow.
 */

import type { Prediction, ValueBet, MatchFeatures, LeagueCode } from '../src/types/index.js';
import { calculateKelly } from './kelly.js';
import { BETTING_CONFIG, getLeagueTier } from '../src/config.js';
import type { LeagueTier } from '../src/config.js';

export interface MarketOdds {
  home: number;
  draw: number;
  away: number;
  over25?: number;
  under25?: number;
  bttsYes?: number;
  bttsNo?: number;
}

// ─── Segment-aware edge thresholds ───
// Derived from bias analysis: lower threshold where market is less efficient,
// higher threshold (or block) where market is efficient or we have negative bias.

export type BetType = 'H' | 'D' | 'A';

export interface SegmentConfig {
  minEdge: number;
  blocked: boolean;  // true = never bet this segment (strongly negative bias)
}

/**
 * Get segment-aware edge threshold based on odds range, bet type, and league tier.
 *
 * Based on bias analysis findings:
 * - Favourites (1.00-1.50): market underestimates, especially Tier 2 → lower threshold
 * - Longshots (5.00+) on Draw: strongly overestimated → block
 * - Mid-range (2.00-3.50): efficient market → standard threshold
 * - Away bets at 3.50-5.00: overestimated → higher threshold
 */
export function getSegmentConfig(
  odds: number,
  betType: BetType,
  leagueTier?: LeagueTier,
): SegmentConfig {
  const tier = leagueTier ?? 1;

  // Block: longshot draws (5.00+) → -14% to -22% yield historically
  if (odds >= 5.0 && betType === 'D') {
    return { minEdge: 999, blocked: true };
  }

  // Block: longshot away Tier 2 (3.50-5.00) → -9.5% yield
  if (odds >= 3.5 && odds < 5.0 && betType === 'A' && tier === 2) {
    return { minEdge: 999, blocked: true };
  }

  // Strong favourites (1.00-1.50): market underestimates by +2.7-4.5%
  if (odds >= 1.0 && odds < 1.5) {
    if (tier === 2) return { minEdge: 0.02, blocked: false };  // T2 has +4.5% bias → aggressive
    return { minEdge: 0.03, blocked: false };                   // T1/T3 has +1.7-2.9% → moderate
  }

  // Moderate favourites (1.50-2.00): slight underestimation +0.7%
  if (odds >= 1.5 && odds < 2.0) {
    return { minEdge: 0.04, blocked: false };
  }

  // Mid-range (2.00-2.50): efficient, standard threshold
  if (odds >= 2.0 && odds < 2.5) {
    return { minEdge: 0.05, blocked: false };
  }

  // Mid-range (2.50-3.50): efficient for H/A, slight draw underestimation
  if (odds >= 2.5 && odds < 3.5) {
    if (betType === 'D' && tier === 2) return { minEdge: 0.04, blocked: false }; // +0.9% bias on T2 draws
    return { minEdge: 0.05, blocked: false };
  }

  // Longshots (3.50-5.00): generally overestimated
  if (odds >= 3.5 && odds < 5.0) {
    return { minEdge: 0.06, blocked: false };
  }

  // Deep longshots (5.00+, non-draw): strongly overestimated
  return { minEdge: 0.07, blocked: false };
}

/**
 * Detect value bets for a match given model probabilities and market odds.
 * Now segment-aware: uses dynamic edge thresholds per odds range / bet type / tier.
 */
export function detectValueBets(
  matchId: string,
  modelProbs: { home: number; draw: number; away: number },
  marketOdds: MarketOdds,
  minEdge: number = BETTING_CONFIG.minEdge,
  league?: LeagueCode,
): ValueBet[] {
  const valueBets: ValueBet[] = [];
  const tier = league ? getLeagueTier(league) : undefined;

  // 1X2 Markets
  const markets1X2: Array<{
    market: string;
    selection: string;
    modelProb: number;
    odds: number;
    betType: BetType;
  }> = [
    { market: '1X2_H', selection: 'Home Win', modelProb: modelProbs.home, odds: marketOdds.home, betType: 'H' },
    { market: '1X2_D', selection: 'Draw', modelProb: modelProbs.draw, odds: marketOdds.draw, betType: 'D' },
    { market: '1X2_A', selection: 'Away Win', modelProb: modelProbs.away, odds: marketOdds.away, betType: 'A' },
  ];

  for (const m of markets1X2) {
    if (!m.odds || m.odds <= 1) continue;

    // Get segment-specific threshold
    const segConfig = getSegmentConfig(m.odds, m.betType, tier);
    if (segConfig.blocked) continue;

    const effectiveMinEdge = Math.max(segConfig.minEdge, minEdge);

    const impliedProb = 1 / m.odds;
    const edge = m.modelProb - impliedProb;
    const value = (m.modelProb * m.odds) - 1;

    if (edge >= effectiveMinEdge) {
      const kelly = calculateKelly(m.modelProb, m.odds);
      valueBets.push({
        market: m.market,
        selection: m.selection,
        modelProb: m.modelProb,
        marketOdds: m.odds,
        impliedProb,
        edge,
        value,
        kellyFraction: kelly.fraction,
        recommendedStake: kelly.stake,
      });
    }
  }

  // Sort by value (descending)
  valueBets.sort((a, b) => b.value - a.value);

  return valueBets;
}

/**
 * Build full prediction with value bet analysis.
 */
export function buildPrediction(
  match: MatchFeatures,
  modelProbs: { home: number; draw: number; away: number },
  marketOdds?: MarketOdds,
): Prediction {
  const odds = marketOdds || (match.closingOdds ? {
    home: match.closingOdds.home,
    draw: match.closingOdds.draw,
    away: match.closingOdds.away,
  } : { home: 0, draw: 0, away: 0 });

  const valueBets = odds.home > 1 ? detectValueBets(
    match.matchId,
    modelProbs,
    odds,
  ) : [];

  // Confidence: how peaked is the distribution
  const maxProb = Math.max(modelProbs.home, modelProbs.draw, modelProbs.away);
  const confidence = (maxProb - 0.333) / (1 - 0.333);

  return {
    matchId: match.matchId,
    homeTeam: match.homeTeam,
    awayTeam: match.awayTeam,
    league: match.league,
    date: match.date,
    probabilities: modelProbs,
    confidence,
    valueBets,
  };
}

/**
 * Calculate CLV (Closing Line Value) for a bet.
 *
 * CLV = (played_odds / closing_odds - 1) * 100
 *
 * Positive CLV means you got better odds than the market closed at,
 * which is the strongest indicator of long-term profitability.
 */
export function calculateCLV(
  playedOdds: number,
  closingOdds: number,
): number {
  if (closingOdds <= 1) return 0;
  return ((playedOdds / closingOdds) - 1) * 100;
}

/**
 * Analyze CLV across a portfolio of bets.
 */
export function analyzeCLV(
  bets: Array<{ playedOdds: number; closingOdds: number; won?: boolean }>,
): {
  meanCLV: number;
  positiveRate: number;
  count: number;
  distribution: { bucket: string; count: number }[];
} {
  if (bets.length === 0) {
    return { meanCLV: 0, positiveRate: 0, count: 0, distribution: [] };
  }

  const clvValues = bets.map(b => calculateCLV(b.playedOdds, b.closingOdds));
  const meanCLV = clvValues.reduce((a, b) => a + b, 0) / clvValues.length;
  const positiveRate = clvValues.filter(v => v > 0).length / clvValues.length;

  // Distribution buckets
  const buckets = [
    { label: '< -5%', min: -Infinity, max: -5 },
    { label: '-5% to 0%', min: -5, max: 0 },
    { label: '0% to 5%', min: 0, max: 5 },
    { label: '> 5%', min: 5, max: Infinity },
  ];

  const distribution = buckets.map(bucket => ({
    bucket: bucket.label,
    count: clvValues.filter(v => v > bucket.min && v <= bucket.max).length,
  }));

  return {
    meanCLV,
    positiveRate,
    count: bets.length,
    distribution,
  };
}

/**
 * Filter bets by minimum edge and confidence.
 */
export function filterValueBets(
  valueBets: ValueBet[],
  options: {
    minEdge?: number;
    minValue?: number;
    minConfidence?: number;
    maxBetsPerMatch?: number;
  } = {},
): ValueBet[] {
  const { minEdge = 0.05, minValue = 0, minConfidence = 0, maxBetsPerMatch = 1 } = options;

  let filtered = valueBets.filter(
    bet => bet.edge >= minEdge && bet.value >= minValue
  );

  // Keep only top N per match
  filtered = filtered.slice(0, maxBetsPerMatch);

  return filtered;
}
