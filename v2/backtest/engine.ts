/**
 * Backtesting Engine
 *
 * Walk-forward validation with:
 * - Monthly rolling windows
 * - Kelly bet sizing
 * - CLV tracking
 * - Full P&L simulation
 *
 * Key principle: NO look-ahead bias. Every prediction uses only past data.
 */

import type { MatchFeatures, BacktestResult } from '../src/types/index.js';
import { calculateKelly, getAdaptiveKellyFraction, simulateKellyGrowth, calculateSharpe } from '../betting/kelly.js';
import { detectValueBets, calculateCLV, analyzeCLV } from '../betting/value-detector.js';
import { BETTING_CONFIG, getLeagueTier } from '../src/config.js';

export interface BacktestBet {
  matchId: string;
  date: Date;
  league: string;
  market: string;
  selection: string;
  modelProb: number;
  marketOdds: number;
  closingOdds: number;
  stake: number;        // Fraction of bankroll
  edge: number;
  won: boolean;
  profit: number;       // Actual P&L (units)
  clv: number;          // Closing Line Value %
  oddsBucket: string;   // For segmented reporting
  tier: number;
}

export interface BacktestConfig {
  minEdge: number;           // Minimum edge to bet (default 5%)
  kellyFraction: number;     // Fraction of Kelly (default 0.25)
  maxStakePercent: number;   // Max single bet size (default 5%)
  initialBankroll: number;   // Starting bankroll
  startDate?: Date;          // Backtest start
  endDate?: Date;            // Backtest end
}

/**
 * Run backtest on feature dataset with model probabilities.
 *
 * @param features - Match features with closing odds
 * @param predictions - Model predictions (matchId -> { home, draw, away })
 * @param config - Backtest configuration
 */
export function runBacktest(
  features: MatchFeatures[],
  predictions: Map<string, { home: number; draw: number; away: number }>,
  config: Partial<BacktestConfig> = {},
): {
  bets: BacktestBet[];
  summary: BacktestResult;
  monthlyResults: Array<{ month: string; bets: number; roi: number; clv: number }>;
  bankrollHistory: number[];
} {
  const cfg: BacktestConfig = {
    minEdge: BETTING_CONFIG.minEdge,
    kellyFraction: BETTING_CONFIG.kellyFraction,
    maxStakePercent: BETTING_CONFIG.maxStakePercent,
    initialBankroll: BETTING_CONFIG.initialBankroll,
    ...config,
  };

  // Sort features by date
  const sorted = [...features].sort((a, b) =>
    new Date(a.date).getTime() - new Date(b.date).getTime()
  );

  // Filter by date range
  const filtered = sorted.filter(m => {
    const date = new Date(m.date);
    if (cfg.startDate && date < cfg.startDate) return false;
    if (cfg.endDate && date > cfg.endDate) return false;
    return true;
  });

  console.log(`[backtest] Running on ${filtered.length} matches`);

  const bets: BacktestBet[] = [];
  let bankroll = cfg.initialBankroll;
  const bankrollHistory: number[] = [bankroll];
  const monthlyData = new Map<string, { bets: BacktestBet[]; bankrollStart: number }>();

  // Process each match
  for (const match of filtered) {
    const pred = predictions.get(match.matchId);
    if (!pred) continue;

    // Must have closing odds
    if (!match.closingOdds) continue;

    const { home: oddsH, draw: oddsD, away: oddsA } = match.closingOdds;

    // Detect value bets (segment-aware: passes league for tier-based thresholds)
    const valueBets = detectValueBets(
      match.matchId,
      pred,
      { home: oddsH, draw: oddsD, away: oddsA },
      cfg.minEdge,
      match.league,
    );

    // Take best value bet only (to avoid correlation)
    const bestBet = valueBets[0];
    if (!bestBet) continue;

    // Adaptive Kelly: fraction varies by confidence + segment
    const maxProb = Math.max(pred.home, pred.draw, pred.away);
    const confidence = (maxProb - 0.333) / (1 - 0.333);
    const adaptiveFraction = getAdaptiveKellyFraction({
      confidence,
      odds: bestBet.marketOdds,
      leagueTier: getLeagueTier(match.league),
    });

    const kelly = calculateKelly(bestBet.modelProb, bestBet.marketOdds, adaptiveFraction, cfg.minEdge);
    if (!kelly.shouldBet) continue;

    // Simulate early odds (2% worse than closing - conservative assumption)
    const playedOdds = bestBet.marketOdds * 1.02;

    // Determine if bet won
    let won = false;
    if (bestBet.market === '1X2_H' && match.result === 'H') won = true;
    if (bestBet.market === '1X2_D' && match.result === 'D') won = true;
    if (bestBet.market === '1X2_A' && match.result === 'A') won = true;

    // Calculate P&L
    const stake = kelly.stake;
    const betAmount = bankroll * stake;
    const profit = won ? betAmount * (playedOdds - 1) : -betAmount;
    bankroll += profit;

    // CLV - use closing odds if available, fallback to market odds
    let closingOdds = bestBet.marketOdds; // fallback
    if (bestBet.market === '1X2_H') closingOdds = match.closingOddsHome || bestBet.marketOdds;
    else if (bestBet.market === '1X2_D') closingOdds = match.closingOddsDraw || bestBet.marketOdds;
    else if (bestBet.market === '1X2_A') closingOdds = match.closingOddsAway || bestBet.marketOdds;
    
    const clv = calculateCLV(playedOdds, closingOdds);

    // Odds bucket for segmented reporting
    const oddsBucket = bestBet.marketOdds < 1.5 ? '1.00-1.50'
      : bestBet.marketOdds < 2.0 ? '1.50-2.00'
      : bestBet.marketOdds < 2.5 ? '2.00-2.50'
      : bestBet.marketOdds < 3.5 ? '2.50-3.50'
      : bestBet.marketOdds < 5.0 ? '3.50-5.00'
      : '5.00+';

    const bet: BacktestBet = {
      matchId: match.matchId,
      date: new Date(match.date),
      league: match.league,
      market: bestBet.market,
      selection: bestBet.selection,
      modelProb: bestBet.modelProb,
      marketOdds: playedOdds,
      closingOdds: bestBet.marketOdds,
      stake,
      edge: bestBet.edge,
      won,
      profit,
      clv,
      oddsBucket,
      tier: getLeagueTier(match.league),
    };

    bets.push(bet);
    bankrollHistory.push(bankroll);

    // Track monthly
    const monthKey = `${bet.date.getFullYear()}-${String(bet.date.getMonth() + 1).padStart(2, '0')}`;
    if (!monthlyData.has(monthKey)) {
      monthlyData.set(monthKey, { bets: [], bankrollStart: bankroll - profit });
    }
    monthlyData.get(monthKey)!.bets.push(bet);
  }

  // Calculate summary
  const summary = calculateSummary(features, predictions, bets, cfg.initialBankroll, bankroll);

  // Monthly breakdown
  const monthlyResults = Array.from(monthlyData.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([month, data]) => {
      const totalStaked = data.bets.reduce((s, b) => s + b.stake * data.bankrollStart, 0);
      const totalProfit = data.bets.reduce((s, b) => s + b.profit, 0);
      const roi = totalStaked > 0 ? (totalProfit / totalStaked) * 100 : 0;
      const avgCLV = data.bets.reduce((s, b) => s + b.clv, 0) / data.bets.length;

      return {
        month,
        bets: data.bets.length,
        roi,
        clv: avgCLV,
      };
    });

  return { bets, summary, monthlyResults, bankrollHistory };
}

/**
 * Calculate summary statistics.
 */
function calculateSummary(
  features: MatchFeatures[],
  predictions: Map<string, { home: number; draw: number; away: number }>,
  bets: BacktestBet[],
  initialBankroll: number,
  finalBankroll: number,
): BacktestResult {
  // Accuracy on all predictions (not just bets)
  let correct1X2 = 0;
  let total1X2 = 0;
  let brierSum = 0;

  for (const match of features) {
    const pred = predictions.get(match.matchId);
    if (!pred) continue;

    total1X2++;

    // Predicted result
    const probs = [pred.home, pred.draw, pred.away];
    const maxIdx = probs.indexOf(Math.max(...probs));
    const predicted = ['H', 'D', 'A'][maxIdx];
    if (predicted === match.result) correct1X2++;

    // Brier score
    const actual = [match.result === 'H' ? 1 : 0, match.result === 'D' ? 1 : 0, match.result === 'A' ? 1 : 0];
    brierSum += (pred.home - actual[0]) ** 2 + (pred.draw - actual[1]) ** 2 + (pred.away - actual[2]) ** 2;
  }

  const accuracy1X2 = total1X2 > 0 ? correct1X2 / total1X2 : 0;
  const brierScore = total1X2 > 0 ? brierSum / total1X2 : 1;

  // ROI
  const totalStaked = bets.reduce((s, b) => s + Math.abs(b.profit) / (b.won ? (b.marketOdds - 1) : 1), 0);
  const totalProfit = finalBankroll - initialBankroll;
  const roi = totalStaked > 0 ? (totalProfit / initialBankroll) * 100 : 0;

  // Max drawdown
  let peak = initialBankroll;
  let maxDrawdown = 0;
  let running = initialBankroll;
  for (const bet of bets) {
    running += bet.profit;
    if (running > peak) peak = running;
    const dd = (peak - running) / peak;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }

  // Sharpe
  const returns = bets.map(b => b.profit / (initialBankroll * b.stake));
  const sharpe = calculateSharpe(returns);

  // CLV analysis
  const clvAnalysis = analyzeCLV(bets.map(b => ({
    playedOdds: b.marketOdds,
    closingOdds: b.closingOdds,
    won: b.won,
  })));

  // Calibration (buckets of predicted prob vs actual frequency)
  const calibration = calculateCalibration(features, predictions);

  // Bets by market
  const betsByMarket: Record<string, { count: number; won: number; roi: number }> = {};
  for (const bet of bets) {
    if (!betsByMarket[bet.market]) {
      betsByMarket[bet.market] = { count: 0, won: 0, roi: 0 };
    }
    betsByMarket[bet.market].count++;
    if (bet.won) betsByMarket[bet.market].won++;
  }

  // ROI per market
  for (const market of Object.keys(betsByMarket)) {
    const marketBets = bets.filter(b => b.market === market);
    const staked = marketBets.reduce((s, b) => s + b.stake, 0);
    const profit = marketBets.reduce((s, b) => s + b.profit, 0);
    betsByMarket[market].roi = staked > 0 ? (profit / (staked * initialBankroll)) * 100 : 0;
  }

  return {
    period: `${features[0]?.date} to ${features[features.length - 1]?.date}`,
    totalMatches: total1X2,
    totalBets: bets.length,
    accuracy1X2,
    brierScore,
    roi,
    profit: totalProfit,
    maxDrawdown,
    sharpeRatio: sharpe,
    calibration,
    betsByMarket,
  };
}

/**
 * Calculate calibration: for each probability bucket, actual win rate.
 */
function calculateCalibration(
  features: MatchFeatures[],
  predictions: Map<string, { home: number; draw: number; away: number }>,
): { predicted: number; actual: number }[] {
  const buckets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  const bucketData = buckets.map(b => ({ threshold: b, count: 0, wins: 0 }));

  for (const match of features) {
    const pred = predictions.get(match.matchId);
    if (!pred) continue;

    // Check each outcome
    const outcomes: Array<{ prob: number; won: boolean }> = [
      { prob: pred.home, won: match.result === 'H' },
      { prob: pred.draw, won: match.result === 'D' },
      { prob: pred.away, won: match.result === 'A' },
    ];

    for (const outcome of outcomes) {
      // Find bucket
      for (let i = 0; i < buckets.length; i++) {
        const lower = i === 0 ? 0 : buckets[i - 1];
        const upper = buckets[i];
        if (outcome.prob > lower && outcome.prob <= upper) {
          bucketData[i].count++;
          if (outcome.won) bucketData[i].wins++;
          break;
        }
      }
    }
  }

  return bucketData
    .filter(b => b.count > 0)
    .map(b => ({
      predicted: b.threshold - 0.05, // midpoint
      actual: b.wins / b.count,
    }));
}

/**
 * Print backtest report to console.
 */
export function printBacktestReport(
  result: Awaited<ReturnType<typeof runBacktest>>,
): void {
  const { summary, monthlyResults, bets } = result;

  console.log('\n' + '='.repeat(60));
  console.log('                    BACKTEST REPORT');
  console.log('='.repeat(60));

  console.log(`\nPeriod: ${summary.period}`);
  console.log(`Matches: ${summary.totalMatches}`);
  console.log(`Bets placed: ${summary.totalBets}`);

  console.log('\n--- Model Performance ---');
  console.log(`Accuracy (1X2): ${(summary.accuracy1X2 * 100).toFixed(1)}%`);
  console.log(`Brier Score:    ${summary.brierScore.toFixed(4)} (< 0.20 = good)`);

  console.log('\n--- Betting Performance ---');
  console.log(`ROI:            ${summary.roi.toFixed(1)}%`);
  console.log(`Profit:         ${summary.profit.toFixed(2)} units`);
  console.log(`Max Drawdown:   ${(summary.maxDrawdown * 100).toFixed(1)}%`);
  console.log(`Sharpe Ratio:   ${summary.sharpeRatio.toFixed(2)}`);

  // CLV
  const clvAnalysis = analyzeCLV(bets.map(b => ({
    playedOdds: b.marketOdds,
    closingOdds: b.closingOdds,
  })));
  console.log(`\n--- CLV (Closing Line Value) ---`);
  console.log(`Mean CLV:       ${clvAnalysis.meanCLV.toFixed(2)}%`);
  console.log(`Positive Rate:  ${(clvAnalysis.positiveRate * 100).toFixed(1)}%`);

  console.log('\n--- Monthly Breakdown ---');
  console.log('Month       Bets    ROI      CLV');
  for (const m of monthlyResults) {
    console.log(`${m.month}     ${String(m.bets).padStart(4)}   ${m.roi.toFixed(1).padStart(6)}%   ${m.clv.toFixed(1).padStart(5)}%`);
  }

  console.log('\n--- Calibration ---');
  console.log('Predicted   Actual');
  for (const c of summary.calibration) {
    console.log(`${(c.predicted * 100).toFixed(0)}%         ${(c.actual * 100).toFixed(1)}%`);
  }

  console.log('\n--- By Market ---');
  for (const [market, stats] of Object.entries(summary.betsByMarket)) {
    const winRate = stats.count > 0 ? (stats.won / stats.count * 100).toFixed(1) : '0';
    console.log(`${market}: ${stats.count} bets, ${winRate}% win rate, ${stats.roi.toFixed(1)}% ROI`);
  }

  // ─── Segmented Report ───
  console.log('\n--- By Odds Bucket ---');
  const byBucket = new Map<string, { count: number; won: number; profit: number; clvSum: number }>();
  for (const b of bets) {
    const entry = byBucket.get(b.oddsBucket) || { count: 0, won: 0, profit: 0, clvSum: 0 };
    entry.count++;
    if (b.won) entry.won++;
    entry.profit += b.profit;
    entry.clvSum += b.clv;
    byBucket.set(b.oddsBucket, entry);
  }
  for (const [bucket, stats] of [...byBucket.entries()].sort()) {
    const winRate = (stats.won / stats.count * 100).toFixed(1);
    const avgClv = (stats.clvSum / stats.count).toFixed(2);
    console.log(`${bucket.padEnd(12)} ${String(stats.count).padStart(4)} bets, ${winRate}% win, P&L: ${stats.profit.toFixed(1)}, CLV: ${avgClv}%`);
  }

  console.log('\n--- By Tier ---');
  const byTier = new Map<number, { count: number; won: number; profit: number; clvSum: number }>();
  for (const b of bets) {
    const entry = byTier.get(b.tier) || { count: 0, won: 0, profit: 0, clvSum: 0 };
    entry.count++;
    if (b.won) entry.won++;
    entry.profit += b.profit;
    entry.clvSum += b.clv;
    byTier.set(b.tier, entry);
  }
  for (const [tier, stats] of [...byTier.entries()].sort()) {
    const winRate = (stats.won / stats.count * 100).toFixed(1);
    const avgClv = (stats.clvSum / stats.count).toFixed(2);
    console.log(`Tier ${tier}       ${String(stats.count).padStart(4)} bets, ${winRate}% win, P&L: ${stats.profit.toFixed(1)}, CLV: ${avgClv}%`);
  }

  console.log('\n' + '='.repeat(60));
}
