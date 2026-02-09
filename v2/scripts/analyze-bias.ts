#!/usr/bin/env tsx
/**
 * Market Bias Analysis
 *
 * Analyzes historical odds vs actual outcomes to find systematic market biases.
 * Segments by: odds range, bet type (H/D/A), league tier, time period.
 *
 * Key concepts:
 * - Implied probability = 1 / decimal_odds
 * - Overround = sum of implied probs - 1 (bookmaker margin)
 * - Fair probability = implied_prob / (1 + overround)  [vig-removed]
 * - Bias = actual_win_rate - fair_probability
 *   positive = market underestimates (potential value)
 *   negative = market overestimates
 *
 * Output: JSON bias map + console report
 */

import { loadAllMatches } from '../data/scrapers/football-data.js';
import { LEAGUES, getLeagueTier } from '../src/config.js';
import { writeFile, mkdir } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { LeagueCode } from '../src/types/index.js';
import type { Match } from '../src/types/index.js';

const __dirname = join(fileURLToPath(import.meta.url), '..');
const OUTPUT_DIR = join(__dirname, '..', 'data', 'processed');

// ─── Configuration ───

const ODDS_BUCKETS = [
  { label: '1.00-1.50', min: 1.00, max: 1.50 },
  { label: '1.50-2.00', min: 1.50, max: 2.00 },
  { label: '2.00-2.50', min: 2.00, max: 2.50 },
  { label: '2.50-3.50', min: 2.50, max: 3.50 },
  { label: '3.50-5.00', min: 3.50, max: 5.00 },
  { label: '5.00+',     min: 5.00, max: 999  },
];

const TIME_PERIODS = [
  { label: '2010-2014', minYear: 2010, maxYear: 2014 },
  { label: '2015-2019', minYear: 2015, maxYear: 2019 },
  { label: '2020-2025', minYear: 2020, maxYear: 2025 },
];

const BET_TYPES = ['H', 'D', 'A'] as const;
type BetType = typeof BET_TYPES[number];

// ─── Types ───

interface BiasResult {
  segment: string;
  n: number;
  actualRate: number;
  impliedProb: number;
  fairProb: number;
  rawBias: number;        // actual - implied (includes overround)
  fairBias: number;       // actual - fair (vig-removed, the real signal)
  ci95Lower: number;
  ci95Upper: number;
  significant: boolean;   // CI doesn't cross 0
  pValue: number;
  avgOdds: number;
  avgOverround: number;
  // Yield if flat-staking this segment
  flatYield: number;
}

interface SegmentedBias {
  byOddsBucket: Record<string, Record<BetType, BiasResult>>;
  byTier: Record<string, Record<BetType, BiasResult>>;
  byPeriod: Record<string, Record<BetType, BiasResult>>;
  byTierAndBucket: Record<string, Record<BetType, BiasResult>>;
  byPeriodAndBucket: Record<string, Record<BetType, BiasResult>>;
  exploitable: BiasResult[];
}

// ─── Main ───

async function main() {
  console.log('=== BetAI v2 - Market Bias Analysis ===\n');

  const allLeagues = Object.keys(LEAGUES) as LeagueCode[];
  const matches = await loadAllMatches(allLeagues);
  const withOdds = matches.filter(m => m.oddsHome && m.oddsDraw && m.oddsAway);

  console.log(`\nTotal: ${matches.length} matches, ${withOdds.length} with full odds\n`);

  const results: SegmentedBias = {
    byOddsBucket: {},
    byTier: {},
    byPeriod: {},
    byTierAndBucket: {},
    byPeriodAndBucket: {},
    exploitable: [],
  };

  // ─── 1. By odds bucket ───
  console.log('═══════════════════════════════════════════════════════════════════');
  console.log('1. BIAS BY ODDS RANGE');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  for (const bucket of ODDS_BUCKETS) {
    results.byOddsBucket[bucket.label] = {} as Record<BetType, BiasResult>;
    for (const bt of BET_TYPES) {
      const filtered = filterByOddsBucket(withOdds, bt, bucket.min, bucket.max);
      const bias = computeBias(filtered, bt, `${bucket.label}/${bt}`);
      results.byOddsBucket[bucket.label][bt] = bias;
    }
  }
  printBiasTable('Odds Range', results.byOddsBucket);

  // ─── 2. By league tier ───
  console.log('\n═══════════════════════════════════════════════════════════════════');
  console.log('2. BIAS BY LEAGUE TIER');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  for (const tier of [1, 2, 3]) {
    const label = `Tier ${tier}`;
    const tierMatches = withOdds.filter(m => getLeagueTier(m.league) === tier);
    results.byTier[label] = {} as Record<BetType, BiasResult>;
    for (const bt of BET_TYPES) {
      results.byTier[label][bt] = computeBias(tierMatches, bt, `${label}/${bt}`);
    }
  }
  printBiasTable('Tier', results.byTier);

  // ─── 3. By time period ───
  console.log('\n═══════════════════════════════════════════════════════════════════');
  console.log('3. BIAS BY TIME PERIOD (stability check)');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  for (const period of TIME_PERIODS) {
    const label = period.label;
    const periodMatches = withOdds.filter(m => {
      const y = m.date.getFullYear();
      return y >= period.minYear && y <= period.maxYear;
    });
    results.byPeriod[label] = {} as Record<BetType, BiasResult>;
    for (const bt of BET_TYPES) {
      results.byPeriod[label][bt] = computeBias(periodMatches, bt, `${label}/${bt}`);
    }
  }
  printBiasTable('Period', results.byPeriod);

  // ─── 4. Tier × Odds bucket (2D) ───
  console.log('\n═══════════════════════════════════════════════════════════════════');
  console.log('4. BIAS BY TIER × ODDS RANGE (2D segmentation)');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  for (const tier of [1, 2, 3]) {
    const tierMatches = withOdds.filter(m => getLeagueTier(m.league) === tier);
    for (const bucket of ODDS_BUCKETS) {
      const label = `T${tier}/${bucket.label}`;
      results.byTierAndBucket[label] = {} as Record<BetType, BiasResult>;
      for (const bt of BET_TYPES) {
        const filtered = filterByOddsBucket(tierMatches, bt, bucket.min, bucket.max);
        results.byTierAndBucket[label][bt] = computeBias(filtered, bt, `${label}/${bt}`);
      }
    }
  }
  printBiasTable('Tier/Odds', results.byTierAndBucket);

  // ─── 5. Period × Odds bucket (stability of bias over time) ───
  console.log('\n═══════════════════════════════════════════════════════════════════');
  console.log('5. BIAS STABILITY: PERIOD × ODDS RANGE');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  for (const period of TIME_PERIODS) {
    const periodMatches = withOdds.filter(m => {
      const y = m.date.getFullYear();
      return y >= period.minYear && y <= period.maxYear;
    });
    for (const bucket of ODDS_BUCKETS) {
      const label = `${period.label}/${bucket.label}`;
      results.byPeriodAndBucket[label] = {} as Record<BetType, BiasResult>;
      for (const bt of BET_TYPES) {
        const filtered = filterByOddsBucket(periodMatches, bt, bucket.min, bucket.max);
        results.byPeriodAndBucket[label][bt] = computeBias(filtered, bt, `${label}/${bt}`);
      }
    }
  }
  printBiasTable('Period/Odds', results.byPeriodAndBucket);

  // ─── 6. Find exploitable biases ───
  console.log('\n═══════════════════════════════════════════════════════════════════');
  console.log('6. EXPLOITABLE BIASES (significant, positive fair bias, yield > 0)');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  const allBiases: BiasResult[] = [];
  for (const section of [results.byOddsBucket, results.byTier, results.byPeriod, results.byTierAndBucket, results.byPeriodAndBucket]) {
    for (const [_, btMap] of Object.entries(section)) {
      for (const bt of BET_TYPES) {
        if (btMap[bt]) allBiases.push(btMap[bt]);
      }
    }
  }

  const exploitable = allBiases.filter(b =>
    b.significant && b.fairBias > 0 && b.flatYield > 0 && b.n >= 500
  ).sort((a, b) => b.flatYield - a.flatYield);

  results.exploitable = exploitable;

  if (exploitable.length === 0) {
    console.log('  No statistically significant exploitable biases found with n >= 500.\n');
    // Show near-misses
    const nearMiss = allBiases.filter(b =>
      b.fairBias > 0 && b.flatYield > -2 && b.n >= 300
    ).sort((a, b) => b.flatYield - a.flatYield).slice(0, 10);

    if (nearMiss.length > 0) {
      console.log('  Near-misses (positive fair bias, yield > -2%, n >= 300):\n');
      printExploitableTable(nearMiss);
    }
  } else {
    printExploitableTable(exploitable);
  }

  // ─── 7. Favourite-Longshot Bias summary ───
  console.log('\n═══════════════════════════════════════════════════════════════════');
  console.log('7. FAVOURITE-LONGSHOT BIAS (classic pattern)');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  console.log('  For each odds bucket, comparing fair probability vs actual outcome rate.');
  console.log('  Positive bias = market underestimates → value on longshots');
  console.log('  Negative bias = market overestimates → value on favourites (lay)\n');

  // Aggregate across all bet types
  for (const bucket of ODDS_BUCKETS) {
    const allInBucket: { odds: number; won: boolean; overround: number }[] = [];

    for (const m of withOdds) {
      const over = 1/m.oddsHome! + 1/m.oddsDraw! + 1/m.oddsAway! - 1;
      for (const bt of BET_TYPES) {
        const odds = bt === 'H' ? m.oddsHome! : bt === 'D' ? m.oddsDraw! : m.oddsAway!;
        if (odds >= bucket.min && odds < bucket.max) {
          allInBucket.push({ odds, won: m.ftResult === bt, overround: over });
        }
      }
    }

    if (allInBucket.length < 100) continue;
    const winRate = allInBucket.filter(x => x.won).length / allInBucket.length;
    const avgImplied = allInBucket.reduce((s, x) => s + 1/x.odds, 0) / allInBucket.length;
    const avgOver = allInBucket.reduce((s, x) => s + x.overround, 0) / allInBucket.length;
    const fairProb = avgImplied / (1 + avgOver);
    const bias = winRate - fairProb;
    const avgOdds = allInBucket.reduce((s, x) => s + x.odds, 0) / allInBucket.length;
    const flatYield = (winRate * avgOdds - 1) * 100;
    const biasStr = bias >= 0 ? `+${(bias*100).toFixed(2)}%` : `${(bias*100).toFixed(2)}%`;
    const yieldStr = flatYield >= 0 ? `+${flatYield.toFixed(2)}%` : `${flatYield.toFixed(2)}%`;

    console.log(`  ${pad(bucket.label, 12)} n=${pad(String(allInBucket.length), 7)} Fair: ${(fairProb*100).toFixed(1)}%  Actual: ${(winRate*100).toFixed(1)}%  Bias: ${biasStr}  Yield: ${yieldStr}`);
  }

  // ─── Save results ───
  await mkdir(OUTPUT_DIR, { recursive: true });
  const outputPath = join(OUTPUT_DIR, 'bias_analysis.json');
  await writeFile(outputPath, JSON.stringify(results, null, 2), 'utf-8');
  console.log(`\nResults saved to ${outputPath}`);
}

// ─── Helpers ───

function getOddsForBetType(m: Match, bt: BetType): number | undefined {
  if (bt === 'H') return m.oddsHome;
  if (bt === 'D') return m.oddsDraw;
  return m.oddsAway;
}

function filterByOddsBucket(matches: Match[], bt: BetType, min: number, max: number): Match[] {
  return matches.filter(m => {
    const odds = getOddsForBetType(m, bt);
    return odds !== undefined && odds >= min && odds < max;
  });
}

function computeBias(matches: Match[], bt: BetType, segment: string): BiasResult {
  const n = matches.length;
  if (n === 0) {
    return {
      segment, n: 0, actualRate: 0, impliedProb: 0, fairProb: 0,
      rawBias: 0, fairBias: 0, ci95Lower: 0, ci95Upper: 0,
      significant: false, pValue: 1, avgOdds: 0, avgOverround: 0, flatYield: 0,
    };
  }

  let wins = 0;
  let sumImplied = 0;
  let sumOdds = 0;
  let sumOverround = 0;
  let sumFair = 0;
  let profitFlat = 0;

  for (const m of matches) {
    const odds = getOddsForBetType(m, bt)!;
    const implied = 1 / odds;
    const overround = 1/m.oddsHome! + 1/m.oddsDraw! + 1/m.oddsAway! - 1;
    const fair = implied / (1 + overround);
    const won = m.ftResult === bt;

    if (won) wins++;
    sumImplied += implied;
    sumOdds += odds;
    sumOverround += overround;
    sumFair += fair;
    profitFlat += won ? (odds - 1) : -1;
  }

  const actualRate = wins / n;
  const impliedProb = sumImplied / n;
  const fairProb = sumFair / n;
  const avgOdds = sumOdds / n;
  const avgOverround = sumOverround / n;
  const rawBias = actualRate - impliedProb;
  const fairBias = actualRate - fairProb;
  const flatYield = (profitFlat / n) * 100;

  // Wilson confidence interval for proportion
  const z = 1.96;
  const pHat = actualRate;
  const denom = 1 + z * z / n;
  const center = (pHat + z * z / (2 * n)) / denom;
  const halfWidth = (z * Math.sqrt(pHat * (1 - pHat) / n + z * z / (4 * n * n))) / denom;
  const ci95Lower = center - halfWidth - fairProb;
  const ci95Upper = center + halfWidth - fairProb;

  // Significance: does the CI of (actual - fair) exclude 0?
  const significant = (ci95Lower > 0 && ci95Upper > 0) || (ci95Lower < 0 && ci95Upper < 0);

  // Approximate z-test p-value
  const se = Math.sqrt(fairProb * (1 - fairProb) / n);
  const zStat = se > 0 ? Math.abs(fairBias) / se : 0;
  const pValue = 2 * (1 - normalCDF(zStat));

  return {
    segment, n, actualRate, impliedProb, fairProb, rawBias, fairBias,
    ci95Lower, ci95Upper, significant, pValue, avgOdds, avgOverround, flatYield,
  };
}

function normalCDF(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

function printBiasTable(dimLabel: string, data: Record<string, Record<BetType, BiasResult>>) {
  console.log(
    pad(dimLabel, 20) + pad('Type', 6) +
    pad('N', 8) + pad('Actual', 9) + pad('Fair', 9) +
    pad('Bias', 10) + pad('CI95', 20) + pad('Sig', 5) +
    pad('Yield', 9) + 'p-val'
  );
  console.log('-'.repeat(110));

  for (const [label, btMap] of Object.entries(data)) {
    for (const bt of BET_TYPES) {
      const b = btMap[bt];
      if (!b || b.n === 0) continue;
      const biasStr = b.fairBias >= 0 ? `+${(b.fairBias*100).toFixed(2)}%` : `${(b.fairBias*100).toFixed(2)}%`;
      const ciStr = `[${(b.ci95Lower*100).toFixed(2)}, ${(b.ci95Upper*100).toFixed(2)}]`;
      const sigStr = b.significant ? (b.fairBias > 0 ? '++' : '--') : '';
      const yieldStr = b.flatYield >= 0 ? `+${b.flatYield.toFixed(2)}%` : `${b.flatYield.toFixed(2)}%`;
      console.log(
        pad(label, 20) + pad(bt, 6) +
        pad(String(b.n), 8) + pad(`${(b.actualRate*100).toFixed(1)}%`, 9) +
        pad(`${(b.fairProb*100).toFixed(1)}%`, 9) +
        pad(biasStr, 10) + pad(ciStr, 20) + pad(sigStr, 5) +
        pad(yieldStr, 9) + b.pValue.toFixed(4)
      );
    }
  }
}

function printExploitableTable(biases: BiasResult[]) {
  console.log(
    pad('Segment', 28) + pad('N', 8) + pad('Fair Bias', 11) +
    pad('Yield', 9) + pad('Sig', 5) + 'p-value'
  );
  console.log('-'.repeat(75));
  for (const b of biases) {
    const biasStr = `+${(b.fairBias*100).toFixed(2)}%`;
    const yieldStr = b.flatYield >= 0 ? `+${b.flatYield.toFixed(2)}%` : `${b.flatYield.toFixed(2)}%`;
    console.log(
      pad(b.segment, 28) + pad(String(b.n), 8) + pad(biasStr, 11) +
      pad(yieldStr, 9) + pad(b.significant ? '**' : '', 5) + b.pValue.toFixed(4)
    );
  }
}

function pad(str: string, len: number): string {
  return str.padEnd(len);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
