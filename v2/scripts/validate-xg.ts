#!/usr/bin/env npx tsx
/**
 * Validate and analyze xG data quality
 * Checks: distribution, outliers, coverage, correlation with results
 */

import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

interface XGMatch {
  id: number;
  homeTeam: string;
  awayTeam: string;
  homeGoals: number;
  awayGoals: number;
  homeXG: number;
  awayXG: number;
  date: string;
  league: string;
  season: string;
}

/**
 * Calculate basic stats
 */
function stats(arr: number[]): { min: number; max: number; mean: number; median: number; std: number } {
  const sorted = [...arr].sort((a, b) => a - b);
  const n = arr.length;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  const median = sorted[Math.floor(n / 2)];
  const variance = arr.reduce((a, x) => a + Math.pow(x - mean, 2), 0) / n;
  const std = Math.sqrt(variance);

  return {
    min: sorted[0],
    max: sorted[n - 1],
    mean: Math.round(mean * 100) / 100,
    median: Math.round(median * 100) / 100,
    std: Math.round(std * 100) / 100,
  };
}

/**
 * Correlation between xG and goals
 */
function correlation(xg: number[], goals: number[]): number {
  const n = xg.length;
  const meanXg = xg.reduce((a, b) => a + b, 0) / n;
  const meanGoals = goals.reduce((a, b) => a + b, 0) / n;

  const numerator = xg.reduce((a, x, i) => a + (x - meanXg) * (goals[i] - meanGoals), 0);
  const denomX = Math.sqrt(xg.reduce((a, x) => a + Math.pow(x - meanXg, 2), 0));
  const denomGoals = Math.sqrt(goals.reduce((a, g) => a + Math.pow(g - meanGoals, 2), 0));

  return denomX === 0 || denomGoals === 0 ? 0 : numerator / (denomX * denomGoals);
}

/**
 * Main
 */
async function main() {
  console.log('=== xG Data Quality Validation ===\n');

  const dataDir = 'd:\\BetAI\\v2\\data\\raw';
  const files = readdirSync(dataDir).filter((f) => f.startsWith('xg_'));

  if (files.length === 0) {
    console.error('❌ No xG files found');
    process.exit(1);
  }

  console.log(`Found ${files.length} xG files\n`);

  const allMatches: XGMatch[] = [];
  const allHomeXG: number[] = [];
  const allAwayXG: number[] = [];
  const allHomeGoals: number[] = [];
  const allAwayGoals: number[] = [];
  const allTotalXG: number[] = [];
  const allTotalGoals: number[] = [];

  // Load all data
  for (const file of files) {
    const content = readFileSync(join(dataDir, file), 'utf-8');
    const matches = JSON.parse(content) as XGMatch[];
    allMatches.push(...matches);

    for (const m of matches) {
      allHomeXG.push(m.homeXG);
      allAwayXG.push(m.awayXG);
      allHomeGoals.push(m.homeGoals);
      allAwayGoals.push(m.awayGoals);
      allTotalXG.push(m.homeXG + m.awayXG);
      allTotalGoals.push(m.homeGoals + m.awayGoals);
    }
  }

  console.log(`📊 TOTAL DATA`);
  console.log(`  Matches: ${allMatches.length}`);
  console.log(`  Date range: ${allMatches[0].date} to ${allMatches[allMatches.length - 1].date}\n`);

  // xG Stats
  console.log(`📈 HOME TEAM xG`);
  const homeXGStats = stats(allHomeXG);
  console.log(`  Mean:   ${homeXGStats.mean} (±${homeXGStats.std})`);
  console.log(`  Median: ${homeXGStats.median}`);
  console.log(`  Range:  ${homeXGStats.min} - ${homeXGStats.max}\n`);

  console.log(`📈 AWAY TEAM xG`);
  const awayXGStats = stats(allAwayXG);
  console.log(`  Mean:   ${awayXGStats.mean} (±${awayXGStats.std})`);
  console.log(`  Median: ${awayXGStats.median}`);
  console.log(`  Range:  ${awayXGStats.min} - ${awayXGStats.max}\n`);

  console.log(`⚽ HOME TEAM GOALS (Actual)`);
  const homeGoalsStats = stats(allHomeGoals);
  console.log(`  Mean:   ${homeGoalsStats.mean} (±${homeGoalsStats.std})`);
  console.log(`  Median: ${homeGoalsStats.median}`);
  console.log(`  Range:  ${homeGoalsStats.min} - ${homeGoalsStats.max}\n`);

  console.log(`⚽ AWAY TEAM GOALS (Actual)`);
  const awayGoalsStats = stats(allAwayGoals);
  console.log(`  Mean:   ${awayGoalsStats.mean} (±${awayGoalsStats.std})`);
  console.log(`  Median: ${awayGoalsStats.median}`);
  console.log(`  Range:  ${awayGoalsStats.min} - ${awayGoalsStats.max}\n`);

  // Total goals distribution
  console.log(`📊 TOTAL GOALS PER MATCH (Actual)`);
  const totalGoalsStats = stats(allTotalGoals);
  console.log(`  Mean:   ${totalGoalsStats.mean} goals/match`);
  console.log(`  Median: ${totalGoalsStats.median} goals/match`);

  const goalCounts = new Map<number, number>();
  for (const g of allTotalGoals) {
    goalCounts.set(g, (goalCounts.get(g) || 0) + 1);
  }
  const under25 = Array.from(goalCounts.entries())
    .filter(([g]) => g <= 2)
    .reduce((a, [, c]) => a + c, 0);
  const over25 = allTotalGoals.length - under25;
  console.log(`  Under 2.5: ${under25} (${Math.round((under25 / allTotalGoals.length) * 100)}%)`);
  console.log(`  Over 2.5:  ${over25} (${Math.round((over25 / allTotalGoals.length) * 100)}%)\n`);

  // Correlation
  console.log(`🔗 CORRELATION: xG vs Actual Goals`);
  const homeCorr = correlation(allHomeXG, allHomeGoals);
  const awayCorr = correlation(allAwayXG, allAwayGoals);
  console.log(`  Home: ${(homeCorr * 100).toFixed(1)}%`);
  console.log(`  Away: ${(awayCorr * 100).toFixed(1)}%`);
  console.log(`  Average: ${(((homeCorr + awayCorr) / 2) * 100).toFixed(1)}%\n`);

  // Coverage
  console.log(`🔍 COVERAGE`);
  const withXG = allMatches.filter((m) => m.homeXG > 0 || m.awayXG > 0).length;
  console.log(`  Matches with xG: ${withXG}/${allMatches.length} (${Math.round((withXG / allMatches.length) * 100)}%)\n`);

  // Quality assessment
  console.log(`✅ QUALITY ASSESSMENT`);
  const issues: string[] = [];

  if (homeCorr < 0.3 || awayCorr < 0.3) {
    issues.push(`⚠️  Low correlation (xG vs goals) - ${Math.round(((homeCorr + awayCorr) / 2) * 100)}%`);
  } else {
    console.log(`  ✅ xG correlation good (${Math.round(((homeCorr + awayCorr) / 2) * 100)}%)`);
  }

  if (withXG < allMatches.length * 0.9) {
    issues.push(`⚠️  Missing xG in ${allMatches.length - withXG} matches`);
  } else {
    console.log(`  ✅ xG coverage complete (${Math.round((withXG / allMatches.length) * 100)}%)`);
  }

  if (Math.abs(homeXGStats.mean - 1.5) > 0.5) {
    issues.push(`⚠️  Home xG distribution seems off (mean: ${homeXGStats.mean})`);
  } else {
    console.log(`  ✅ xG distribution looks realistic`);
  }

  if (issues.length === 0) {
    console.log('  ✅ All checks passed!\n');
    console.log('🚀 Data is ready for features pipeline');
  } else {
    console.log(`\n⚠️  Issues found:`);
    for (const issue of issues) {
      console.log(`  ${issue}`);
    }
    console.log('\n⚠️  Note: Derived xG may have lower correlation than real xG');
    console.log('   This is expected - you can still use it for features');
  }
}

main().catch(console.error);
