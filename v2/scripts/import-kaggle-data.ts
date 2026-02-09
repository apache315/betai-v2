#!/usr/bin/env npx tsx
/**
 * Import and clean Kaggle dataset
 * Input: d:\BetAI\archive\Matches.csv
 * Output: d:\BetAI\v2\data\raw\matches_cleaned.json
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { parse } from 'csv-parse/sync';

interface RawMatch {
  Division: string;
  MatchDate: string;
  MatchTime: string;
  HomeTeam: string;
  AwayTeam: string;
  HomeElo: string;
  AwayElo: string;
  Form3Home: string;
  Form5Home: string;
  Form3Away: string;
  Form5Away: string;
  FTHome: string;
  FTAway: string;
  FTResult: string;
  HTHome: string;
  HTAway: string;
  HTResult: string;
  HomeShots: string;
  AwayShots: string;
  HomeTarget: string;
  AwayTarget: string;
  HomeFouls: string;
  AwayFouls: string;
  HomeCorners: string;
  AwayCorners: string;
  HomeYellow: string;
  AwayYellow: string;
  HomeRed: string;
  AwayRed: string;
  [key: string]: string;
}

interface CleanedMatch {
  id: number;
  league: string;
  date: string;
  homeTeam: string;
  awayTeam: string;
  homeGoals: number;
  awayGoals: number;
  homeXG: number;
  awayXG: number;
  homeShots: number;
  awayShots: number;
  homeTarget: number;
  awayTarget: number;
  result: string;
  homeElo: number;
  awayElo: number;
  form3Home: number;
  form5Home: number;
  form3Away: number;
  form5Away: number;
}

// Target leagues: E0 (EPL), D1 (Bundesliga), F1 (La Liga), I1 (Serie A), SP1 (Eredivisie)
const TARGET_LEAGUES = ['E0', 'D1', 'F1', 'I1', 'SP1'];
const TARGET_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];

/**
 * Derive xG from shots
 * Formula: xG ≈ shotsOnTarget * 0.08 + (totalShots - shotsOnTarget) * 0.01
 * This is an approximation - real xG uses position, angle, etc.
 */
function deriveXG(shotsOnTarget: number, totalShots: number): number {
  if (isNaN(shotsOnTarget) || isNaN(totalShots)) return 0;
  const shotsOnTarget_safe = Math.max(0, Math.min(shotsOnTarget, totalShots));
  const otherShots = Math.max(0, totalShots - shotsOnTarget_safe);
  return shotsOnTarget_safe * 0.08 + otherShots * 0.01;
}

/**
 * Parse safe number
 */
function parseNum(val: string | undefined): number {
  if (!val || val === '' || val === 'NaN') return 0;
  const num = parseFloat(val);
  return isNaN(num) ? 0 : num;
}

/**
 * Extract year from date string (YYYY-MM-DD)
 */
function extractYear(dateStr: string): number {
  if (!dateStr) return 0;
  return parseInt(dateStr.split('-')[0], 10);
}

/**
 * Filter and transform matches
 */
function cleanMatches(rawMatches: RawMatch[]): CleanedMatch[] {
  const cleaned: CleanedMatch[] = [];
  let id = 1;

  for (const match of rawMatches) {
    // Skip if missing critical fields
    if (!match.Division || !match.MatchDate || !match.HomeTeam || !match.AwayTeam) {
      continue;
    }

    // Filter by league
    if (!TARGET_LEAGUES.includes(match.Division)) {
      continue;
    }

    // Filter by year
    const year = extractYear(match.MatchDate);
    if (!TARGET_YEARS.includes(year)) {
      continue;
    }

    // Parse values
    const homeGoals = parseNum(match.FTHome);
    const awayGoals = parseNum(match.FTAway);
    const homeShots = parseNum(match.HomeShots);
    const awayShots = parseNum(match.AwayShots);
    const homeTarget = parseNum(match.HomeTarget);
    const awayTarget = parseNum(match.AwayTarget);

    // Skip incomplete matches (no goals = no shots data usually)
    if (homeGoals < 0 || awayGoals < 0) {
      continue;
    }

    // Derive xG
    const homeXG = deriveXG(homeTarget, homeShots);
    const awayXG = deriveXG(awayTarget, awayShots);

    cleaned.push({
      id: id++,
      league: match.Division,
      date: match.MatchDate,
      homeTeam: match.HomeTeam.trim(),
      awayTeam: match.AwayTeam.trim(),
      homeGoals,
      awayGoals,
      homeXG,
      awayXG,
      homeShots,
      awayShots,
      homeTarget,
      awayTarget,
      result: match.FTResult || '',
      homeElo: parseNum(match.HomeElo),
      awayElo: parseNum(match.AwayElo),
      form3Home: parseNum(match.Form3Home),
      form5Home: parseNum(match.Form5Home),
      form3Away: parseNum(match.Form3Away),
      form5Away: parseNum(match.Form5Away),
    });
  }

  return cleaned;
}

/**
 * Main
 */
async function main() {
  console.log('=== Importing Kaggle Dataset ===\n');

  const archiveDir = 'd:\\BetAI\\archive';
  const csvPath = join(archiveDir, 'Matches.csv');
  const outputDir = 'd:\\BetAI\\v2\\data\\raw';
  const outputPath = join(outputDir, 'matches_cleaned.json');

  try {
    // Read CSV
    console.log(`Reading: ${csvPath}`);
    const csv = readFileSync(csvPath, 'utf-8');

    // Parse CSV
    console.log('Parsing CSV...');
    const records = parse(csv, {
      columns: true,
      skip_empty_lines: true,
    }) as RawMatch[];

    console.log(`  Total rows: ${records.length}`);

    // Clean
    console.log('Cleaning and filtering...');
    const cleaned = cleanMatches(records);

    console.log(`  ✅ Matches kept: ${cleaned.length}`);
    console.log(`  Filtered out: ${records.length - cleaned.length}`);

    // Statistics
    const byLeague = cleaned.reduce(
      (acc, m) => {
        acc[m.league] = (acc[m.league] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );

    console.log('\n  By league:');
    for (const [league, count] of Object.entries(byLeague)) {
      console.log(`    ${league}: ${count} matches`);
    }

    // Write output
    console.log(`\nWriting: ${outputPath}`);
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(outputPath, JSON.stringify(cleaned, null, 2), 'utf-8');

    console.log(`\n✅ Done! ${cleaned.length} matches imported and cleaned`);
  } catch (error: any) {
    console.error(`\n❌ Error: ${error.message}`);
    process.exit(1);
  }
}

main();
