#!/usr/bin/env npx tsx
/**
 * Convert cleaned matches to xG format compatible with scrapers
 * This creates files matching the fbref/understat/statsbomb format
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

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

interface XGMatch {
  id: number;
  isResult: boolean;
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
 * Extract season from date (YYYY-MM-DD)
 * E.g., 2015-07-01 to 2015-06-30 -> 15-16
 */
function extractSeason(dateStr: string): string {
  const [year, month] = dateStr.split('-');
  const yearNum = parseInt(year, 10);
  const monthNum = parseInt(month, 10);

  // If month < 7, it's the previous season
  const seasonStart = monthNum < 7 ? yearNum - 1 : yearNum;
  const seasonEnd = seasonStart + 1;
  return `${seasonStart}-${String(seasonEnd).slice(2)}`;
}

/**
 * Convert to xG format
 */
function convertMatches(cleaned: CleanedMatch[]): XGMatch[] {
  return cleaned.map((m) => ({
    id: m.id,
    isResult: true,
    homeTeam: m.homeTeam,
    awayTeam: m.awayTeam,
    homeGoals: m.homeGoals,
    awayGoals: m.awayGoals,
    homeXG: m.homeXG,
    awayXG: m.awayXG,
    date: m.date,
    league: m.league,
    season: extractSeason(m.date),
  }));
}

/**
 * Group by league+season and write files
 */
function writeByLeagueSeason(matches: XGMatch[], outputDir: string) {
  const grouped = new Map<string, XGMatch[]>();

  for (const match of matches) {
    const key = `${match.league}_${match.season}`;
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key)!.push(match);
  }

  mkdirSync(outputDir, { recursive: true });

  for (const [key, group] of grouped) {
    const [league, season] = key.split('_');
    const filename = `xg_${league}_${season}.json`;
    const filepath = join(outputDir, filename);
    writeFileSync(filepath, JSON.stringify(group, null, 2), 'utf-8');
    console.log(`  ${filename}: ${group.length} matches`);
  }
}

/**
 * Main
 */
async function main() {
  console.log('=== Converting to xG Format ===\n');

  const inputPath = 'd:\\BetAI\\v2\\data\\raw\\matches_cleaned.json';
  const outputDir = 'd:\\BetAI\\v2\\data\\raw';

  try {
    console.log(`Reading: ${inputPath}`);
    const json = readFileSync(inputPath, 'utf-8');
    const cleaned = JSON.parse(json) as CleanedMatch[];

    console.log(`  ${cleaned.length} matches`);

    console.log('\nConverting to xG format...');
    const converted = convertMatches(cleaned);

    console.log('\nWriting by league+season:');
    writeByLeagueSeason(converted, outputDir);

    console.log(`\n✅ Done! xG data ready for features pipeline`);
  } catch (error: any) {
    console.error(`\n❌ Error: ${error.message}`);
    process.exit(1);
  }
}

main();
