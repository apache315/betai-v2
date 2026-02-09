#!/usr/bin/env npx tsx
/**
 * Build features from downloaded CSV data + local xG data.
 *
 * Usage:
 *   npx tsx scripts/build-features.ts
 *   npx tsx scripts/build-features.ts --league E0
 *   npx tsx scripts/build-features.ts --output custom-features.json
 */

import { join } from 'path';
import { fileURLToPath } from 'url';
import { readdir, readFile } from 'fs/promises';
import { loadAllMatches } from '../data/scrapers/football-data.js';
import { loadArchiveMatches } from '../data/scrapers/archive-loader.js';
import { mergeMatchSources } from '../data/merge-matches.js';
import { buildFeatures, getFeatureNames } from '../ml/features.js';
import type { LeagueCode } from '../src/types/index.js';
import type { Match } from '../src/types/index.js';

const __dirname = join(fileURLToPath(import.meta.url), '..');

/**
 * Interface for loaded xG data
 */
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
 * Load all xG data from local JSON files
 */
async function loadLocalXGData(): Promise<XGMatch[]> {
  const dataDir = 'd:\\BetAI\\v2\\data\\raw';
  const files = await readdir(dataDir);
  const xgFiles = files.filter((f) => f.startsWith('xg_') && f.endsWith('.json'));

  console.log(`Loading ${xgFiles.length} xG files from local data...`);

  const allXG: XGMatch[] = [];
  for (const file of xgFiles) {
    const content = await readFile(join(dataDir, file), 'utf-8');
    const data = JSON.parse(content) as XGMatch[];
    allXG.push(...data);
  }

  return allXG;
}

/**
 * Normalize team name for matching between sources
 */
function normalizeTeam(name: string): string {
  return name
    .toLowerCase()
    .replace(/fc |afc |ac |as |ss |us |ssc |1\. |/gi, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Merge local xG data into matches
 */
function mergeXGData(matches: Match[], xgData: XGMatch[]): { merged: number; total: number } {
  // Build lookup map: "leagueCode_date_homeNorm_awayNorm" -> xG data
  const xgMap = new Map<string, XGMatch>();

  for (const xg of xgData) {
    const key = `${xg.league}_${xg.date}_${normalizeTeam(xg.homeTeam)}_${normalizeTeam(xg.awayTeam)}`;
    xgMap.set(key, xg);
  }

  let merged = 0;

  for (const match of matches) {
    const dateStr = match.date.toISOString().slice(0, 10);
    const key = `${match.league}_${dateStr}_${normalizeTeam(match.homeTeam)}_${normalizeTeam(match.awayTeam)}`;

    const xg = xgMap.get(key);
    if (xg) {
      match.homeXG = xg.homeXG;
      match.awayXG = xg.awayXG;
      merged++;
    }
  }

  return { merged, total: matches.length };
}

async function main() {
  console.log('=== BetAI v2 - Build Features ===\n');

  // Parse arguments
  const args = process.argv.slice(2);
  let leagues: LeagueCode[] | undefined;
  let outputPath = join(__dirname, '..', 'data', 'processed', 'features.json');
  let useArchive = true;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--league' && args[i + 1]) {
      leagues = [args[i + 1].toUpperCase() as LeagueCode];
    }
    if (args[i] === '--output' && args[i + 1]) {
      outputPath = args[i + 1];
    }
    if (args[i] === '--no-archive') {
      useArchive = false;
    }
  }

  // Load v2 raw matches (5 leagues, 2015-2025)
  console.log('Loading v2 matches from CSVs...');
  const v2Matches = await loadAllMatches(leagues);
  console.log(`v2: ${v2Matches.length} matches`);

  // Load archive matches (230k, 38 leagues, 2000-2025)
  let matches: Match[];
  if (useArchive) {
    console.log('\nLoading archive matches...');
    const archiveMatches = await loadArchiveMatches(
      leagues ? { leagues } : undefined,
    );

    console.log('\nMerging data sources...');
    matches = mergeMatchSources(v2Matches, archiveMatches);
  } else {
    console.log('\n--no-archive: using only v2 data');
    matches = v2Matches;
  }

  // Load and merge xG data (from local files, covers 5 primary leagues)
  console.log('\nLoading xG data from local files...');
  const xgData = await loadLocalXGData();
  console.log(`Found ${xgData.length} matches with xG data`);

  if (xgData.length > 0) {
    const { merged, total } = mergeXGData(matches, xgData);
    console.log(`Merged xG: ${merged}/${total} matches (${((merged / total) * 100).toFixed(1)}%)`);
  } else {
    console.log('No xG data found. Run: npm run import && npm run convert');
  }

  if (matches.length === 0) {
    console.error('No matches found. Run download-data.ts first.');
    process.exit(1);
  }

  console.log(`\nTotal: ${matches.length} matches`);
  console.log(`Date range: ${matches[0].date.toISOString().slice(0, 10)} to ${matches[matches.length - 1].date.toISOString().slice(0, 10)}`);

  // Build features
  console.log('\nBuilding features...');
  const features = buildFeatures(matches);

  console.log(`\nFeature count: ${getFeatureNames().length}`);
  console.log(`Matches with features: ${features.length}`);
  console.log(`Matches with closing odds: ${features.filter(f => f.closingOdds).length}`);

  // Save to JSON (stream to avoid string length limit for large datasets)
  console.log(`\nSaving ${features.length} features to ${outputPath}...`);
  const { createWriteStream } = await import('fs');
  await new Promise<void>((resolve, reject) => {
    const stream = createWriteStream(outputPath, { encoding: 'utf-8' });
    stream.on('error', reject);
    stream.write('[\n');
    for (let i = 0; i < features.length; i++) {
      const json = JSON.stringify(features[i]);
      stream.write(i === 0 ? json : ',\n' + json);
    }
    stream.write('\n]');
    stream.end(() => resolve());
  });
  console.log(`Saved to: ${outputPath}`);

  // Show sample
  if (features.length > 0) {
    const sample = features[features.length - 1];
    console.log('\n--- Sample (latest match) ---');
    console.log(`Match: ${sample.homeTeam} vs ${sample.awayTeam}`);
    console.log(`Date: ${new Date(sample.date).toISOString().slice(0, 10)}`);
    console.log(`Result: ${sample.result}`);
    console.log(`Features: ${Object.keys(sample.features).length}`);
    console.log(`Sample features:`);
    const sampleKeys = ['home_win_rate_5', 'away_win_rate_5', 'odds_implied_home', 'diff_ppg_5'];
    for (const key of sampleKeys) {
      if (sample.features[key] !== undefined) {
        console.log(`  ${key}: ${sample.features[key].toFixed(3)}`);
      }
    }
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
