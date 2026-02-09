/**
 * football-data.co.uk CSV Downloader
 *
 * Downloads historical match data CSV files for target leagues and seasons.
 * Source: https://www.football-data.co.uk/
 *
 * CSV columns include: results, shots, corners, cards, and closing odds
 * from Bet365, Pinnacle, William Hill, and market averages.
 */

import axios from 'axios';
import { writeFile, mkdir, readFile, stat } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { parse } from 'csv-parse/sync';
import { LEAGUES, getFootballDataURL, getTargetSeasons } from '../../src/config.js';
import type { LeagueCode, RawMatchCSV, Match } from '../../src/types/index.js';

const __dirname = join(fileURLToPath(import.meta.url), '..');
const DATA_RAW_DIR = join(__dirname, '..', 'raw');

/**
 * Download a single CSV file from football-data.co.uk
 */
export async function downloadCSV(
  leagueCode: LeagueCode,
  seasonCode: string,
  seasonLabel: string,
): Promise<string | null> {
  const league = LEAGUES[leagueCode];
  const url = getFootballDataURL(league.csvCode, seasonCode);
  const filename = `${league.csvCode}_${seasonCode}.csv`;
  const filepath = join(DATA_RAW_DIR, filename);

  // Check if already exists
  try {
    await stat(filepath);
    console.log(`  [skip] ${filename} already exists`);
    return filepath;
  } catch {
    // File doesn't exist, download it
  }

  try {
    console.log(`  [download] ${league.name} ${seasonLabel} -> ${url}`);
    const response = await axios.get(url, {
      responseType: 'text',
      timeout: 30000,
      headers: {
        'User-Agent': 'BetAI/2.0 (research)',
      },
    });

    if (!response.data || response.data.length < 100) {
      console.log(`  [warn] Empty or too small response for ${filename}`);
      return null;
    }

    await mkdir(dirname(filepath), { recursive: true });
    await writeFile(filepath, response.data, 'utf-8');
    console.log(`  [ok] Saved ${filename} (${(response.data.length / 1024).toFixed(1)} KB)`);
    return filepath;
  } catch (error: any) {
    if (error.response?.status === 404) {
      console.log(`  [404] Not found: ${url}`);
    } else {
      console.error(`  [error] ${filename}: ${error.message}`);
    }
    return null;
  }
}

/**
 * Download all CSV files for all leagues and seasons
 */
export async function downloadAll(
  leagues?: LeagueCode[],
): Promise<{ downloaded: number; skipped: number; failed: number }> {
  const targetLeagues = leagues || (Object.keys(LEAGUES) as LeagueCode[]);
  const seasons = getTargetSeasons();

  let downloaded = 0;
  let skipped = 0;
  let failed = 0;

  await mkdir(DATA_RAW_DIR, { recursive: true });

  for (const leagueCode of targetLeagues) {
    const league = LEAGUES[leagueCode];
    console.log(`\n=== ${league.name} (${league.country}) ===`);

    for (const season of seasons) {
      const result = await downloadCSV(leagueCode, season.code, season.label);
      if (result) {
        downloaded++;
      } else {
        failed++;
      }

      // Rate limiting: 500ms between requests
      await new Promise(r => setTimeout(r, 500));
    }
  }

  console.log(`\nDone: ${downloaded} downloaded, ${skipped} skipped, ${failed} failed`);
  return { downloaded, skipped, failed };
}

/**
 * Parse a raw CSV file into normalized Match objects
 */
export function parseCSV(
  csvContent: string,
  leagueCode: LeagueCode,
  season: string,
): Match[] {
  const records: RawMatchCSV[] = parse(csvContent, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  });

  const matches: Match[] = [];

  for (const row of records) {
    if (!row.HomeTeam || !row.AwayTeam || !row.FTHG || !row.FTR) {
      continue; // Skip incomplete rows
    }

    const date = parseDate(row.Date);
    if (!date) continue;

    const ftHomeGoals = Number(row.FTHG);
    const ftAwayGoals = Number(row.FTAG);

    if (isNaN(ftHomeGoals) || isNaN(ftAwayGoals)) continue;

    // Pick best opening odds: Pinnacle > Bet365 > Average
    const oddsHome = pickOdds(row.PSH, row.B365H, row.AvgH);
    const oddsDraw = pickOdds(row.PSD, row.B365D, row.AvgD);
    const oddsAway = pickOdds(row.PSA, row.B365A, row.AvgA);

    // Closing odds (suffix C, available from 2019/20+): Pinnacle > Bet365 > Average
    const closingOddsHome = pickOdds(row.PSCH, row.B365CH, row.AvgCH);
    const closingOddsDraw = pickOdds(row.PSCD, row.B365CD, row.AvgCD);
    const closingOddsAway = pickOdds(row.PSCA, row.B365CA, row.AvgCA);

    // Per-bookmaker odds (for bias analysis)
    const oddsByBookmaker: Record<string, { home: number; draw: number; away: number }> = {};
    const bookmakers = [
      { key: 'B365', h: row.B365H, d: row.B365D, a: row.B365A },
      { key: 'PS',   h: row.PSH,   d: row.PSD,   a: row.PSA },
      { key: 'WH',   h: row.WHH,   d: row.WHD,   a: row.WHA },
      { key: 'BW',   h: row.BWH,   d: row.BWD,   a: row.BWA },
      { key: 'IW',   h: row.IWH,   d: row.IWD,   a: row.IWA },
    ];
    for (const bk of bookmakers) {
      const h = Number(bk.h), d = Number(bk.d), a = Number(bk.a);
      if (h > 1 && d > 1 && a > 1) {
        oddsByBookmaker[bk.key] = { home: h, draw: d, away: a };
      }
    }

    const id = `${leagueCode}_${formatDateId(date)}_${normalize(row.HomeTeam)}_${normalize(row.AwayTeam)}`;

    matches.push({
      id,
      league: leagueCode,
      season,
      date,
      homeTeam: row.HomeTeam,
      awayTeam: row.AwayTeam,
      ftHomeGoals,
      ftAwayGoals,
      ftResult: row.FTR as 'H' | 'D' | 'A',
      htHomeGoals: optionalNum(row.HTHG),
      htAwayGoals: optionalNum(row.HTAG),
      homeShots: optionalNum(row.HS),
      awayShots: optionalNum(row.AS),
      homeShotsOnTarget: optionalNum(row.HST),
      awayShotsOnTarget: optionalNum(row.AST),
      homeCorners: optionalNum(row.HC),
      awayCorners: optionalNum(row.AC),
      homeYellow: optionalNum(row.HY),
      awayYellow: optionalNum(row.AY),
      homeRed: optionalNum(row.HR),
      awayRed: optionalNum(row.AR),
      oddsHome,
      oddsDraw,
      oddsAway,
      closingOddsHome,
      closingOddsDraw,
      closingOddsAway,
      oddsByBookmaker: Object.keys(oddsByBookmaker).length > 0 ? oddsByBookmaker : undefined,
      avgOddsHome: optionalNum(row.AvgH),
      avgOddsDraw: optionalNum(row.AvgD),
      avgOddsAway: optionalNum(row.AvgA),
      maxOddsHome: optionalNum(row.MaxH),
      maxOddsDraw: optionalNum(row.MaxD),
      maxOddsAway: optionalNum(row.MaxA),
      oddsOver25: optionalNum(row['Avg>2.5']),
      oddsUnder25: optionalNum(row['Avg<2.5']),
    });
  }

  return matches;
}

/**
 * Load all downloaded CSVs and return normalized matches
 */
export async function loadAllMatches(
  leagues?: LeagueCode[],
): Promise<Match[]> {
  const targetLeagues = leagues || (Object.keys(LEAGUES) as LeagueCode[]);
  const seasons = getTargetSeasons();
  const allMatches: Match[] = [];

  for (const leagueCode of targetLeagues) {
    const league = LEAGUES[leagueCode];

    for (const season of seasons) {
      const filename = `${league.csvCode}_${season.code}.csv`;
      const filepath = join(DATA_RAW_DIR, filename);

      try {
        const content = await readFile(filepath, 'utf-8');
        const matches = parseCSV(content, leagueCode, season.label);
        allMatches.push(...matches);
      } catch {
        // File not found, skip
      }
    }
  }

  // Sort by date
  allMatches.sort((a, b) => a.date.getTime() - b.date.getTime());

  console.log(`Loaded ${allMatches.length} matches from ${targetLeagues.length} leagues`);
  return allMatches;
}

// ─── Helpers ───

function parseDate(dateStr: string): Date | null {
  if (!dateStr) return null;

  // Try DD/MM/YYYY
  const parts = dateStr.split('/');
  if (parts.length === 3) {
    let [day, month, year] = parts;
    if (year.length === 2) {
      year = Number(year) > 50 ? `19${year}` : `20${year}`;
    }
    const d = new Date(Number(year), Number(month) - 1, Number(day));
    if (!isNaN(d.getTime())) return d;
  }

  // Try ISO
  const d = new Date(dateStr);
  return isNaN(d.getTime()) ? null : d;
}

function formatDateId(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}${m}${d}`;
}

function normalize(teamName: string): string {
  return teamName.replace(/\s+/g, '_').toLowerCase();
}

function pickOdds(...values: (number | string | undefined)[]): number | undefined {
  for (const v of values) {
    const n = Number(v);
    if (!isNaN(n) && n > 1) return n;
  }
  return undefined;
}

function optionalNum(v: any): number | undefined {
  const n = Number(v);
  return isNaN(n) ? undefined : n;
}
