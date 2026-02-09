/**
 * Archive Dataset Loader
 *
 * Parses the archive/Matches.csv file (230k+ matches, 38 leagues, 2000-2025)
 * into the standard Match[] interface used by the rest of the pipeline.
 *
 * Archive columns:
 *   Division, MatchDate, MatchTime, HomeTeam, AwayTeam, HomeElo, AwayElo,
 *   Form3Home, Form5Home, Form3Away, Form5Away, FTHome, FTAway, FTResult,
 *   HTHome, HTAway, HTResult, HomeShots, AwayShots, HomeTarget, AwayTarget,
 *   HomeFouls, AwayFouls, HomeCorners, AwayCorners, HomeYellow, AwayYellow,
 *   HomeRed, AwayRed, OddHome, OddDraw, OddAway, MaxHome, MaxDraw, MaxAway,
 *   Over25, Under25, MaxOver25, MaxUnder25, HandiSize, HandiHome, HandiAway,
 *   C_LTH, C_LTA, C_VHD, C_VAD, C_HTB, C_PHB
 */

import { readFile } from 'fs/promises';
import { parse } from 'csv-parse/sync';
import { ARCHIVE_LEAGUE_MAP } from '../../src/config.js';
import type { LeagueCode, Match } from '../../src/types/index.js';

const ARCHIVE_PATH = 'd:\\BetAI\\archive\\Matches.csv';

interface ArchiveRow {
  Division: string;
  MatchDate: string;
  MatchTime?: string;
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
  OddHome: string;
  OddDraw: string;
  OddAway: string;
  MaxHome: string;
  MaxDraw: string;
  MaxAway: string;
  Over25: string;
  Under25: string;
}

export interface ArchiveLoadOptions {
  /** Filter to specific league codes (after mapping from Division) */
  leagues?: LeagueCode[];
  /** Minimum year (inclusive), e.g. 2000 */
  minYear?: number;
  /** Maximum year (inclusive), e.g. 2025 */
  maxYear?: number;
  /** Path override for testing */
  archivePath?: string;
}

/**
 * Derive season string from a match date.
 * If month >= 7 (July+), season = "YYYY-(YY+1)" (e.g. "2015-16")
 * If month < 7, season = "(YYYY-1)-YY" (e.g. "2015-16" for Jan 2016)
 */
function deriveSeason(date: Date): string {
  const month = date.getMonth() + 1; // 1-12
  const year = date.getFullYear();

  if (month >= 7) {
    const endYear = year + 1;
    return `${year}-${String(endYear).slice(2)}`;
  } else {
    const startYear = year - 1;
    return `${startYear}-${String(year).slice(2)}`;
  }
}

function normalize(teamName: string): string {
  return teamName.replace(/\s+/g, '_').toLowerCase();
}

function formatDateId(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}${m}${d}`;
}

function optNum(v: string | undefined): number | undefined {
  if (v === undefined || v === '') return undefined;
  const n = Number(v);
  return isNaN(n) ? undefined : n;
}

function optOdds(v: string | undefined): number | undefined {
  if (v === undefined || v === '') return undefined;
  const n = Number(v);
  return isNaN(n) || n <= 1 ? undefined : n;
}

/**
 * Load all matches from the archive CSV.
 */
export async function loadArchiveMatches(
  options?: ArchiveLoadOptions,
): Promise<Match[]> {
  const csvPath = options?.archivePath ?? ARCHIVE_PATH;
  const content = await readFile(csvPath, 'utf-8');

  const records: ArchiveRow[] = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  });

  const matches: Match[] = [];
  let skippedNoLeague = 0;
  let skippedNoData = 0;
  let skippedDateFilter = 0;

  const leagueFilter = options?.leagues ? new Set(options.leagues) : null;

  for (const row of records) {
    // Map Division -> LeagueCode
    const leagueInfo = ARCHIVE_LEAGUE_MAP[row.Division];
    if (!leagueInfo) {
      skippedNoLeague++;
      continue;
    }

    const leagueCode = leagueInfo.leagueCode;

    // Filter by league
    if (leagueFilter && !leagueFilter.has(leagueCode)) {
      continue;
    }

    // Parse date (archive uses YYYY-MM-DD)
    if (!row.MatchDate) {
      skippedNoData++;
      continue;
    }
    const date = new Date(row.MatchDate);
    if (isNaN(date.getTime())) {
      skippedNoData++;
      continue;
    }

    // Filter by year
    const year = date.getFullYear();
    if (options?.minYear && year < options.minYear) {
      skippedDateFilter++;
      continue;
    }
    if (options?.maxYear && year > options.maxYear) {
      skippedDateFilter++;
      continue;
    }

    // Validate required fields
    if (!row.HomeTeam || !row.AwayTeam || !row.FTResult) {
      skippedNoData++;
      continue;
    }

    const ftHomeGoals = Number(row.FTHome);
    const ftAwayGoals = Number(row.FTAway);
    if (isNaN(ftHomeGoals) || isNaN(ftAwayGoals)) {
      skippedNoData++;
      continue;
    }

    const season = deriveSeason(date);
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
      ftResult: row.FTResult as 'H' | 'D' | 'A',
      htHomeGoals: optNum(row.HTHome),
      htAwayGoals: optNum(row.HTAway),
      homeShots: optNum(row.HomeShots),
      awayShots: optNum(row.AwayShots),
      homeShotsOnTarget: optNum(row.HomeTarget),
      awayShotsOnTarget: optNum(row.AwayTarget),
      homeCorners: optNum(row.HomeCorners),
      awayCorners: optNum(row.AwayCorners),
      homeYellow: optNum(row.HomeYellow),
      awayYellow: optNum(row.AwayYellow),
      homeRed: optNum(row.HomeRed),
      awayRed: optNum(row.AwayRed),
      oddsHome: optOdds(row.OddHome),
      oddsDraw: optOdds(row.OddDraw),
      oddsAway: optOdds(row.OddAway),
      oddsOver25: optOdds(row.Over25),
      oddsUnder25: optOdds(row.Under25),
      archiveHomeElo: optNum(row.HomeElo),
      archiveAwayElo: optNum(row.AwayElo),
      source: 'archive',
    });
  }

  // Sort by date
  matches.sort((a, b) => a.date.getTime() - b.date.getTime());

  console.log(`Archive: loaded ${matches.length} matches from ${csvPath}`);
  if (skippedNoLeague > 0) console.log(`  Skipped (unmapped league): ${skippedNoLeague}`);
  if (skippedNoData > 0) console.log(`  Skipped (missing data): ${skippedNoData}`);
  if (skippedDateFilter > 0) console.log(`  Skipped (date filter): ${skippedDateFilter}`);

  return matches;
}
