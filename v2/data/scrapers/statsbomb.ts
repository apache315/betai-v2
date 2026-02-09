/**
 * StatsBomb xG Scraper
 *
 * Scrapes expected goals (xG) data from StatsBomb free API.
 * StatsBomb provides match-level and shot-level xG for major competitions.
 *
 * API: https://www.statsbomb.com/soccer-data/api
 * Free data: Open Leagues (international friendlies, European competitions)
 * Requires aggregation of shot events into match-level xG.
 *
 * Rate limit: ~50ms between requests (respectful)
 */

import axios from 'axios';
import { writeFile, mkdir, readFile, stat } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { LEAGUES } from '../../src/config.js';
import type { LeagueCode } from '../../src/types/index.js';

const __dirname = join(fileURLToPath(import.meta.url), '..');
const DATA_RAW_DIR = join(__dirname, '..', 'raw');

// StatsBomb API base
const STATSBOMB_API = 'https://raw.githubusercontent.com/statsbomb/StatsBomb/master/data';

// Competition mapping: LeagueCode -> StatsBomb competition_id
const COMPETITION_MAPPING: Record<string, number> = {
  E0: 2, // Premier League
  D1: 3, // Bundesliga
  F1: 4, // La Liga
  I1: 5, // Serie A
  SP1: 7, // Eredivisie (Netherlands)
};

// Season mapping: startYear -> StatsBomb season_id
// StatsBomb seasons are enumerated from their dataset
const STATSBOMB_SEASONS: Record<number, number> = {
  2015: 1, 2016: 2, 2017: 3, 2018: 4, 2019: 5,
  2020: 6, 2021: 7, 2022: 8, 2023: 9, 2024: 10,
};

export interface StatsBombMatch {
  id: number;
  isResult: boolean;
  homeTeam: string;
  awayTeam: string;
  homeGoals: number;
  awayGoals: number;
  homeXG: number;
  awayXG: number;
  date: string; // YYYY-MM-DD HH:mm:ss
  league: string;
  season: string;
}

interface SBCompetition {
  competition_id: number;
  season_id: number;
  competition_name: string;
  season_name: string;
}

interface SBMatch {
  id: string;
  match_date: string;
  kick_off: string;
  status: string;
  minute: number;
  second: number;
  period: number;
  home_team: { id: number; name: string };
  away_team: { id: number; name: string };
  home_score: number;
  away_score: number;
  competition: SBCompetition;
}

interface SBEvent {
  type: { name: string };
  team: { id: number; name: string };
  shot?: {
    statsbomb_xg: number;
    result: { name: string };
  };
}

/**
 * Retry logic with exponential backoff
 */
async function fetchWithRetry(url: string, maxAttempts = 3): Promise<any> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const response = await axios.get(url, {
        timeout: 30000,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        },
      });
      return response.data;
    } catch (error: any) {
      const isLastAttempt = attempt === maxAttempts;
      const isNetworkError =
        error.code === 'ENOTFOUND' ||
        error.code === 'ECONNREFUSED' ||
        error.code === 'ETIMEDOUT' ||
        error.response?.status >= 500;

      if (!isLastAttempt && isNetworkError) {
        const delayMs = 2000 * Math.pow(2, attempt - 1);
        console.log(`    [retry ${attempt}/${maxAttempts}] ${error.message} - waiting ${delayMs}ms...`);
        await new Promise((r) => setTimeout(r, delayMs));
        continue;
      }

      if (isLastAttempt) {
        throw error;
      }
    }
  }
  throw new Error('All retry attempts failed');
}

/**
 * Fetch match list for a competition+season from StatsBomb
 */
async function fetchMatches(
  competitionId: number,
  seasonId: number,
): Promise<SBMatch[]> {
  const url = `${STATSBOMB_API}/matches/${competitionId}/${seasonId}.json`;
  console.log(`    [fetch] ${url}`);

  try {
    const data = await fetchWithRetry(url);
    return data;
  } catch (error: any) {
    console.error(`    [error] Failed to fetch matches: ${error.message}`);
    return [];
  }
}

/**
 * Fetch events for a specific match (contains shot xG data)
 */
async function fetchMatchEvents(matchId: string): Promise<SBEvent[]> {
  const url = `${STATSBOMB_API}/events/${matchId}.json`;

  try {
    const data = await fetchWithRetry(url);
    return data;
  } catch (error: any) {
    console.error(`    [error] Failed to fetch events for match ${matchId}: ${error.message}`);
    return [];
  }
}

/**
 * Aggregate shot events into match-level xG
 */
function aggregateXG(events: SBEvent[], homeTeamId: number, awayTeamId: number) {
  let homeXG = 0;
  let awayXG = 0;

  for (const event of events) {
    if (event.type.name === 'Shot' && event.shot) {
      const xG = event.shot.statsbomb_xg;
      if (event.team.id === homeTeamId) {
        homeXG += xG;
      } else if (event.team.id === awayTeamId) {
        awayXG += xG;
      }
    }
  }

  return { homeXG, awayXG };
}

/**
 * Convert StatsBomb match to UnderstatMatch format for compatibility
 */
async function convertMatch(
  sbMatch: SBMatch,
  leagueCode: LeagueCode,
): Promise<StatsBombMatch | null> {
  if (sbMatch.status !== 'Complete') {
    return null; // Skip unplayed matches
  }

  // Fetch events to get xG data
  const events = await fetchMatchEvents(sbMatch.id);
  const { homeXG, awayXG } = aggregateXG(events, sbMatch.home_team.id, sbMatch.away_team.id);

  // Rate limiting: respect API
  await new Promise((r) => setTimeout(r, 50));

  return {
    id: Number(sbMatch.id),
    isResult: true,
    homeTeam: sbMatch.home_team.name,
    awayTeam: sbMatch.away_team.name,
    homeGoals: sbMatch.home_score,
    awayGoals: sbMatch.away_score,
    homeXG,
    awayXG,
    date: `${sbMatch.match_date} ${sbMatch.kick_off}`,
    league: leagueCode,
    season: sbMatch.competition.season_name,
  };
}

/**
 * Scrape all matches for a league+season from StatsBomb
 */
export async function scrapeLeagueSeason(
  leagueCode: LeagueCode,
  seasonStartYear: number,
): Promise<StatsBombMatch[]> {
  const sbCompId = COMPETITION_MAPPING[leagueCode];
  const sbSeasonId = STATSBOMB_SEASONS[seasonStartYear];

  if (!sbCompId || !sbSeasonId) {
    console.log(`  [skip] No StatsBomb mapping for ${leagueCode} ${seasonStartYear}`);
    return [];
  }

  const cacheFile = join(DATA_RAW_DIR, `statsbomb_${leagueCode}_${seasonStartYear}.json`);

  // Check cache
  try {
    await stat(cacheFile);
    const cached = await readFile(cacheFile, 'utf-8');
    const data = JSON.parse(cached);
    console.log(`  [cache] ${leagueCode} ${seasonStartYear}: ${data.length} matches`);
    return data;
  } catch {
    // Not cached
  }

  console.log(`  [scrape] ${leagueCode} ${seasonStartYear}`);

  try {
    const sbMatches = await fetchMatches(sbCompId, sbSeasonId);

    if (sbMatches.length === 0) {
      console.log(`  [warn] No matches found for ${leagueCode} ${seasonStartYear}`);
      return [];
    }

    const matches: StatsBombMatch[] = [];
    for (const sbMatch of sbMatches) {
      const match = await convertMatch(sbMatch, leagueCode);
      if (match) {
        matches.push(match);
      }
    }

    if (matches.length > 0) {
      await mkdir(dirname(cacheFile), { recursive: true });
      await writeFile(cacheFile, JSON.stringify(matches, null, 2), 'utf-8');
      console.log(`  [ok] ${leagueCode} ${seasonStartYear}: ${matches.length} matches with xG`);
    } else {
      console.log(`  [warn] No completed matches for ${leagueCode} ${seasonStartYear}`);
    }

    return matches;
  } catch (error: any) {
    console.error(`  [error] ${leagueCode} ${seasonStartYear}: ${error.message}`);
    return [];
  }
}

/**
 * Scrape all leagues and seasons
 */
export async function scrapeAll(
  leagues?: LeagueCode[],
  startYear: number = 2015,
  endYear: number = 2024,
): Promise<StatsBombMatch[]> {
  const targetLeagues = (leagues ||
    (Object.keys(COMPETITION_MAPPING) as LeagueCode[])) as LeagueCode[];
  const allMatches: StatsBombMatch[] = [];

  for (const leagueCode of targetLeagues) {
    console.log(`\n--- ${leagueCode} ---`);

    for (let year = startYear; year <= endYear; year++) {
      const matches = await scrapeLeagueSeason(leagueCode, year);
      allMatches.push(...matches);
    }
  }

  console.log(`\nTotal: ${allMatches.length} matches with xG data`);
  return allMatches;
}

/**
 * Load cached StatsBomb data (no scraping)
 */
export async function loadCachedXG(leagues?: LeagueCode[]): Promise<StatsBombMatch[]> {
  const targetLeagues = (leagues ||
    (Object.keys(COMPETITION_MAPPING) as LeagueCode[])) as LeagueCode[];
  const allMatches: StatsBombMatch[] = [];

  for (const leagueCode of targetLeagues) {
    for (let year = 2015; year <= 2024; year++) {
      const cacheFile = join(DATA_RAW_DIR, `statsbomb_${leagueCode}_${year}.json`);
      try {
        const content = await readFile(cacheFile, 'utf-8');
        const matches: StatsBombMatch[] = JSON.parse(content);
        allMatches.push(...matches);
      } catch {
        // Not cached
      }
    }
  }

  return allMatches;
}
