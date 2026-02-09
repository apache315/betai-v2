/**
 * FBRef (Sports Reference) xG Scraper
 * 
 * Public data: https://fbref.com
 * No login required, no API limit
 * Scrapes league seasons with xG data
 */

import axios from 'axios';
import * as cheerio from 'cheerio';
import { writeFile, mkdir } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { LEAGUES } from '../../src/config.js';
import type { LeagueCode } from '../../src/types/index.js';

const __dirname = join(fileURLToPath(import.meta.url), '..');
const DATA_RAW_DIR = join(__dirname, '..', 'raw');

// FBRef URLs by league and season
const FBREF_BASE = 'https://fbref.com/en/comps';

// League slug mapping: LeagueCode -> (FBRef comp ID, name)
const FBREF_LEAGUES: Record<string, { id: number; name: string }> = {
  E0: { id: 9, name: 'Premier League' },
  D1: { id: 20, name: 'Bundesliga' },
  F1: { id: 12, name: 'La Liga' },
  I1: { id: 11, name: 'Serie A' },
  SP1: { id: 23, name: 'Eredivisie' },
};

export interface FBRefMatch {
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
 * Fetch with retry
 */
async function fetchWithRetry(url: string, maxAttempts = 3): Promise<string> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const response = await axios.get(url, {
        timeout: 20000,
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        },
      });
      return response.data;
    } catch (error: any) {
      const isLastAttempt = attempt === maxAttempts;
      if (!isLastAttempt && (error.code === 'ENOTFOUND' || error.response?.status >= 500)) {
        const delay = 2000 * Math.pow(2, attempt - 1);
        console.log(`    [retry ${attempt}/${maxAttempts}] waiting ${delay}ms...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      if (isLastAttempt) throw error;
    }
  }
  throw new Error('All retries failed');
}

/**
 * Parse matches table from FBRef
 */
function parseMatches(html: string, leagueCode: LeagueCode, season: string): FBRefMatch[] {
  const $ = cheerio.load(html);
  const matches: FBRefMatch[] = [];
  let id = 1;

  // Find matches table - usually id="schedule"
  const rows = $('#schedule tbody tr');

  if (rows.length === 0) {
    console.log(`    [warn] No matches found in table`);
    return [];
  }

  rows.each((_, row) => {
    const cells = $(row).find('td');
    if (cells.length === 0) return; // Skip header rows

    try {
      const dateText = $(cells[0]).text().trim();
      const dayText = $(cells[1]).text().trim();
      const timeText = $(cells[2]).text().trim();
      const homeTeam = $(cells[3]).text().trim();
      const homeGoalsText = $(cells[4]).text().trim();
      const awayGoalsText = $(cells[5]).text().trim();
      const awayTeam = $(cells[6]).text().trim();
      const homeXGText = $(cells[7]).text().trim();
      const awayXGText = $(cells[8]).text().trim();

      // Skip if no xG data
      if (!homeXGText || !awayXGText) return;

      const homeGoals = parseInt(homeGoalsText, 10);
      const awayGoals = parseInt(awayGoalsText, 10);
      const homeXG = parseFloat(homeXGText);
      const awayXG = parseFloat(awayXGText);

      // Skip incomplete matches
      if (isNaN(homeGoals) || isNaN(awayGoals)) return;

      matches.push({
        id: id++,
        isResult: true,
        homeTeam,
        awayTeam,
        homeGoals,
        awayGoals,
        homeXG,
        awayXG,
        date: `${dateText} ${timeText || '00:00'}`,
        league: leagueCode,
        season,
      });
    } catch (e) {
      // Skip malformed rows
    }
  });

  return matches;
}

/**
 * Scrape one league season
 */
export async function scrapeLeagueSeason(
  leagueCode: LeagueCode,
  seasonStartYear: number,
): Promise<FBRefMatch[]> {
  const league = FBREF_LEAGUES[leagueCode];
  if (!league) {
    console.log(`  [skip] No FBRef mapping for ${leagueCode}`);
    return [];
  }

  const seasonEnd = seasonStartYear + 1;
  const url = `${FBREF_BASE}/${league.id}/${seasonEnd}/schedule/${seasonStartYear}-${seasonEnd}-${league.name.replace(/\s+/g, '-')}-Matches`;

  console.log(`  [scrape] ${leagueCode} ${seasonStartYear}-${seasonEnd}`);

  try {
    const html = await fetchWithRetry(url);
    const matches = parseMatches(html, leagueCode, `${seasonStartYear}-${String(seasonEnd).slice(2)}`);

    if (matches.length === 0) {
      console.log(`  [warn] No matches found for ${leagueCode} ${seasonStartYear}`);
      return [];
    }

    const cacheFile = join(DATA_RAW_DIR, `fbref_${leagueCode}_${seasonStartYear}.json`);
    await mkdir(dirname(cacheFile), { recursive: true });
    await writeFile(cacheFile, JSON.stringify(matches, null, 2), 'utf-8');
    console.log(`  [ok] ${leagueCode} ${seasonStartYear}: ${matches.length} matches`);

    // Rate limit: be respectful
    await new Promise((r) => setTimeout(r, 1000));

    return matches;
  } catch (error: any) {
    console.error(`  [error] ${leagueCode} ${seasonStartYear}: ${error.message}`);
    return [];
  }
}

/**
 * Scrape all
 */
export async function scrapeAll(
  leagues?: LeagueCode[],
  startYear: number = 2015,
  endYear: number = 2024,
): Promise<FBRefMatch[]> {
  const targetLeagues = (leagues || (Object.keys(FBREF_LEAGUES) as LeagueCode[])) as LeagueCode[];
  const allMatches: FBRefMatch[] = [];

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
 * Load cached
 */
export async function loadCachedXG(leagues?: LeagueCode[]): Promise<FBRefMatch[]> {
  const targetLeagues = (leagues || (Object.keys(FBREF_LEAGUES) as LeagueCode[])) as LeagueCode[];
  const allMatches: FBRefMatch[] = [];

  for (const leagueCode of targetLeagues) {
    for (let year = 2015; year <= 2024; year++) {
      const cacheFile = join(DATA_RAW_DIR, `fbref_${leagueCode}_${year}.json`);
      try {
        const { readFileSync } = await import('fs');
        const content = readFileSync(cacheFile, 'utf-8');
        const matches: FBRefMatch[] = JSON.parse(content);
        allMatches.push(...matches);
      } catch {
        // Not cached
      }
    }
  }

  return allMatches;
}
