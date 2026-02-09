/**
 * Understat xG Scraper
 *
 * Scrapes expected goals (xG) data from understat.com.
 * Understat provides match-level xG for top 5 European leagues.
 *
 * Data is embedded in JavaScript on the page, parsed from JSON.
 * Seasons available: 2014-15 onwards.
 */

import axios from 'axios';
import * as cheerio from 'cheerio';
import { writeFile, mkdir, readFile, stat } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { LEAGUES } from '../../src/config.js';
import type { LeagueCode } from '../../src/types/index.js';

const __dirname = join(fileURLToPath(import.meta.url), '..');
const DATA_RAW_DIR = join(__dirname, '..', 'raw');

// Retry configuration
const RETRY_ATTEMPTS = 3;
const RETRY_DELAY_MS = 2000;

export interface UnderstatMatch {
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

/**
 * Retry logic with exponential backoff
 */
async function fetchWithRetry(url: string, maxAttempts = RETRY_ATTEMPTS): Promise<string> {
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
      const isNetworkError = error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT';
      
      if (!isLastAttempt && isNetworkError) {
        const delayMs = RETRY_DELAY_MS * Math.pow(2, attempt - 1);
        console.log(`    [retry ${attempt}/${maxAttempts}] Network error: ${error.code} - waiting ${delayMs}ms...`);
        await new Promise(r => setTimeout(r, delayMs));
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
 * Scrape all match xG data for a league+season from Understat
 */
export async function scrapeLeagueSeason(
  leagueCode: LeagueCode,
  seasonStartYear: number,
): Promise<UnderstatMatch[]> {
  const league = LEAGUES[leagueCode];
  if (!league.understatSlug) {
    console.log(`  [skip] No Understat slug for ${league.name}`);
    return [];
  }

  const cacheFile = join(DATA_RAW_DIR, `understat_${leagueCode}_${seasonStartYear}.json`);

  // Check cache
  try {
    await stat(cacheFile);
    const cached = await readFile(cacheFile, 'utf-8');
    const data = JSON.parse(cached);
    console.log(`  [cache] ${league.name} ${seasonStartYear}: ${data.length} matches`);
    return data;
  } catch {
    // Not cached
  }

  const url = `https://understat.com/league/${league.understatSlug}/${seasonStartYear}`;
  console.log(`  [scrape] ${league.name} ${seasonStartYear} -> ${url}`);

  try {
    const html = await fetchWithRetry(url);
    const matches = extractMatchData(html, leagueCode, seasonStartYear);

    if (matches.length > 0) {
      await mkdir(dirname(cacheFile), { recursive: true });
      await writeFile(cacheFile, JSON.stringify(matches, null, 2), 'utf-8');
      console.log(`  [ok] ${league.name} ${seasonStartYear}: ${matches.length} matches`);
    } else {
      console.log(`  [warn] No data found for ${league.name} ${seasonStartYear}`);
    }

    return matches;
  } catch (error: any) {
    console.error(`  [error] ${league.name} ${seasonStartYear}: ${error.message}`);
    return [];
  }
}

/**
 * Extract match data from Understat HTML page.
 * Data is in a JS variable: var datesData = JSON.parse('...');
 */
function extractMatchData(
  html: string,
  leagueCode: LeagueCode,
  seasonStartYear: number,
): UnderstatMatch[] {
  // Understat embeds data in script tags as encoded JSON
  const scriptRegex = /var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)/;
  const match = scriptRegex.exec(html);

  if (!match) {
    console.log('  [warn] Could not find datesData in HTML');
    return [];
  }

  // Decode escaped unicode
  const encoded = match[1];
  const decoded = encoded.replace(/\\x([0-9A-Fa-f]{2})/g, (_, hex) =>
    String.fromCharCode(parseInt(hex, 16))
  );

  try {
    const data = JSON.parse(decoded);
    const matches: UnderstatMatch[] = [];

    // data is an object keyed by date
    for (const dateKey of Object.keys(data)) {
      const dayMatches = data[dateKey];
      for (const m of dayMatches) {
        if (!m.isResult) continue; // Skip unplayed

        matches.push({
          id: Number(m.id),
          isResult: true,
          homeTeam: m.h?.title || m.h?.short_title || '',
          awayTeam: m.a?.title || m.a?.short_title || '',
          homeGoals: Number(m.goals?.h ?? m.h?.goals ?? 0),
          awayGoals: Number(m.goals?.a ?? m.a?.goals ?? 0),
          homeXG: parseFloat(m.xG?.h ?? m.h?.xG ?? '0'),
          awayXG: parseFloat(m.xG?.a ?? m.a?.xG ?? '0'),
          date: m.datetime || dateKey,
          league: leagueCode,
          season: `${seasonStartYear}-${String(seasonStartYear + 1).slice(2)}`,
        });
      }
    }

    return matches;
  } catch (err: any) {
    console.log(`  [error] JSON parse failed: ${err.message}`);
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
): Promise<UnderstatMatch[]> {
  const targetLeagues = leagues || (Object.keys(LEAGUES) as LeagueCode[]);
  const allMatches: UnderstatMatch[] = [];

  for (const leagueCode of targetLeagues) {
    console.log(`\n--- ${LEAGUES[leagueCode].name} ---`);

    for (let year = startYear; year <= endYear; year++) {
      const matches = await scrapeLeagueSeason(leagueCode, year);
      allMatches.push(...matches);

      // Rate limiting: 2s between requests (be respectful)
      await new Promise(r => setTimeout(r, 2000));
    }
  }

  console.log(`\nTotal: ${allMatches.length} matches with xG data`);
  return allMatches;
}

/**
 * Load cached Understat data (no scraping)
 */
export async function loadCachedXG(
  leagues?: LeagueCode[],
): Promise<UnderstatMatch[]> {
  const targetLeagues = leagues || (Object.keys(LEAGUES) as LeagueCode[]);
  const allMatches: UnderstatMatch[] = [];

  for (const leagueCode of targetLeagues) {
    for (let year = 2015; year <= 2024; year++) {
      const cacheFile = join(DATA_RAW_DIR, `understat_${leagueCode}_${year}.json`);
      try {
        const content = await readFile(cacheFile, 'utf-8');
        const matches: UnderstatMatch[] = JSON.parse(content);
        allMatches.push(...matches);
      } catch {
        // Not cached
      }
    }
  }

  return allMatches;
}
