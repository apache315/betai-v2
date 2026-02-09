/**
 * The Odds API Client (Free Tier)
 *
 * https://the-odds-api.com/
 * Free tier: 500 requests/month
 *
 * Used ONLY for live/upcoming match odds (not backtest).
 * Backtest uses historical odds from football-data.co.uk CSVs.
 */

import axios from 'axios';
import { ODDS_API_KEY, ODDS_API_BASE } from '../../src/config.js';

export interface OddsApiMatch {
  id: string;
  sportKey: string;
  commenceTime: string; // ISO datetime
  homeTeam: string;
  awayTeam: string;
  bookmakers: OddsBookmaker[];
}

export interface OddsBookmaker {
  key: string;
  title: string;
  markets: OddsMarket[];
}

export interface OddsMarket {
  key: string; // 'h2h', 'spreads', 'totals'
  outcomes: { name: string; price: number; point?: number }[];
}

export interface NormalizedOdds {
  homeTeam: string;
  awayTeam: string;
  commenceTime: string;
  h2h: { home: number; draw: number; away: number } | null;
  totals25: { over: number; under: number } | null;
  source: string; // bookmaker key
}

// Sport keys for The Odds API
const LEAGUE_SPORT_KEYS: Record<string, string> = {
  EPL: 'soccer_epl',
  SERIE_A: 'soccer_italy_serie_a',
  LA_LIGA: 'soccer_spain_la_liga',
  BUNDESLIGA: 'soccer_germany_bundesliga',
  LIGUE_1: 'soccer_france_ligue_one',
};

/**
 * Fetch upcoming odds for a league
 */
export async function fetchUpcomingOdds(
  leagueCode: string,
): Promise<NormalizedOdds[]> {
  if (!ODDS_API_KEY) {
    console.warn('[odds-api] No API key set (ODDS_API_KEY). Skipping.');
    return [];
  }

  const sportKey = LEAGUE_SPORT_KEYS[leagueCode];
  if (!sportKey) {
    console.warn(`[odds-api] Unknown league: ${leagueCode}`);
    return [];
  }

  try {
    const response = await axios.get<OddsApiMatch[]>(
      `${ODDS_API_BASE}/sports/${sportKey}/odds`,
      {
        params: {
          apiKey: ODDS_API_KEY,
          regions: 'eu',
          markets: 'h2h,totals',
          oddsFormat: 'decimal',
        },
        timeout: 15000,
      },
    );

    console.log(
      `[odds-api] ${leagueCode}: ${response.data.length} matches, ` +
      `remaining: ${response.headers['x-requests-remaining'] || '?'}`
    );

    return response.data.map(match => normalizeOdds(match));
  } catch (error: any) {
    console.error(`[odds-api] Error: ${error.message}`);
    return [];
  }
}

/**
 * Normalize odds from The Odds API response.
 * Prefers Pinnacle > Bet365 > first available bookmaker.
 */
function normalizeOdds(match: OddsApiMatch): NormalizedOdds {
  const preferred = ['pinnacle', 'betfair_ex_eu', 'bet365_eu', 'unibet_eu'];
  const bookmakers = match.bookmakers;

  // Find best bookmaker
  let bestBook = bookmakers[0];
  for (const pref of preferred) {
    const found = bookmakers.find(b => b.key === pref);
    if (found) { bestBook = found; break; }
  }

  let h2h: { home: number; draw: number; away: number } | null = null;
  let totals25: { over: number; under: number } | null = null;

  if (bestBook) {
    // H2H market
    const h2hMarket = bestBook.markets.find(m => m.key === 'h2h');
    if (h2hMarket) {
      const homeOutcome = h2hMarket.outcomes.find(o => o.name === match.homeTeam);
      const drawOutcome = h2hMarket.outcomes.find(o => o.name === 'Draw');
      const awayOutcome = h2hMarket.outcomes.find(o => o.name === match.awayTeam);
      if (homeOutcome && drawOutcome && awayOutcome) {
        h2h = {
          home: homeOutcome.price,
          draw: drawOutcome.price,
          away: awayOutcome.price,
        };
      }
    }

    // Totals market (Over/Under 2.5)
    const totalsMarket = bestBook.markets.find(m => m.key === 'totals');
    if (totalsMarket) {
      const over = totalsMarket.outcomes.find(o => o.name === 'Over' && o.point === 2.5);
      const under = totalsMarket.outcomes.find(o => o.name === 'Under' && o.point === 2.5);
      if (over && under) {
        totals25 = { over: over.price, under: under.price };
      }
    }
  }

  return {
    homeTeam: match.homeTeam,
    awayTeam: match.awayTeam,
    commenceTime: match.commenceTime,
    h2h,
    totals25,
    source: bestBook?.key || 'unknown',
  };
}

/**
 * Check remaining API quota
 */
export async function checkQuota(): Promise<{ remaining: number; used: number } | null> {
  if (!ODDS_API_KEY) return null;

  try {
    const response = await axios.get(
      `${ODDS_API_BASE}/sports`,
      { params: { apiKey: ODDS_API_KEY }, timeout: 10000 },
    );
    return {
      remaining: Number(response.headers['x-requests-remaining'] || 0),
      used: Number(response.headers['x-requests-used'] || 0),
    };
  } catch {
    return null;
  }
}
