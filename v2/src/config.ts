/**
 * BetAI v2 - Configuration
 */

import type { LeagueConfig, LeagueCode } from './types/index.js';
import { isPrimaryLeague, PRIMARY_LEAGUES } from './types/index.js';

export const LEAGUES: Record<LeagueCode, LeagueConfig> = {
  // ─── Tier 1: Top 5 European leagues (primary, full feature coverage) ───
  EPL: {
    code: 'EPL',
    csvCode: 'E0',
    name: 'Premier League',
    country: 'England',
    understatSlug: 'EPL',
  },
  SERIE_A: {
    code: 'SERIE_A',
    csvCode: 'I1',
    name: 'Serie A',
    country: 'Italy',
    understatSlug: 'Serie_A',
  },
  LA_LIGA: {
    code: 'LA_LIGA',
    csvCode: 'SP1',
    name: 'La Liga',
    country: 'Spain',
    understatSlug: 'La_liga',
  },
  BUNDESLIGA: {
    code: 'BUNDESLIGA',
    csvCode: 'D1',
    name: 'Bundesliga',
    country: 'Germany',
    understatSlug: 'Bundesliga',
  },
  LIGUE_1: {
    code: 'LIGUE_1',
    csvCode: 'F1',
    name: 'Ligue 1',
    country: 'France',
    understatSlug: 'Ligue_1',
  },
  // ─── Tier 2: Second divisions of top 5 ───
  ENG_CHAMP: {
    code: 'ENG_CHAMP',
    csvCode: 'E1',
    name: 'Championship',
    country: 'England',
  },
  SERIE_B: {
    code: 'SERIE_B',
    csvCode: 'I2',
    name: 'Serie B',
    country: 'Italy',
  },
  LA_LIGA_2: {
    code: 'LA_LIGA_2',
    csvCode: 'SP2',
    name: 'La Liga 2',
    country: 'Spain',
  },
  BUNDESLIGA_2: {
    code: 'BUNDESLIGA_2',
    csvCode: 'D2',
    name: '2. Bundesliga',
    country: 'Germany',
  },
  LIGUE_2: {
    code: 'LIGUE_2',
    csvCode: 'F2',
    name: 'Ligue 2',
    country: 'France',
  },
  // ─── Tier 2: Other top European leagues ───
  EREDIVISIE: {
    code: 'EREDIVISIE',
    csvCode: 'N1',
    name: 'Eredivisie',
    country: 'Netherlands',
  },
  JUPILER: {
    code: 'JUPILER',
    csvCode: 'B1',
    name: 'Jupiler Pro League',
    country: 'Belgium',
  },
  PRIMEIRA: {
    code: 'PRIMEIRA',
    csvCode: 'P1',
    name: 'Primeira Liga',
    country: 'Portugal',
  },
  SUPER_LIG: {
    code: 'SUPER_LIG',
    csvCode: 'T1',
    name: 'Super Lig',
    country: 'Turkey',
  },
  SUPER_LEAGUE_GR: {
    code: 'SUPER_LEAGUE_GR',
    csvCode: 'G1',
    name: 'Super League',
    country: 'Greece',
  },
  // ─── Tier 3: Lower English + Scottish ───
  ENG_L1: {
    code: 'ENG_L1',
    csvCode: 'E2',
    name: 'League One',
    country: 'England',
  },
  ENG_L2: {
    code: 'ENG_L2',
    csvCode: 'E3',
    name: 'League Two',
    country: 'England',
  },
  SCOT_PREM: {
    code: 'SCOT_PREM',
    csvCode: 'SC0',
    name: 'Scottish Premiership',
    country: 'Scotland',
  },
  SCOT_CHAMP: {
    code: 'SCOT_CHAMP',
    csvCode: 'SC1',
    name: 'Scottish Championship',
    country: 'Scotland',
  },
  SCOT_L1: {
    code: 'SCOT_L1',
    csvCode: 'SC2',
    name: 'Scottish League One',
    country: 'Scotland',
  },
  SCOT_L2: {
    code: 'SCOT_L2',
    csvCode: 'SC3',
    name: 'Scottish League Two',
    country: 'Scotland',
  },
};

/**
 * football-data.co.uk URL patterns:
 * Current season: https://www.football-data.co.uk/mmz4281/{season_code}/{div}.csv
 * Archive: same pattern
 *
 * Season codes: 2425 = 2024-25, 2324 = 2023-24, etc.
 */
export function getFootballDataURL(csvCode: string, seasonCode: string): string {
  return `https://www.football-data.co.uk/mmz4281/${seasonCode}/${csvCode}.csv`;
}

/**
 * Generate season code for football-data.co.uk
 * e.g. 2024 -> "2425" (for 2024-25 season)
 */
export function getSeasonCode(startYear: number): string {
  const endYear = startYear + 1;
  return `${String(startYear).slice(2)}${String(endYear).slice(2)}`;
}

/**
 * Get all seasons to download.
 * Default: 2010-11 to 2024-25 (15 seasons) for maximum segmentation power.
 * Use startFrom parameter to limit (e.g. 2015 for primary-only analysis).
 */
export function getTargetSeasons(startFrom: number = 2010): { label: string; code: string; startYear: number }[] {
  const seasons: { label: string; code: string; startYear: number }[] = [];
  for (let year = startFrom; year <= 2024; year++) {
    seasons.push({
      label: `${year}-${String(year + 1).slice(2)}`,
      code: getSeasonCode(year),
      startYear: year,
    });
  }
  return seasons;
}

/** League tier classification for segmentation */
export type LeagueTier = 1 | 2 | 3;

export function getLeagueTier(code: LeagueCode): LeagueTier {
  if ((PRIMARY_LEAGUES as readonly string[]).includes(code)) return 1;
  const tier2 = [
    'ENG_CHAMP', 'SERIE_B', 'LA_LIGA_2', 'BUNDESLIGA_2', 'LIGUE_2',
    'EREDIVISIE', 'JUPILER', 'PRIMEIRA', 'SUPER_LIG', 'SUPER_LEAGUE_GR',
  ];
  if (tier2.includes(code)) return 2;
  return 3;
}

// ─── Archive League Map ───
// Maps archive Division codes to LeagueCode + metadata

export interface ArchiveLeagueInfo {
  leagueCode: LeagueCode;
  name: string;
  country: string;
  tier: number; // 1 = top division, 2 = second, etc.
}

export const ARCHIVE_LEAGUE_MAP: Record<string, ArchiveLeagueInfo> = {
  // England
  'E0':  { leagueCode: 'EPL',           name: 'Premier League',     country: 'England',     tier: 1 },
  'E1':  { leagueCode: 'ENG_CHAMP',     name: 'Championship',       country: 'England',     tier: 2 },
  'E2':  { leagueCode: 'ENG_L1',        name: 'League One',         country: 'England',     tier: 3 },
  'E3':  { leagueCode: 'ENG_L2',        name: 'League Two',         country: 'England',     tier: 4 },
  'EC':  { leagueCode: 'ENG_CONF',      name: 'Conference',         country: 'England',     tier: 5 },
  // Scotland
  'SC0': { leagueCode: 'SCOT_PREM',     name: 'Scottish Premiership', country: 'Scotland',  tier: 1 },
  'SC1': { leagueCode: 'SCOT_CHAMP',    name: 'Scottish Championship', country: 'Scotland', tier: 2 },
  'SC2': { leagueCode: 'SCOT_L1',       name: 'Scottish League One', country: 'Scotland',   tier: 3 },
  'SC3': { leagueCode: 'SCOT_L2',       name: 'Scottish League Two', country: 'Scotland',   tier: 4 },
  // Germany
  'D1':  { leagueCode: 'BUNDESLIGA',    name: 'Bundesliga',         country: 'Germany',     tier: 1 },
  'D2':  { leagueCode: 'BUNDESLIGA_2',  name: '2. Bundesliga',      country: 'Germany',     tier: 2 },
  // Italy
  'I1':  { leagueCode: 'SERIE_A',       name: 'Serie A',            country: 'Italy',       tier: 1 },
  'I2':  { leagueCode: 'SERIE_B',       name: 'Serie B',            country: 'Italy',       tier: 2 },
  // Spain
  'SP1': { leagueCode: 'LA_LIGA',       name: 'La Liga',            country: 'Spain',       tier: 1 },
  'SP2': { leagueCode: 'LA_LIGA_2',     name: 'La Liga 2',          country: 'Spain',       tier: 2 },
  // France
  'F1':  { leagueCode: 'LIGUE_1',       name: 'Ligue 1',            country: 'France',      tier: 1 },
  'F2':  { leagueCode: 'LIGUE_2',       name: 'Ligue 2',            country: 'France',      tier: 2 },
  // Netherlands
  'N1':  { leagueCode: 'EREDIVISIE',    name: 'Eredivisie',         country: 'Netherlands', tier: 1 },
  // Belgium
  'B1':  { leagueCode: 'JUPILER',       name: 'Jupiler Pro League',  country: 'Belgium',    tier: 1 },
  // Portugal
  'P1':  { leagueCode: 'PRIMEIRA',      name: 'Primeira Liga',       country: 'Portugal',   tier: 1 },
  // Turkey
  'T1':  { leagueCode: 'SUPER_LIG',     name: 'Super Lig',           country: 'Turkey',     tier: 1 },
  // Greece
  'G1':  { leagueCode: 'SUPER_LEAGUE_GR', name: 'Super League',      country: 'Greece',     tier: 1 },
  // South America
  'ARG': { leagueCode: 'ARGENTINA',     name: 'Primera Division',    country: 'Argentina',  tier: 1 },
  'BRA': { leagueCode: 'BRASILEIRAO',   name: 'Brasileirao',         country: 'Brazil',     tier: 1 },
  // North America
  'USA': { leagueCode: 'MLS',           name: 'MLS',                 country: 'USA',        tier: 1 },
  'MEX': { leagueCode: 'LIGA_MX',       name: 'Liga MX',             country: 'Mexico',     tier: 1 },
  // Asia
  'JAP': { leagueCode: 'J_LEAGUE',      name: 'J-League',            country: 'Japan',      tier: 1 },
  'CHN': { leagueCode: 'CSL',           name: 'Chinese Super League', country: 'China',     tier: 1 },
  // Nordics
  'SWE': { leagueCode: 'ALLSVENSKAN',   name: 'Allsvenskan',         country: 'Sweden',     tier: 1 },
  'NOR': { leagueCode: 'ELITESERIEN',   name: 'Eliteserien',         country: 'Norway',     tier: 1 },
  'DEN': { leagueCode: 'SUPERLIGA_DK',  name: 'Superliga',           country: 'Denmark',    tier: 1 },
  'FIN': { leagueCode: 'VEIKKAUSLIIGA', name: 'Veikkausliiga',       country: 'Finland',    tier: 1 },
  // Eastern Europe
  'POL': { leagueCode: 'EKSTRAKLASA',   name: 'Ekstraklasa',         country: 'Poland',     tier: 1 },
  'ROM': { leagueCode: 'LIGA_1_RO',     name: 'Liga 1',              country: 'Romania',    tier: 1 },
  'RUS': { leagueCode: 'RPL',           name: 'Russian Premier',     country: 'Russia',     tier: 1 },
  // Other
  'AUT': { leagueCode: 'BUNDESLIGA_AT', name: 'Bundesliga (AT)',     country: 'Austria',    tier: 1 },
  'SUI': { leagueCode: 'SUPER_LEAGUE_CH', name: 'Super League',      country: 'Switzerland', tier: 1 },
  'IRL': { leagueCode: 'LOI',           name: 'League of Ireland',   country: 'Ireland',    tier: 1 },
};

/** Reverse map: LeagueCode -> archive Division code */
export const LEAGUE_TO_DIV: Record<string, string> = Object.fromEntries(
  Object.entries(ARCHIVE_LEAGUE_MAP).map(([div, cfg]) => [cfg.leagueCode, div]),
);

/** Country encoding for features (ordinal) */
export const COUNTRY_ENCODING: Record<string, number> = {
  'England': 0, 'Italy': 1, 'Spain': 2, 'Germany': 3, 'France': 4,
  'Netherlands': 5, 'Belgium': 6, 'Portugal': 7, 'Turkey': 8,
  'Greece': 9, 'Scotland': 10, 'Argentina': 11, 'Brazil': 12,
  'USA': 13, 'Mexico': 14, 'Japan': 15, 'China': 16,
  'Sweden': 17, 'Norway': 18, 'Denmark': 19, 'Finland': 20,
  'Poland': 21, 'Romania': 22, 'Russia': 23, 'Austria': 24,
  'Switzerland': 25, 'Ireland': 26,
};

export { isPrimaryLeague };

// Betting config
export const BETTING_CONFIG = {
  kellyFraction: 0.25,       // Quarter Kelly (conservative)
  minEdge: 0.05,             // Minimum 5% edge to bet
  maxStakePercent: 0.05,     // Max 5% of bankroll per bet
  initialBankroll: 1000,     // Starting bankroll for backtest
};

// The Odds API (free tier)
export const ODDS_API_KEY = process.env.ODDS_API_KEY || '';
export const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';
