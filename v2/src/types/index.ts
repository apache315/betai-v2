/**
 * BetAI v2 - Core Types
 */

// ─── Raw Match Data (from football-data.co.uk CSV) ───

export interface RawMatchCSV {
  Div: string;         // League code (E0, I1, SP1, D1, F1)
  Date: string;        // DD/MM/YYYY or DD/MM/YY
  Time?: string;
  HomeTeam: string;
  AwayTeam: string;
  FTHG: number;        // Full Time Home Goals
  FTAG: number;        // Full Time Away Goals
  FTR: 'H' | 'D' | 'A'; // Full Time Result
  HTHG?: number;       // Half Time Home Goals
  HTAG?: number;       // Half Time Away Goals
  HTR?: string;        // Half Time Result
  HS?: number;         // Home Shots
  AS?: number;         // Away Shots
  HST?: number;        // Home Shots on Target
  AST?: number;        // Away Shots on Target
  HF?: number;         // Home Fouls
  AF?: number;         // Away Fouls
  HC?: number;         // Home Corners
  AC?: number;         // Away Corners
  HY?: number;         // Home Yellow Cards
  AY?: number;         // Away Yellow Cards
  HR?: number;         // Home Red Cards
  AR?: number;         // Away Red Cards
  // Betting odds (pre-match / opening snapshot)
  B365H?: number;      // Bet365 Home
  B365D?: number;      // Bet365 Draw
  B365A?: number;      // Bet365 Away
  PSH?: number;        // Pinnacle Home
  PSD?: number;        // Pinnacle Draw
  PSA?: number;        // Pinnacle Away
  WHH?: number;        // William Hill Home
  WHD?: number;        // William Hill Draw
  WHA?: number;        // William Hill Away
  BWH?: number;        // Bet&Win Home
  BWD?: number;        // Bet&Win Draw
  BWA?: number;        // Bet&Win Away
  IWH?: number;        // Interwetten Home
  IWD?: number;        // Interwetten Draw
  IWA?: number;        // Interwetten Away
  // Closing odds (suffix C, available from 2019/20+)
  B365CH?: number;     // Bet365 Closing Home
  B365CD?: number;     // Bet365 Closing Draw
  B365CA?: number;     // Bet365 Closing Away
  PSCH?: number;       // Pinnacle Closing Home
  PSCD?: number;       // Pinnacle Closing Draw
  PSCA?: number;       // Pinnacle Closing Away
  // Market average odds
  AvgH?: number;       // Average Home
  AvgD?: number;       // Average Draw
  AvgA?: number;       // Average Away
  AvgCH?: number;      // Closing Average Home
  AvgCD?: number;      // Closing Average Draw
  AvgCA?: number;      // Closing Average Away
  'Avg>2.5'?: number;  // Average Over 2.5
  'Avg<2.5'?: number;  // Average Under 2.5
  // Max odds
  MaxH?: number;
  MaxD?: number;
  MaxA?: number;
  MaxCH?: number;      // Closing Max Home
  MaxCD?: number;      // Closing Max Draw
  MaxCA?: number;      // Closing Max Away
}

// ─── Normalized Match ───

export interface Match {
  id: string;          // `${league}_${date}_${home}_${away}`
  league: LeagueCode;
  season: string;      // e.g. "2023-24"
  date: Date;
  homeTeam: string;
  awayTeam: string;
  // Result
  ftHomeGoals: number;
  ftAwayGoals: number;
  ftResult: 'H' | 'D' | 'A';
  htHomeGoals?: number;
  htAwayGoals?: number;
  // Stats
  homeShots?: number;
  awayShots?: number;
  homeShotsOnTarget?: number;
  awayShotsOnTarget?: number;
  homeCorners?: number;
  awayCorners?: number;
  homeYellow?: number;
  awayYellow?: number;
  homeRed?: number;
  awayRed?: number;
  // Best available odds (Pinnacle > Bet365 > Average)
  oddsHome?: number;
  oddsDraw?: number;
  oddsAway?: number;
  oddsOver25?: number;
  oddsUnder25?: number;
  // Closing odds (best available closing, for CLV calculation)
  closingOddsHome?: number;
  closingOddsDraw?: number;
  closingOddsAway?: number;
  // Per-bookmaker odds (for segmentation & bias analysis)
  oddsByBookmaker?: Record<string, { home: number; draw: number; away: number }>;
  // Market average & max
  avgOddsHome?: number;
  avgOddsDraw?: number;
  avgOddsAway?: number;
  maxOddsHome?: number;
  maxOddsDraw?: number;
  maxOddsAway?: number;
  // xG (from Understat, if available)
  homeXG?: number;
  awayXG?: number;
  // Archive Elo ratings (optional, from archive dataset)
  archiveHomeElo?: number;
  archiveAwayElo?: number;
  // Data source marker for deduplication
  source?: 'football-data' | 'archive';
}

// ─── Leagues ───

/** All supported league codes (string to support 38+ archive leagues) */
export type LeagueCode = string;

/** The 5 primary leagues with full data coverage (Pinnacle odds, xG, fatigue coords) */
export const PRIMARY_LEAGUES = ['EPL', 'SERIE_A', 'LA_LIGA', 'BUNDESLIGA', 'LIGUE_1'] as const;
export type PrimaryLeagueCode = typeof PRIMARY_LEAGUES[number];

export function isPrimaryLeague(code: LeagueCode): code is PrimaryLeagueCode {
  return (PRIMARY_LEAGUES as readonly string[]).includes(code);
}

export interface LeagueConfig {
  code: LeagueCode;
  csvCode: string;       // football-data.co.uk code (E0, I1, SP1, D1, F1)
  name: string;
  country: string;
  understatSlug?: string; // Understat league slug
}

// ─── Features ───

export interface MatchFeatures {
  matchId: string;
  league: LeagueCode;
  season: string;
  date: Date;
  homeTeam: string;
  awayTeam: string;
  // Target
  result: 'H' | 'D' | 'A';
  totalGoals: number;
  btts: boolean;
  // Features (all computed from PAST data only)
  features: Record<string, number>;
  // Odds (for value bet detection, NOT as training feature by default)
  closingOdds?: {
    home: number;
    draw: number;
    away: number;
  };
}

// ─── Prediction ───

export interface Prediction {
  matchId: string;
  homeTeam: string;
  awayTeam: string;
  league: LeagueCode;
  date: Date;
  probabilities: {
    home: number;
    draw: number;
    away: number;
  };
  confidence: number; // 0-1
  valueBets: ValueBet[];
}

export interface ValueBet {
  market: string;       // '1X2_H', '1X2_D', '1X2_A', 'O2.5', 'U2.5', 'BTTS_Y', 'BTTS_N'
  selection: string;    // Human readable
  modelProb: number;
  marketOdds: number;
  impliedProb: number;
  edge: number;         // modelProb - impliedProb
  value: number;        // (modelProb * marketOdds) - 1
  kellyFraction: number;
  recommendedStake: number; // % of bankroll (fractional Kelly)
}

// ─── Backtest ───

export interface BacktestResult {
  period: string;
  totalMatches: number;
  totalBets: number;
  accuracy1X2: number;
  brierScore: number;
  roi: number;
  profit: number;
  maxDrawdown: number;
  sharpeRatio: number;
  calibration: { predicted: number; actual: number }[];
  betsByMarket: Record<string, { count: number; won: number; roi: number }>;
}
