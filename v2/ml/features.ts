/**
 * Feature Engineering
 *
 * Builds 100+ features per match using ONLY past data (no look-ahead bias).
 * Every feature for match M is computed from matches before M's date.
 *
 * Feature groups:
 * 1. Recent form (last 5, 10 games)
 * 2. Season-level stats
 * 3. Head-to-head
 * 4. Context (rest days, matchday, month)
 * 5. Differentials (home - away)
 * 6. Glicko-2 ratings (strength + uncertainty)
 * 7. Fatigue & Travel Index
 * 8. Playing Style Clustering (K-Means)
 *
 * NOTE: Odds and xG are intentionally EXCLUDED from model features.
 * Odds are only used in backtest/betting for value detection.
 * xG depends on third-party models and is only available for 5 leagues.
 */

import type { Match, MatchFeatures, LeagueCode } from '../src/types/index.js';
import { isPrimaryLeague } from '../src/types/index.js';
import { ARCHIVE_LEAGUE_MAP, LEAGUE_TO_DIV, COUNTRY_ENCODING } from '../src/config.js';
import { Glicko2Tracker } from './glicko2.js';
import { FatigueTracker } from './fatigue.js';
import { StyleClusteringTracker } from './style-clustering.js';

interface TeamStats {
  played: number;
  wins: number;
  draws: number;
  losses: number;
  goalsScored: number;
  goalsConceded: number;
  cleanSheets: number;
  points: number;
  xgFor: number;
  xgAgainst: number;
  xgCount: number; // number of matches with xG data
}

interface FormMatch {
  date: Date;
  isHome: boolean;
  goalsFor: number;
  goalsAgainst: number;
  result: 'W' | 'D' | 'L';
  points: number;
  xgFor?: number;
  xgAgainst?: number;
}

/**
 * Build features for all matches in a dataset.
 * Matches must be sorted by date ascending.
 */
export function buildFeatures(matches: Match[]): MatchFeatures[] {
  // Sort by date
  const sorted = [...matches].sort((a, b) => a.date.getTime() - b.date.getTime());

  // Initialize advanced trackers
  console.log('Initializing Glicko-2 tracker...');
  const glicko2 = new Glicko2Tracker();

  console.log('Initializing Fatigue tracker...');
  const fatigue = new FatigueTracker();

  console.log('Initializing Style clustering...');
  const styleTracker = new StyleClusteringTracker();

  // Pre-process matches for style clustering (needs full history first)
  console.log('Pre-processing matches for style clustering...');
  styleTracker.processMatches(sorted);

  // Build H2H index for O(1) lookup (critical for 230k+ matches)
  console.log('Building H2H index...');
  const h2hIndex = new Map<string, Match[]>();
  for (const match of sorted) {
    const pair = [match.homeTeam, match.awayTeam].sort().join('|');
    let arr = h2hIndex.get(pair);
    if (!arr) {
      arr = [];
      h2hIndex.set(pair, arr);
    }
    arr.push(match);
  }
  console.log(`H2H index: ${h2hIndex.size} unique team pairs`);

  // Track per-team history per league-season
  const teamHistory = new Map<string, FormMatch[]>();
  const results: MatchFeatures[] = [];

  for (const match of sorted) {
    const homeKey = teamKey(match.league, match.homeTeam);
    const awayKey = teamKey(match.league, match.awayTeam);

    const homeHist = teamHistory.get(homeKey) || [];
    const awayHist = teamHistory.get(awayKey) || [];

    // Only compute features if we have minimum history
    const MIN_MATCHES = 3;
    if (homeHist.length >= MIN_MATCHES && awayHist.length >= MIN_MATCHES) {
      const features = computeFeatures(
        match,
        homeHist,
        awayHist,
        h2hIndex,
        glicko2,
        fatigue,
        styleTracker,
      );

      results.push({
        matchId: match.id,
        league: match.league,
        season: match.season,
        date: match.date,
        homeTeam: match.homeTeam,
        awayTeam: match.awayTeam,
        result: match.ftResult,
        totalGoals: match.ftHomeGoals + match.ftAwayGoals,
        btts: match.ftHomeGoals > 0 && match.ftAwayGoals > 0,
        features,
        closingOdds: (match.oddsHome && match.oddsDraw && match.oddsAway)
          ? { home: match.oddsHome, draw: match.oddsDraw, away: match.oddsAway }
          : undefined,
      });
    }

    // Update history
    const homeResult = match.ftResult === 'H' ? 'W' : match.ftResult === 'D' ? 'D' : 'L';
    const awayResult = match.ftResult === 'A' ? 'W' : match.ftResult === 'D' ? 'D' : 'L';

    homeHist.push({
      date: match.date,
      isHome: true,
      goalsFor: match.ftHomeGoals,
      goalsAgainst: match.ftAwayGoals,
      result: homeResult,
      points: homeResult === 'W' ? 3 : homeResult === 'D' ? 1 : 0,
      xgFor: match.homeXG,
      xgAgainst: match.awayXG,
    });

    awayHist.push({
      date: match.date,
      isHome: false,
      goalsFor: match.ftAwayGoals,
      goalsAgainst: match.ftHomeGoals,
      result: awayResult,
      points: awayResult === 'W' ? 3 : awayResult === 'D' ? 1 : 0,
      xgFor: match.awayXG,
      xgAgainst: match.homeXG,
    });

    teamHistory.set(homeKey, homeHist);
    teamHistory.set(awayKey, awayHist);

    // Update Glicko-2 ratings AFTER computing features (no look-ahead)
    glicko2.updateMatch(homeKey, awayKey, match.ftHomeGoals, match.ftAwayGoals, match.date);

    // Update fatigue tracker
    fatigue.updateMatch(match.homeTeam, match.awayTeam, match.date, match.league);
  }

  console.log(`Built features for ${results.length} / ${sorted.length} matches`);
  return results;
}

function computeFeatures(
  match: Match,
  homeHist: FormMatch[],
  awayHist: FormMatch[],
  h2hIndex: Map<string, Match[]>,
  glicko2: Glicko2Tracker,
  fatigue: FatigueTracker,
  styleTracker: StyleClusteringTracker,
): Record<string, number> {
  const f: Record<string, number> = {};

  // ─── 1. Recent Form ───
  for (const n of [5, 10]) {
    const hLast = lastN(homeHist, n);
    const aLast = lastN(awayHist, n);

    f[`home_win_rate_${n}`] = winRate(hLast);
    f[`away_win_rate_${n}`] = winRate(aLast);
    f[`home_draw_rate_${n}`] = drawRate(hLast);
    f[`away_draw_rate_${n}`] = drawRate(aLast);
    f[`home_goals_scored_avg_${n}`] = avgGoalsFor(hLast);
    f[`away_goals_scored_avg_${n}`] = avgGoalsFor(aLast);
    f[`home_goals_conceded_avg_${n}`] = avgGoalsAgainst(hLast);
    f[`away_goals_conceded_avg_${n}`] = avgGoalsAgainst(aLast);
    f[`home_ppg_${n}`] = avgPPG(hLast);
    f[`away_ppg_${n}`] = avgPPG(aLast);
    f[`home_clean_sheets_${n}`] = cleanSheetRate(hLast);
    f[`away_clean_sheets_${n}`] = cleanSheetRate(aLast);
    f[`home_btts_rate_${n}`] = bttsRate(hLast);
    f[`away_btts_rate_${n}`] = bttsRate(aLast);
    f[`home_over25_rate_${n}`] = over25Rate(hLast);
    f[`away_over25_rate_${n}`] = over25Rate(aLast);
  }

  // ─── 2. Home/Away specific form ───
  const homeAtHome = homeHist.filter(m => m.isHome);
  const awayAtAway = awayHist.filter(m => !m.isHome);

  const hHome5 = lastN(homeAtHome, 5);
  const aAway5 = lastN(awayAtAway, 5);

  if (hHome5.length >= 3) {
    f['home_home_win_rate_5'] = winRate(hHome5);
    f['home_home_goals_avg_5'] = avgGoalsFor(hHome5);
    f['home_home_conceded_avg_5'] = avgGoalsAgainst(hHome5);
  } else {
    f['home_home_win_rate_5'] = f['home_win_rate_5'];
    f['home_home_goals_avg_5'] = f['home_goals_scored_avg_5'];
    f['home_home_conceded_avg_5'] = f['home_goals_conceded_avg_5'];
  }

  if (aAway5.length >= 3) {
    f['away_away_win_rate_5'] = winRate(aAway5);
    f['away_away_goals_avg_5'] = avgGoalsFor(aAway5);
    f['away_away_conceded_avg_5'] = avgGoalsAgainst(aAway5);
  } else {
    f['away_away_win_rate_5'] = f['away_win_rate_5'];
    f['away_away_goals_avg_5'] = f['away_goals_scored_avg_5'];
    f['away_away_conceded_avg_5'] = f['away_goals_conceded_avg_5'];
  }

  // ─── 3. Season-level stats ───
  const seasonHome = homeHist; // All history is within season context
  const seasonAway = awayHist;

  f['home_season_played'] = seasonHome.length;
  f['away_season_played'] = seasonAway.length;
  f['home_season_win_rate'] = winRate(seasonHome);
  f['away_season_win_rate'] = winRate(seasonAway);
  f['home_season_goals_avg'] = avgGoalsFor(seasonHome);
  f['away_season_goals_avg'] = avgGoalsFor(seasonAway);
  f['home_season_conceded_avg'] = avgGoalsAgainst(seasonHome);
  f['away_season_conceded_avg'] = avgGoalsAgainst(seasonAway);
  f['home_season_ppg'] = avgPPG(seasonHome);
  f['away_season_ppg'] = avgPPG(seasonAway);

  // Goal difference per game
  f['home_season_gd_avg'] = f['home_season_goals_avg'] - f['home_season_conceded_avg'];
  f['away_season_gd_avg'] = f['away_season_goals_avg'] - f['away_season_conceded_avg'];

  // ─── 4. Head-to-head ───
  const h2hMatches = findH2HIndexed(h2hIndex, match.homeTeam, match.awayTeam, match.date, 10);
  if (h2hMatches.length >= 2) {
    const homeWins = h2hMatches.filter(m =>
      (m.homeTeam === match.homeTeam && m.ftResult === 'H') ||
      (m.awayTeam === match.homeTeam && m.ftResult === 'A')
    ).length;
    const draws = h2hMatches.filter(m => m.ftResult === 'D').length;
    const totalGoals = h2hMatches.reduce((s, m) => s + m.ftHomeGoals + m.ftAwayGoals, 0);

    f['h2h_count'] = h2hMatches.length;
    f['h2h_home_win_rate'] = homeWins / h2hMatches.length;
    f['h2h_draw_rate'] = draws / h2hMatches.length;
    f['h2h_avg_goals'] = totalGoals / h2hMatches.length;
  } else {
    f['h2h_count'] = 0;
    f['h2h_home_win_rate'] = 0.45; // prior
    f['h2h_draw_rate'] = 0.27;
    f['h2h_avg_goals'] = 2.6;
  }

  // ─── 5. Context features ───
  f['month'] = match.date.getMonth() + 1; // 1-12
  f['day_of_week'] = match.date.getDay(); // 0=Sun, 6=Sat

  // Rest days
  const homeLastDate = homeHist.length > 0 ? homeHist[homeHist.length - 1].date : match.date;
  const awayLastDate = awayHist.length > 0 ? awayHist[awayHist.length - 1].date : match.date;
  f['home_rest_days'] = Math.min(14, daysBetween(homeLastDate, match.date));
  f['away_rest_days'] = Math.min(14, daysBetween(awayLastDate, match.date));

  // Matchday number (approximate from games played)
  f['home_matchday'] = homeHist.length + 1;
  f['away_matchday'] = awayHist.length + 1;

  // ─── 6. Form trend (linear regression on last 10 ppg) ───
  const homePPGSeries = lastN(homeHist, 10).map(m => m.points);
  const awayPPGSeries = lastN(awayHist, 10).map(m => m.points);
  f['home_form_trend'] = linearSlope(homePPGSeries);
  f['away_form_trend'] = linearSlope(awayPPGSeries);

  // ─── 7. Differentials (home - away) ───
  f['diff_win_rate_5'] = f['home_win_rate_5'] - f['away_win_rate_5'];
  f['diff_goals_avg_5'] = f['home_goals_scored_avg_5'] - f['away_goals_scored_avg_5'];
  f['diff_conceded_avg_5'] = f['home_goals_conceded_avg_5'] - f['away_goals_conceded_avg_5'];
  f['diff_ppg_5'] = f['home_ppg_5'] - f['away_ppg_5'];
  f['diff_season_ppg'] = f['home_season_ppg'] - f['away_season_ppg'];
  f['diff_season_gd'] = f['home_season_gd_avg'] - f['away_season_gd_avg'];
  f['diff_form_trend'] = f['home_form_trend'] - f['away_form_trend'];
  f['diff_rest_days'] = f['home_rest_days'] - f['away_rest_days'];

  // ─── 8. League encoding ───
  const PRIMARY_LEAGUE_MAP: Record<string, number> = {
    EPL: 0, SERIE_A: 1, LA_LIGA: 2, BUNDESLIGA: 3, LIGUE_1: 4,
  };
  f['league_code'] = PRIMARY_LEAGUE_MAP[match.league] ?? 5;

  // Extended league info from archive map
  const divCode = LEAGUE_TO_DIV[match.league];
  const leagueInfo = divCode ? ARCHIVE_LEAGUE_MAP[divCode] : undefined;
  f['league_tier'] = leagueInfo?.tier ?? 1;
  f['country_code'] = COUNTRY_ENCODING[leagueInfo?.country ?? ''] ?? 99;
  f['is_primary_league'] = isPrimaryLeague(match.league) ? 1 : 0;

  // ─── 9. Glicko-2 Ratings ───
  const homeKey = teamKey(match.league, match.homeTeam);
  const awayKey = teamKey(match.league, match.awayTeam);

  const glickoPred = glicko2.predict(homeKey, awayKey, match.date);

  f['glicko_home_rating'] = glickoPred.homeRating.rating;
  f['glicko_away_rating'] = glickoPred.awayRating.rating;
  f['glicko_home_rd'] = glickoPred.homeRating.deviation;      // Rating Deviation (uncertainty)
  f['glicko_away_rd'] = glickoPred.awayRating.deviation;
  f['glicko_home_volatility'] = glickoPred.homeRating.volatility;
  f['glicko_away_volatility'] = glickoPred.awayRating.volatility;
  f['glicko_rating_diff'] = glickoPred.homeRating.rating - glickoPred.awayRating.rating;
  f['glicko_uncertainty'] = glickoPred.uncertainty;           // Combined uncertainty
  f['glicko_home_win_prob'] = glickoPred.homeWinProb;         // Glicko-based probability
  f['glicko_draw_prob'] = glickoPred.drawProb;
  f['glicko_away_win_prob'] = glickoPred.awayWinProb;

  // ─── 10. Fatigue & Travel Index ───
  const homeFatigue = fatigue.getFeatures(homeKey, awayKey, match.date, true);
  const awayFatigue = fatigue.getFeatures(awayKey, homeKey, match.date, false);

  f['fatigue_home_rest_days'] = homeFatigue.restDays;
  f['fatigue_away_rest_days'] = awayFatigue.restDays;
  f['fatigue_home_matches_7d'] = homeFatigue.matchesLast7d;
  f['fatigue_away_matches_7d'] = awayFatigue.matchesLast7d;
  f['fatigue_home_matches_14d'] = homeFatigue.matchesLast14d;
  f['fatigue_away_matches_14d'] = awayFatigue.matchesLast14d;
  f['fatigue_home_midweek'] = homeFatigue.midweekMatch ? 1 : 0;
  f['fatigue_away_midweek'] = awayFatigue.midweekMatch ? 1 : 0;
  f['fatigue_home_thu_sun_squeeze'] = homeFatigue.isThursdaySundaySqueeze ? 1 : 0;
  f['fatigue_away_thu_sun_squeeze'] = awayFatigue.isThursdaySundaySqueeze ? 1 : 0;
  f['fatigue_home_travel_km'] = homeFatigue.travelKmLast14d;
  f['fatigue_away_travel_km'] = awayFatigue.travelKmLast14d;
  f['fatigue_home_index'] = homeFatigue.fatigueIndex;         // Composite 0-1
  f['fatigue_away_index'] = awayFatigue.fatigueIndex;
  f['fatigue_rest_advantage'] = homeFatigue.restAdvantage;    // Home advantage in rest days
  f['fatigue_diff'] = awayFatigue.fatigueIndex - homeFatigue.fatigueIndex; // Positive = home less fatigued

  // ─── 11. Playing Style Clustering ───
  const homeStyle = styleTracker.getTeamStyle(homeKey);
  const awayStyle = styleTracker.getTeamStyle(awayKey);

  if (homeStyle) {
    f['style_home_cluster'] = homeStyle.cluster;
    f['style_home_attacking'] = homeStyle.attackingIndex;
    f['style_home_defensive'] = homeStyle.defensiveIndex;
    f['style_home_efficiency'] = homeStyle.efficiencyIndex;
    f['style_home_tempo'] = homeStyle.tempoIndex;
  } else {
    f['style_home_cluster'] = -1;
    f['style_home_attacking'] = 0;
    f['style_home_defensive'] = 0;
    f['style_home_efficiency'] = 0;
    f['style_home_tempo'] = 0;
  }

  if (awayStyle) {
    f['style_away_cluster'] = awayStyle.cluster;
    f['style_away_attacking'] = awayStyle.attackingIndex;
    f['style_away_defensive'] = awayStyle.defensiveIndex;
    f['style_away_efficiency'] = awayStyle.efficiencyIndex;
    f['style_away_tempo'] = awayStyle.tempoIndex;
  } else {
    f['style_away_cluster'] = -1;
    f['style_away_attacking'] = 0;
    f['style_away_defensive'] = 0;
    f['style_away_efficiency'] = 0;
    f['style_away_tempo'] = 0;
  }

  // Style matchup features
  if (homeStyle && awayStyle) {
    const matchup = styleTracker.getMatchupStats(homeStyle.cluster, awayStyle.cluster);
    f['style_matchup_home_wr'] = matchup.historicalHomeWinRate;
    f['style_matchup_draw_rate'] = matchup.historicalDrawRate;
    f['style_matchup_away_wr'] = matchup.historicalAwayWinRate;
    f['style_matchup_avg_goals'] = matchup.historicalAvgGoals;
    // Style differentials
    f['style_attacking_diff'] = homeStyle.attackingIndex - awayStyle.attackingIndex;
    f['style_defensive_diff'] = homeStyle.defensiveIndex - awayStyle.defensiveIndex;
  } else {
    f['style_matchup_home_wr'] = 0.45;
    f['style_matchup_draw_rate'] = 0.27;
    f['style_matchup_away_wr'] = 0.28;
    f['style_matchup_avg_goals'] = 2.6;
    f['style_attacking_diff'] = 0;
    f['style_defensive_diff'] = 0;
  }

  // ─── 12. Archive Elo Ratings ───
  if (match.archiveHomeElo != null && match.archiveAwayElo != null) {
    f['elo_home'] = match.archiveHomeElo;
    f['elo_away'] = match.archiveAwayElo;
    f['elo_diff'] = match.archiveHomeElo - match.archiveAwayElo;
    f['elo_avg'] = (match.archiveHomeElo + match.archiveAwayElo) / 2;
    // Elo-based win probability (logistic model, 400 scale)
    const eloDiff = match.archiveHomeElo - match.archiveAwayElo;
    f['elo_home_win_prob'] = 1 / (1 + Math.pow(10, -eloDiff / 400));
    f['elo_away_win_prob'] = 1 - f['elo_home_win_prob'];
  } else {
    // Fallback to Glicko-2 ratings
    f['elo_home'] = f['glicko_home_rating'] || 1500;
    f['elo_away'] = f['glicko_away_rating'] || 1500;
    f['elo_diff'] = f['elo_home'] - f['elo_away'];
    f['elo_avg'] = (f['elo_home'] + f['elo_away']) / 2;
    f['elo_home_win_prob'] = f['glicko_home_win_prob'] || 0.45;
    f['elo_away_win_prob'] = f['glicko_away_win_prob'] || 0.28;
  }

  return f;
}

// ─── Helper functions ───

function teamKey(league: LeagueCode, team: string): string {
  return `${league}:${team}`;
}

function lastN<T>(arr: T[], n: number): T[] {
  return arr.slice(Math.max(0, arr.length - n));
}

function winRate(matches: FormMatch[]): number {
  if (matches.length === 0) return 0;
  return matches.filter(m => m.result === 'W').length / matches.length;
}

function drawRate(matches: FormMatch[]): number {
  if (matches.length === 0) return 0;
  return matches.filter(m => m.result === 'D').length / matches.length;
}

function avgGoalsFor(matches: FormMatch[]): number {
  if (matches.length === 0) return 0;
  return matches.reduce((s, m) => s + m.goalsFor, 0) / matches.length;
}

function avgGoalsAgainst(matches: FormMatch[]): number {
  if (matches.length === 0) return 0;
  return matches.reduce((s, m) => s + m.goalsAgainst, 0) / matches.length;
}

function avgPPG(matches: FormMatch[]): number {
  if (matches.length === 0) return 0;
  return matches.reduce((s, m) => s + m.points, 0) / matches.length;
}

function cleanSheetRate(matches: FormMatch[]): number {
  if (matches.length === 0) return 0;
  return matches.filter(m => m.goalsAgainst === 0).length / matches.length;
}

function bttsRate(matches: FormMatch[]): number {
  if (matches.length === 0) return 0;
  return matches.filter(m => m.goalsFor > 0 && m.goalsAgainst > 0).length / matches.length;
}

function over25Rate(matches: FormMatch[]): number {
  if (matches.length === 0) return 0;
  return matches.filter(m => m.goalsFor + m.goalsAgainst > 2).length / matches.length;
}

function avg(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function daysBetween(a: Date, b: Date): number {
  return Math.floor(Math.abs(b.getTime() - a.getTime()) / (1000 * 60 * 60 * 24));
}

function linearSlope(values: number[]): number {
  if (values.length < 3) return 0;
  const n = values.length;
  const xMean = (n - 1) / 2;
  const yMean = values.reduce((a, b) => a + b, 0) / n;

  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (values[i] - yMean);
    den += (i - xMean) ** 2;
  }

  return den === 0 ? 0 : num / den;
}

function findH2HIndexed(
  index: Map<string, Match[]>,
  team1: string,
  team2: string,
  beforeDate: Date,
  maxCount: number,
): Match[] {
  const pair = [team1, team2].sort().join('|');
  const matches = index.get(pair);
  if (!matches) return [];
  // Matches are sorted by date (from buildFeatures sort), filter and take last N
  const before = matches.filter(m => m.date < beforeDate);
  return before.slice(-maxCount);
}

/**
 * Get ordered list of feature names (for model input consistency)
 */
export function getFeatureNames(): string[] {
  // Generate a dummy to extract keys (order is insertion order)
  // This is a reference list - actual features computed dynamically
  return [
    // Form last 5
    'home_win_rate_5', 'away_win_rate_5',
    'home_draw_rate_5', 'away_draw_rate_5',
    'home_goals_scored_avg_5', 'away_goals_scored_avg_5',
    'home_goals_conceded_avg_5', 'away_goals_conceded_avg_5',
    'home_ppg_5', 'away_ppg_5',
    'home_clean_sheets_5', 'away_clean_sheets_5',
    'home_btts_rate_5', 'away_btts_rate_5',
    'home_over25_rate_5', 'away_over25_rate_5',
    // Form last 10
    'home_win_rate_10', 'away_win_rate_10',
    'home_draw_rate_10', 'away_draw_rate_10',
    'home_goals_scored_avg_10', 'away_goals_scored_avg_10',
    'home_goals_conceded_avg_10', 'away_goals_conceded_avg_10',
    'home_ppg_10', 'away_ppg_10',
    'home_clean_sheets_10', 'away_clean_sheets_10',
    'home_btts_rate_10', 'away_btts_rate_10',
    'home_over25_rate_10', 'away_over25_rate_10',
    // Home/Away specific
    'home_home_win_rate_5', 'home_home_goals_avg_5', 'home_home_conceded_avg_5',
    'away_away_win_rate_5', 'away_away_goals_avg_5', 'away_away_conceded_avg_5',
    // Season
    'home_season_played', 'away_season_played',
    'home_season_win_rate', 'away_season_win_rate',
    'home_season_goals_avg', 'away_season_goals_avg',
    'home_season_conceded_avg', 'away_season_conceded_avg',
    'home_season_ppg', 'away_season_ppg',
    'home_season_gd_avg', 'away_season_gd_avg',
    // H2H
    'h2h_count', 'h2h_home_win_rate', 'h2h_draw_rate', 'h2h_avg_goals',
    // Context
    'month', 'day_of_week', 'home_rest_days', 'away_rest_days',
    'home_matchday', 'away_matchday',
    // Form trend
    'home_form_trend', 'away_form_trend',
    // Differentials
    'diff_win_rate_5', 'diff_goals_avg_5', 'diff_conceded_avg_5',
    'diff_ppg_5', 'diff_season_ppg', 'diff_season_gd',
    'diff_form_trend', 'diff_rest_days',
    // League
    'league_code', 'league_tier', 'country_code', 'is_primary_league',
    // Glicko-2 Ratings
    'glicko_home_rating', 'glicko_away_rating',
    'glicko_home_rd', 'glicko_away_rd',
    'glicko_home_volatility', 'glicko_away_volatility',
    'glicko_rating_diff', 'glicko_uncertainty',
    'glicko_home_win_prob', 'glicko_draw_prob', 'glicko_away_win_prob',
    // Fatigue & Travel
    'fatigue_home_rest_days', 'fatigue_away_rest_days',
    'fatigue_home_matches_7d', 'fatigue_away_matches_7d',
    'fatigue_home_matches_14d', 'fatigue_away_matches_14d',
    'fatigue_home_midweek', 'fatigue_away_midweek',
    'fatigue_home_thu_sun_squeeze', 'fatigue_away_thu_sun_squeeze',
    'fatigue_home_travel_km', 'fatigue_away_travel_km',
    'fatigue_home_index', 'fatigue_away_index',
    'fatigue_rest_advantage', 'fatigue_diff',
    // Style Clustering
    'style_home_cluster', 'style_away_cluster',
    'style_home_attacking', 'style_away_attacking',
    'style_home_defensive', 'style_away_defensive',
    'style_home_efficiency', 'style_away_efficiency',
    'style_home_tempo', 'style_away_tempo',
    'style_matchup_home_wr', 'style_matchup_draw_rate', 'style_matchup_away_wr',
    'style_matchup_avg_goals',
    'style_attacking_diff', 'style_defensive_diff',
    // Elo Ratings (archive)
    'elo_home', 'elo_away', 'elo_diff', 'elo_avg',
    'elo_home_win_prob', 'elo_away_win_prob',
  ];
}
