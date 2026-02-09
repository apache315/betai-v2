/**
 * Fatigue & Travel Index
 *
 * Tracks physical load and travel burden for each team.
 *
 * Research shows:
 * - Teams playing Thursday (Europa) + Sunday lose ~0.3 xG
 * - 2 days rest vs 5+ days rest = significant performance drop
 * - Cumulative load over 7-14 days matters
 *
 * Features generated:
 * - rest_days: Days since last match
 * - matches_last_7d: Fixture congestion
 * - matches_last_14d: Medium-term load
 * - midweek_european: Thursday night effect (boolean)
 * - travel_km_last_14d: Approximate travel burden
 * - fatigue_index: Combined fatigue score (0-1)
 */

import type { Match, LeagueCode } from '../src/types/index.js';

export interface FatigueState {
  lastMatchDate: Date;
  last7DaysMatches: Date[];    // Dates of matches in last 7 days
  last14DaysMatches: Date[];   // Dates of matches in last 14 days
  lastMatchWasAway: boolean;
  travelKmLast14d: number;     // Approximate km traveled
}

export interface FatigueFeatures {
  restDays: number;
  matchesLast7d: number;
  matchesLast14d: number;
  midweekMatch: boolean;       // Played Tue/Wed/Thu this week
  isThursdaySundaySqueeze: boolean; // Thursday match -> Sunday match
  travelKmLast14d: number;
  fatigueIndex: number;        // 0-1, higher = more fatigued
  restAdvantage: number;       // vs opponent (positive = more rested)
}

/**
 * Stadium coordinates for major teams (approximate city centers)
 * Used to estimate travel distances
 */
const CITY_COORDS: Record<string, { lat: number; lng: number }> = {
  // England
  'Arsenal': { lat: 51.5549, lng: -0.1084 },
  'Aston Villa': { lat: 52.5092, lng: -1.8846 },
  'Bournemouth': { lat: 50.7352, lng: -1.8384 },
  'Brentford': { lat: 51.4882, lng: -0.3026 },
  'Brighton': { lat: 50.8619, lng: -0.0833 },
  'Burnley': { lat: 53.7890, lng: -2.2301 },
  'Chelsea': { lat: 51.4817, lng: -0.1910 },
  'Crystal Palace': { lat: 51.3983, lng: -0.0858 },
  'Everton': { lat: 53.4389, lng: -2.9663 },
  'Fulham': { lat: 51.4749, lng: -0.2217 },
  'Ipswich': { lat: 52.0547, lng: 1.1448 },
  'Leicester': { lat: 52.6204, lng: -1.1422 },
  'Liverpool': { lat: 53.4308, lng: -2.9608 },
  'Luton': { lat: 51.8843, lng: -0.4316 },
  'Man City': { lat: 53.4831, lng: -2.2004 },
  'Man United': { lat: 53.4631, lng: -2.2913 },
  'Newcastle': { lat: 54.9756, lng: -1.6217 },
  'Nott\'m Forest': { lat: 52.9399, lng: -1.1328 },
  'Sheffield United': { lat: 53.3703, lng: -1.4709 },
  'Southampton': { lat: 50.9058, lng: -1.3910 },
  'Tottenham': { lat: 51.6042, lng: -0.0662 },
  'West Ham': { lat: 51.5387, lng: -0.0166 },
  'Wolves': { lat: 52.5902, lng: -2.1304 },
  // Italy
  'Juventus': { lat: 45.1096, lng: 7.6413 },
  'Inter': { lat: 45.4781, lng: 9.1240 },
  'Milan': { lat: 45.4781, lng: 9.1240 },
  'Napoli': { lat: 40.8280, lng: 14.1930 },
  'Roma': { lat: 41.9341, lng: 12.4547 },
  'Lazio': { lat: 41.9341, lng: 12.4547 },
  'Atalanta': { lat: 45.7092, lng: 9.6807 },
  'Fiorentina': { lat: 43.7808, lng: 11.2824 },
  // Spain
  'Barcelona': { lat: 41.3809, lng: 2.1228 },
  'Real Madrid': { lat: 40.4530, lng: -3.6883 },
  'Atletico Madrid': { lat: 40.4361, lng: -3.5994 },
  'Sevilla': { lat: 37.3841, lng: -5.9706 },
  'Valencia': { lat: 39.4747, lng: -0.3586 },
  'Villarreal': { lat: 39.9440, lng: -0.1037 },
  // Germany
  'Bayern Munich': { lat: 48.2188, lng: 11.6247 },
  'Dortmund': { lat: 51.4926, lng: 7.4518 },
  'RB Leipzig': { lat: 51.3459, lng: 12.3489 },
  'Leverkusen': { lat: 51.0383, lng: 7.0020 },
  'Frankfurt': { lat: 50.0686, lng: 8.6454 },
  // France
  'Paris S-G': { lat: 48.8414, lng: 2.2530 },
  'Marseille': { lat: 43.2698, lng: 5.3959 },
  'Lyon': { lat: 45.7652, lng: 4.9822 },
  'Monaco': { lat: 43.7277, lng: 7.4157 },
  'Lille': { lat: 50.6120, lng: 3.1305 },
};

/**
 * Calculate distance between two points using Haversine formula
 */
function haversineDistance(
  lat1: number, lng1: number,
  lat2: number, lng2: number,
): number {
  const R = 6371; // Earth's radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLng / 2) * Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Get approximate travel distance between teams
 */
function getTravelDistance(teamA: string, teamB: string): number {
  const coordsA = findTeamCoords(teamA);
  const coordsB = findTeamCoords(teamB);

  if (!coordsA || !coordsB) {
    return 200; // Default ~200km if unknown
  }

  return haversineDistance(coordsA.lat, coordsA.lng, coordsB.lat, coordsB.lng);
}

/**
 * Find team coordinates with fuzzy matching
 */
function findTeamCoords(teamName: string): { lat: number; lng: number } | null {
  // Direct match
  if (CITY_COORDS[teamName]) {
    return CITY_COORDS[teamName];
  }

  // Fuzzy match (contains)
  const lower = teamName.toLowerCase();
  for (const [name, coords] of Object.entries(CITY_COORDS)) {
    if (name.toLowerCase().includes(lower) || lower.includes(name.toLowerCase())) {
      return coords;
    }
  }

  return null;
}

/**
 * Fatigue Tracker
 *
 * Maintains fatigue state for all teams across matches.
 */
export class FatigueTracker {
  private states: Map<string, FatigueState> = new Map();

  /**
   * Get fatigue state for a team
   */
  private getState(teamKey: string): FatigueState {
    if (!this.states.has(teamKey)) {
      this.states.set(teamKey, {
        lastMatchDate: new Date(0),
        last7DaysMatches: [],
        last14DaysMatches: [],
        lastMatchWasAway: false,
        travelKmLast14d: 0,
      });
    }
    return this.states.get(teamKey)!;
  }

  /**
   * Calculate fatigue features before a match
   */
  getFeatures(
    teamKey: string,
    opponentKey: string,
    matchDate: Date,
    isHome: boolean,
  ): FatigueFeatures {
    const state = this.getState(teamKey);
    const oppState = this.getState(opponentKey);

    // Rest days
    const restDays = Math.floor(
      (matchDate.getTime() - state.lastMatchDate.getTime()) / (1000 * 60 * 60 * 24)
    );
    const oppRestDays = Math.floor(
      (matchDate.getTime() - oppState.lastMatchDate.getTime()) / (1000 * 60 * 60 * 24)
    );

    // Filter to only matches within window
    const sevenDaysAgo = new Date(matchDate.getTime() - 7 * 24 * 60 * 60 * 1000);
    const fourteenDaysAgo = new Date(matchDate.getTime() - 14 * 24 * 60 * 60 * 1000);

    const matchesLast7d = state.last7DaysMatches.filter(d => d >= sevenDaysAgo).length;
    const matchesLast14d = state.last14DaysMatches.filter(d => d >= fourteenDaysAgo).length;

    // Midweek match detection (Tue=2, Wed=3, Thu=4)
    const dayOfWeek = matchDate.getDay();
    const midweekMatch = [2, 3, 4].includes(dayOfWeek);

    // Thursday-Sunday squeeze
    // Check if team played on Thursday (day=4) and this match is Sunday (day=0)
    const isThursdaySundaySqueeze =
      dayOfWeek === 0 && restDays <= 3 &&
      state.lastMatchDate.getDay() === 4;

    // Travel in last 14 days
    const travelKmLast14d = Math.min(5000, state.travelKmLast14d);

    // Calculate composite fatigue index (0-1)
    // Higher = more fatigued
    let fatigueIndex = 0;

    // Rest days component (ideal = 5-7 days)
    if (restDays <= 2) fatigueIndex += 0.3;
    else if (restDays <= 3) fatigueIndex += 0.2;
    else if (restDays <= 4) fatigueIndex += 0.1;

    // Match congestion
    if (matchesLast7d >= 2) fatigueIndex += 0.2;
    if (matchesLast14d >= 4) fatigueIndex += 0.15;

    // Thursday-Sunday squeeze (Europa League effect)
    if (isThursdaySundaySqueeze) fatigueIndex += 0.15;

    // Travel burden
    if (travelKmLast14d > 3000) fatigueIndex += 0.15;
    else if (travelKmLast14d > 1500) fatigueIndex += 0.1;
    else if (travelKmLast14d > 500) fatigueIndex += 0.05;

    // Cap at 1.0
    fatigueIndex = Math.min(1.0, fatigueIndex);

    return {
      restDays: Math.min(14, restDays), // Cap at 14
      matchesLast7d,
      matchesLast14d,
      midweekMatch,
      isThursdaySundaySqueeze,
      travelKmLast14d,
      fatigueIndex,
      restAdvantage: Math.min(7, Math.max(-7, restDays - oppRestDays)),
    };
  }

  /**
   * Update fatigue state after a match
   */
  updateMatch(
    homeTeam: string,
    awayTeam: string,
    matchDate: Date,
    leagueKey: string,
  ): void {
    const homeKey = `${leagueKey}:${homeTeam}`;
    const awayKey = `${leagueKey}:${awayTeam}`;

    const homeState = this.getState(homeKey);
    const awayState = this.getState(awayKey);

    // Calculate travel for away team
    const travelDist = getTravelDistance(awayTeam, homeTeam);

    // Update home team
    homeState.lastMatchDate = matchDate;
    homeState.last7DaysMatches.push(matchDate);
    homeState.last14DaysMatches.push(matchDate);
    homeState.lastMatchWasAway = false;
    // Clean old matches
    const sevenDaysAgo = new Date(matchDate.getTime() - 7 * 24 * 60 * 60 * 1000);
    const fourteenDaysAgo = new Date(matchDate.getTime() - 14 * 24 * 60 * 60 * 1000);
    homeState.last7DaysMatches = homeState.last7DaysMatches.filter(d => d >= sevenDaysAgo);
    homeState.last14DaysMatches = homeState.last14DaysMatches.filter(d => d >= fourteenDaysAgo);

    // Update away team
    awayState.lastMatchDate = matchDate;
    awayState.last7DaysMatches.push(matchDate);
    awayState.last14DaysMatches.push(matchDate);
    awayState.lastMatchWasAway = true;
    awayState.travelKmLast14d += travelDist * 2; // Round trip
    // Clean old matches and decay travel
    awayState.last7DaysMatches = awayState.last7DaysMatches.filter(d => d >= sevenDaysAgo);
    awayState.last14DaysMatches = awayState.last14DaysMatches.filter(d => d >= fourteenDaysAgo);
    // Decay old travel (beyond 14 days)
    const daysSinceLast = homeState.last14DaysMatches.length > 1
      ? Math.floor((matchDate.getTime() - homeState.last14DaysMatches[homeState.last14DaysMatches.length - 2].getTime()) / (1000 * 60 * 60 * 24))
      : 14;
    awayState.travelKmLast14d = Math.max(0, awayState.travelKmLast14d * Math.pow(0.9, daysSinceLast / 7));

    this.states.set(homeKey, homeState);
    this.states.set(awayKey, awayState);
  }

  /**
   * Process all matches chronologically
   */
  processMatches(matches: Match[]): void {
    const sorted = [...matches].sort((a, b) => a.date.getTime() - b.date.getTime());

    for (const match of sorted) {
      this.updateMatch(
        match.homeTeam,
        match.awayTeam,
        match.date,
        match.league,
      );
    }
  }
}

/**
 * Singleton tracker
 */
let globalFatigueTracker: FatigueTracker | null = null;

export function getGlobalFatigueTracker(): FatigueTracker {
  if (!globalFatigueTracker) {
    globalFatigueTracker = new FatigueTracker();
  }
  return globalFatigueTracker;
}

export function resetGlobalFatigueTracker(): void {
  globalFatigueTracker = new FatigueTracker();
}
