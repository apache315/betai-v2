/**
 * Playing Style Clustering (K-Means)
 *
 * Clusters teams into playing style groups based on statistics:
 * - Offensive output (goals, shots)
 * - Defensive solidity (goals conceded, shots faced)
 * - Efficiency (goals/shots ratio)
 * - Tempo (total goals in games)
 *
 * Style clusters help the model learn that some matchups are
 * style-dependent, not just strength-dependent.
 *
 * Example clusters:
 * - "Possession" (City, Barca): high shots, patient build-up
 * - "Counter" (Atletico, Inter): low possession, clinical finishing
 * - "Direct" (Burnley, Atalanta): high tempo, physical
 * - "Defensive" (low-block teams): few shots, few conceded
 */

import type { Match } from '../src/types/index.js';

export interface TeamStyle {
  cluster: number;           // Cluster ID (0-3)
  clusterName: string;       // Human-readable name
  // Raw style metrics (z-scored)
  attackingIndex: number;    // Goals + shots composite
  defensiveIndex: number;    // Goals conceded + shots faced (inverted)
  efficiencyIndex: number;   // Goals per shot
  tempoIndex: number;        // Total goals in games
  // Distance to each cluster centroid
  clusterDistances: number[];
}

export interface StyleMatchup {
  homeStyle: number;
  awayStyle: number;
  matchupKey: string;        // e.g., "0v2" = Possession vs Direct
  historicalHomeWinRate: number;
  historicalDrawRate: number;
  historicalAwayWinRate: number;
  historicalAvgGoals: number;
}

// Cluster names (will be assigned after k-means)
const CLUSTER_NAMES = [
  'Possession',    // High shots, patient
  'Counter',       // Clinical, low possession
  'Direct',        // High tempo, physical
  'Defensive',     // Low scoring, compact
];

const NUM_CLUSTERS = 4;
const MAX_ITERATIONS = 100;

/**
 * Simple K-Means implementation
 */
function kMeans(
  data: number[][],
  k: number,
  maxIterations: number = MAX_ITERATIONS,
): { centroids: number[][]; labels: number[] } {
  const n = data.length;
  const dims = data[0].length;

  // Initialize centroids using k-means++
  const centroids: number[][] = [];
  const usedIndices = new Set<number>();

  // First centroid: random
  const firstIdx = Math.floor(Math.random() * n);
  centroids.push([...data[firstIdx]]);
  usedIndices.add(firstIdx);

  // Remaining centroids: weighted by distance
  while (centroids.length < k) {
    const distances = data.map((point, idx) => {
      if (usedIndices.has(idx)) return 0;
      const minDist = Math.min(...centroids.map(c => euclideanDistance(point, c)));
      return minDist * minDist;
    });
    const totalDist = distances.reduce((a, b) => a + b, 0);
    let r = Math.random() * totalDist;
    for (let i = 0; i < n; i++) {
      r -= distances[i];
      if (r <= 0 && !usedIndices.has(i)) {
        centroids.push([...data[i]]);
        usedIndices.add(i);
        break;
      }
    }
  }

  let labels = new Array(n).fill(0);

  for (let iter = 0; iter < maxIterations; iter++) {
    // Assign labels
    const newLabels = data.map(point => {
      let minDist = Infinity;
      let bestCluster = 0;
      for (let c = 0; c < k; c++) {
        const dist = euclideanDistance(point, centroids[c]);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = c;
        }
      }
      return bestCluster;
    });

    // Check convergence
    const changed = newLabels.some((l, i) => l !== labels[i]);
    labels = newLabels;

    if (!changed) break;

    // Update centroids
    for (let c = 0; c < k; c++) {
      const clusterPoints = data.filter((_, i) => labels[i] === c);
      if (clusterPoints.length === 0) continue;

      for (let d = 0; d < dims; d++) {
        centroids[c][d] = clusterPoints.reduce((sum, p) => sum + p[d], 0) / clusterPoints.length;
      }
    }
  }

  return { centroids, labels };
}

function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0));
}

/**
 * Z-score normalization
 */
function zScore(values: number[]): number[] {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length);
  if (std === 0) return values.map(() => 0);
  return values.map(v => (v - mean) / std);
}

/**
 * Style Clustering Tracker
 *
 * Computes and maintains style clusters for all teams.
 */
export class StyleClusteringTracker {
  private teamStats: Map<string, {
    goalsScored: number[];
    goalsConceded: number[];
    shotsFor: number[];
    shotsAgainst: number[];
    matchCount: number;
  }> = new Map();

  private centroids: number[][] = [];
  private teamClusters: Map<string, TeamStyle> = new Map();
  private matchupHistory: Map<string, { home: number; draw: number; away: number; goals: number; count: number }> = new Map();
  private isFitted = false;

  /**
   * Record a match result (during history processing)
   */
  recordMatch(
    homeTeam: string,
    awayTeam: string,
    homeGoals: number,
    awayGoals: number,
    homeShots?: number,
    awayShots?: number,
    league?: string,
  ): void {
    const homeKey = league ? `${league}:${homeTeam}` : homeTeam;
    const awayKey = league ? `${league}:${awayTeam}` : awayTeam;

    // Initialize if needed
    for (const key of [homeKey, awayKey]) {
      if (!this.teamStats.has(key)) {
        this.teamStats.set(key, {
          goalsScored: [],
          goalsConceded: [],
          shotsFor: [],
          shotsAgainst: [],
          matchCount: 0,
        });
      }
    }

    const homeStats = this.teamStats.get(homeKey)!;
    const awayStats = this.teamStats.get(awayKey)!;

    // Update home team
    homeStats.goalsScored.push(homeGoals);
    homeStats.goalsConceded.push(awayGoals);
    if (homeShots !== undefined) homeStats.shotsFor.push(homeShots);
    if (awayShots !== undefined) homeStats.shotsAgainst.push(awayShots);
    homeStats.matchCount++;

    // Update away team
    awayStats.goalsScored.push(awayGoals);
    awayStats.goalsConceded.push(homeGoals);
    if (awayShots !== undefined) awayStats.shotsFor.push(awayShots);
    if (homeShots !== undefined) awayStats.shotsAgainst.push(homeShots);
    awayStats.matchCount++;

    this.teamStats.set(homeKey, homeStats);
    this.teamStats.set(awayKey, awayStats);
  }

  /**
   * Fit clusters on accumulated data
   * Call this after processing historical matches
   */
  fitClusters(minMatches: number = 10): void {
    // Filter teams with enough matches
    const eligibleTeams: string[] = [];
    const featureMatrix: number[][] = [];

    for (const [teamKey, stats] of this.teamStats.entries()) {
      if (stats.matchCount < minMatches) continue;

      const avgGoalsScored = stats.goalsScored.reduce((a, b) => a + b, 0) / stats.goalsScored.length;
      const avgGoalsConceded = stats.goalsConceded.reduce((a, b) => a + b, 0) / stats.goalsConceded.length;
      const avgShotsFor = stats.shotsFor.length > 0
        ? stats.shotsFor.reduce((a, b) => a + b, 0) / stats.shotsFor.length
        : avgGoalsScored * 8; // Estimate if no data
      const avgShotsAgainst = stats.shotsAgainst.length > 0
        ? stats.shotsAgainst.reduce((a, b) => a + b, 0) / stats.shotsAgainst.length
        : avgGoalsConceded * 8;

      // Feature vector: [attacking, defensive, efficiency, tempo]
      const attacking = avgGoalsScored + avgShotsFor * 0.1;
      const defensive = avgGoalsConceded + avgShotsAgainst * 0.1;
      const efficiency = avgShotsFor > 0 ? avgGoalsScored / avgShotsFor : 0.1;
      const tempo = avgGoalsScored + avgGoalsConceded;

      eligibleTeams.push(teamKey);
      featureMatrix.push([attacking, defensive, efficiency, tempo]);
    }

    if (eligibleTeams.length < NUM_CLUSTERS) {
      console.warn(`[style-clustering] Not enough teams (${eligibleTeams.length}) for ${NUM_CLUSTERS} clusters`);
      return;
    }

    // Z-score normalize features
    const transposed = [0, 1, 2, 3].map(d => featureMatrix.map(row => row[d]));
    const normalized = transposed.map(col => zScore(col));
    const normalizedMatrix = featureMatrix.map((_, i) => normalized.map(col => col[i]));

    // Run k-means
    const { centroids, labels } = kMeans(normalizedMatrix, NUM_CLUSTERS);
    this.centroids = centroids;

    // Assign clusters to teams
    for (let i = 0; i < eligibleTeams.length; i++) {
      const teamKey = eligibleTeams[i];
      const label = labels[i];
      const point = normalizedMatrix[i];

      // Calculate distances to all centroids
      const distances = centroids.map(c => euclideanDistance(point, c));

      this.teamClusters.set(teamKey, {
        cluster: label,
        clusterName: CLUSTER_NAMES[label] || `Cluster${label}`,
        attackingIndex: point[0],
        defensiveIndex: -point[1], // Invert (lower conceded = better)
        efficiencyIndex: point[2],
        tempoIndex: point[3],
        clusterDistances: distances,
      });
    }

    this.isFitted = true;
    console.log(`[style-clustering] Fitted ${NUM_CLUSTERS} clusters on ${eligibleTeams.length} teams`);

    // Log cluster sizes
    const clusterCounts = new Array(NUM_CLUSTERS).fill(0);
    labels.forEach(l => clusterCounts[l]++);
    console.log(`[style-clustering] Cluster sizes: ${clusterCounts.map((c, i) => `${CLUSTER_NAMES[i]}=${c}`).join(', ')}`);
  }

  /**
   * Get team style (must call fitClusters first)
   */
  getTeamStyle(teamKey: string): TeamStyle | null {
    return this.teamClusters.get(teamKey) || null;
  }

  /**
   * Record matchup result for historical analysis
   */
  recordMatchup(
    homeCluster: number,
    awayCluster: number,
    homeGoals: number,
    awayGoals: number,
  ): void {
    const key = `${homeCluster}v${awayCluster}`;

    if (!this.matchupHistory.has(key)) {
      this.matchupHistory.set(key, { home: 0, draw: 0, away: 0, goals: 0, count: 0 });
    }

    const hist = this.matchupHistory.get(key)!;
    if (homeGoals > awayGoals) hist.home++;
    else if (homeGoals === awayGoals) hist.draw++;
    else hist.away++;
    hist.goals += homeGoals + awayGoals;
    hist.count++;
  }

  /**
   * Get historical matchup statistics
   */
  getMatchupStats(homeCluster: number, awayCluster: number): StyleMatchup {
    const key = `${homeCluster}v${awayCluster}`;
    const hist = this.matchupHistory.get(key) || { home: 0, draw: 0, away: 0, goals: 0, count: 0 };

    const total = hist.count || 1;

    return {
      homeStyle: homeCluster,
      awayStyle: awayCluster,
      matchupKey: key,
      historicalHomeWinRate: hist.home / total,
      historicalDrawRate: hist.draw / total,
      historicalAwayWinRate: hist.away / total,
      historicalAvgGoals: hist.goals / total,
    };
  }

  /**
   * Process all matches to build clusters
   */
  processMatches(matches: Match[]): void {
    // Sort chronologically
    const sorted = [...matches].sort((a, b) => a.date.getTime() - b.date.getTime());

    // First pass: record all matches
    for (const match of sorted) {
      this.recordMatch(
        match.homeTeam,
        match.awayTeam,
        match.ftHomeGoals,
        match.ftAwayGoals,
        match.homeShots,
        match.awayShots,
        match.league,
      );
    }

    // Fit clusters
    this.fitClusters();

    // Second pass: record matchup history
    for (const match of sorted) {
      const homeKey = `${match.league}:${match.homeTeam}`;
      const awayKey = `${match.league}:${match.awayTeam}`;

      const homeStyle = this.teamClusters.get(homeKey);
      const awayStyle = this.teamClusters.get(awayKey);

      if (homeStyle && awayStyle) {
        this.recordMatchup(
          homeStyle.cluster,
          awayStyle.cluster,
          match.ftHomeGoals,
          match.ftAwayGoals,
        );
      }
    }
  }

  /**
   * Get all matchup statistics
   */
  getAllMatchupStats(): Map<string, StyleMatchup> {
    const result = new Map<string, StyleMatchup>();

    for (let h = 0; h < NUM_CLUSTERS; h++) {
      for (let a = 0; a < NUM_CLUSTERS; a++) {
        const stats = this.getMatchupStats(h, a);
        result.set(stats.matchupKey, stats);
      }
    }

    return result;
  }
}

/**
 * Singleton tracker
 */
let globalStyleTracker: StyleClusteringTracker | null = null;

export function getGlobalStyleTracker(): StyleClusteringTracker {
  if (!globalStyleTracker) {
    globalStyleTracker = new StyleClusteringTracker();
  }
  return globalStyleTracker;
}

export function resetGlobalStyleTracker(): void {
  globalStyleTracker = new StyleClusteringTracker();
}
