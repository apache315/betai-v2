/**
 * Glicko-2 Rating System
 *
 * Superior to Elo for sports prediction because it tracks:
 * 1. Rating (μ) - team strength
 * 2. Rating Deviation (φ) - uncertainty in rating
 * 3. Volatility (σ) - how consistent/volatile the team is
 *
 * Key advantages:
 * - High φ = model should be less confident (new team, roster changes)
 * - High σ = team is unpredictable (upset potential)
 * - Ratings decay when team hasn't played (φ increases)
 *
 * Reference: Mark Glickman (2013) "Example of the Glicko-2 system"
 * http://www.glicko.net/glicko/glicko2.pdf
 */

export interface Glicko2Rating {
  rating: number;      // μ in Glicko-2 scale, ~1500 = average
  deviation: number;   // φ (RD) - uncertainty, lower = more confident
  volatility: number;  // σ - how predictable the team is
  lastPlayed: Date;
}

export interface Glicko2Result {
  homeRating: Glicko2Rating;
  awayRating: Glicko2Rating;
  expectedHome: number;   // Expected score for home (0-1)
  expectedAway: number;
  homeWinProb: number;    // Probability home wins
  drawProb: number;
  awayWinProb: number;
  uncertainty: number;    // Combined uncertainty (higher = less confident)
}

// Glicko-2 constants
const TAU = 0.5;           // System volatility constant (0.3-1.2, lower = less volatile)
const INITIAL_RATING = 1500;
const INITIAL_RD = 350;    // High initial uncertainty
const INITIAL_VOLATILITY = 0.06;
const CONVERGENCE_TOLERANCE = 0.000001;

// Scaling constant (Glicko-2 uses different scale than Glicko-1)
const SCALE = 173.7178;

/**
 * Glicko-2 Rating Tracker
 *
 * Maintains ratings for all teams across matches.
 */
export class Glicko2Tracker {
  private ratings: Map<string, Glicko2Rating> = new Map();

  /**
   * Get or create rating for a team
   */
  getRating(teamKey: string): Glicko2Rating {
    if (!this.ratings.has(teamKey)) {
      this.ratings.set(teamKey, {
        rating: INITIAL_RATING,
        deviation: INITIAL_RD,
        volatility: INITIAL_VOLATILITY,
        lastPlayed: new Date(0),
      });
    }
    return { ...this.ratings.get(teamKey)! };
  }

  /**
   * Calculate expected outcome and probabilities before a match
   */
  predict(homeKey: string, awayKey: string, matchDate: Date): Glicko2Result {
    const home = this.getRatingWithDecay(homeKey, matchDate);
    const away = this.getRatingWithDecay(awayKey, matchDate);

    // Convert to Glicko-2 scale
    const muHome = (home.rating - INITIAL_RATING) / SCALE;
    const muAway = (away.rating - INITIAL_RATING) / SCALE;
    const phiHome = home.deviation / SCALE;
    const phiAway = away.deviation / SCALE;

    // Calculate g(φ) - reduces impact of uncertain ratings
    const gHome = g(phiHome);
    const gAway = g(phiAway);

    // Expected scores (0-1)
    const expectedHome = E(muHome, muAway, phiAway);
    const expectedAway = E(muAway, muHome, phiHome);

    // Convert to 1X2 probabilities
    // Use Bradley-Terry model with Glicko ratings
    const homeStrength = Math.pow(10, home.rating / 400);
    const awayStrength = Math.pow(10, away.rating / 400);
    const totalStrength = homeStrength + awayStrength;

    // Base probabilities from ratings
    let homeWinProb = homeStrength / totalStrength;
    let awayWinProb = awayStrength / totalStrength;

    // Adjust for typical draw rate in football (~26%)
    // Higher combined uncertainty = slightly higher draw probability
    const combinedUncertainty = Math.sqrt(phiHome * phiHome + phiAway * phiAway);
    const baseDrawRate = 0.26;
    const uncertaintyBoost = Math.min(0.05, combinedUncertainty * 0.1);
    const drawProb = baseDrawRate + uncertaintyBoost;

    // Scale H/A to fit
    const combatProb = 1 - drawProb;
    homeWinProb = homeWinProb * combatProb;
    awayWinProb = awayWinProb * combatProb;

    // Normalize
    const total = homeWinProb + drawProb + awayWinProb;

    return {
      homeRating: home,
      awayRating: away,
      expectedHome,
      expectedAway,
      homeWinProb: homeWinProb / total,
      drawProb: drawProb / total,
      awayWinProb: awayWinProb / total,
      uncertainty: combinedUncertainty,
    };
  }

  /**
   * Update ratings after a match result
   *
   * @param homeKey - Home team identifier
   * @param awayKey - Away team identifier
   * @param homeGoals - Goals scored by home
   * @param awayGoals - Goals scored by away
   * @param matchDate - Date of match
   */
  updateMatch(
    homeKey: string,
    awayKey: string,
    homeGoals: number,
    awayGoals: number,
    matchDate: Date,
  ): void {
    // Get ratings with decay
    const home = this.getRatingWithDecay(homeKey, matchDate);
    const away = this.getRatingWithDecay(awayKey, matchDate);

    // Determine match outcome (1 = win, 0.5 = draw, 0 = loss)
    const homeScore = homeGoals > awayGoals ? 1 : homeGoals === awayGoals ? 0.5 : 0;
    const awayScore = 1 - homeScore;

    // Update home team (opponent is away)
    const newHome = this.updateRating(home, away, homeScore);
    newHome.lastPlayed = matchDate;

    // Update away team (opponent is home)
    const newAway = this.updateRating(away, home, awayScore);
    newAway.lastPlayed = matchDate;

    this.ratings.set(homeKey, newHome);
    this.ratings.set(awayKey, newAway);
  }

  /**
   * Apply rating decay for time since last match.
   * Uncertainty (φ) increases when team hasn't played.
   */
  private getRatingWithDecay(teamKey: string, currentDate: Date): Glicko2Rating {
    const rating = this.getRating(teamKey);

    // Days since last match
    const daysSince = Math.floor(
      (currentDate.getTime() - rating.lastPlayed.getTime()) / (1000 * 60 * 60 * 24)
    );

    if (daysSince <= 0) return rating;

    // Increase RD based on time (Glicko-2 decay)
    // After ~6 months of inactivity, RD approaches initial value
    const c = Math.sqrt(
      (INITIAL_RD * INITIAL_RD - 50 * 50) / (365 / 2)
    );

    const newRD = Math.min(
      INITIAL_RD,
      Math.sqrt(rating.deviation * rating.deviation + c * c * daysSince)
    );

    return {
      ...rating,
      deviation: newRD,
    };
  }

  /**
   * Core Glicko-2 rating update algorithm
   */
  private updateRating(
    player: Glicko2Rating,
    opponent: Glicko2Rating,
    score: number, // 1 = win, 0.5 = draw, 0 = loss
  ): Glicko2Rating {
    // Convert to Glicko-2 scale
    const mu = (player.rating - INITIAL_RATING) / SCALE;
    const phi = player.deviation / SCALE;
    const sigma = player.volatility;

    const muOpp = (opponent.rating - INITIAL_RATING) / SCALE;
    const phiOpp = opponent.deviation / SCALE;

    // Step 3: Compute variance v
    const gOpp = g(phiOpp);
    const eVal = E(mu, muOpp, phiOpp);
    const v = 1 / (gOpp * gOpp * eVal * (1 - eVal));

    // Step 4: Compute delta
    const delta = v * gOpp * (score - eVal);

    // Step 5: Compute new volatility σ'
    const newSigma = this.computeNewVolatility(sigma, phi, v, delta);

    // Step 6: Update RD
    const phiStar = Math.sqrt(phi * phi + newSigma * newSigma);

    // Step 7: New RD and rating
    const newPhi = 1 / Math.sqrt(1 / (phiStar * phiStar) + 1 / v);
    const newMu = mu + newPhi * newPhi * gOpp * (score - eVal);

    // Convert back to Glicko-1 scale
    return {
      rating: newMu * SCALE + INITIAL_RATING,
      deviation: newPhi * SCALE,
      volatility: newSigma,
      lastPlayed: player.lastPlayed,
    };
  }

  /**
   * Step 5: Iterative algorithm to find new volatility
   * Uses Illinois algorithm for root finding
   */
  private computeNewVolatility(
    sigma: number,
    phi: number,
    v: number,
    delta: number,
  ): number {
    const a = Math.log(sigma * sigma);
    const phiSq = phi * phi;
    const deltaSq = delta * delta;

    const f = (x: number): number => {
      const ex = Math.exp(x);
      const num = ex * (deltaSq - phiSq - v - ex);
      const den = 2 * Math.pow(phiSq + v + ex, 2);
      return num / den - (x - a) / (TAU * TAU);
    };

    // Find bounds
    let A = a;
    let B: number;

    if (deltaSq > phiSq + v) {
      B = Math.log(deltaSq - phiSq - v);
    } else {
      let k = 1;
      while (f(a - k * TAU) < 0) {
        k++;
      }
      B = a - k * TAU;
    }

    // Illinois algorithm
    let fA = f(A);
    let fB = f(B);

    while (Math.abs(B - A) > CONVERGENCE_TOLERANCE) {
      const C = A + (A - B) * fA / (fB - fA);
      const fC = f(C);

      if (fC * fB <= 0) {
        A = B;
        fA = fB;
      } else {
        fA = fA / 2;
      }

      B = C;
      fB = fC;
    }

    return Math.exp(A / 2);
  }

  /**
   * Process all matches in chronological order
   */
  processMatches(
    matches: Array<{
      homeTeam: string;
      awayTeam: string;
      homeGoals: number;
      awayGoals: number;
      date: Date;
      league: string;
    }>,
  ): void {
    // Sort by date
    const sorted = [...matches].sort((a, b) => a.date.getTime() - b.date.getTime());

    for (const match of sorted) {
      const homeKey = `${match.league}:${match.homeTeam}`;
      const awayKey = `${match.league}:${match.awayTeam}`;
      this.updateMatch(homeKey, awayKey, match.homeGoals, match.awayGoals, match.date);
    }
  }

  /**
   * Get all current ratings
   */
  getAllRatings(): Map<string, Glicko2Rating> {
    return new Map(this.ratings);
  }

  /**
   * Reset a team's uncertainty (e.g., after major roster changes)
   */
  resetUncertainty(teamKey: string, newRD: number = INITIAL_RD * 0.7): void {
    const rating = this.getRating(teamKey);
    rating.deviation = newRD;
    this.ratings.set(teamKey, rating);
  }
}

// ─── Helper Functions ───

/**
 * g(φ) - Reduces impact of opponent's rating based on their uncertainty
 */
function g(phi: number): number {
  return 1 / Math.sqrt(1 + 3 * phi * phi / (Math.PI * Math.PI));
}

/**
 * E(μ, μj, φj) - Expected score against opponent
 */
function E(mu: number, muOpp: number, phiOpp: number): number {
  return 1 / (1 + Math.exp(-g(phiOpp) * (mu - muOpp)));
}

/**
 * Singleton tracker for use across feature engineering
 */
let globalTracker: Glicko2Tracker | null = null;

export function getGlobalGlicko2Tracker(): Glicko2Tracker {
  if (!globalTracker) {
    globalTracker = new Glicko2Tracker();
  }
  return globalTracker;
}

export function resetGlobalGlicko2Tracker(): void {
  globalTracker = new Glicko2Tracker();
}
