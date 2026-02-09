#!/usr/bin/env tsx
/**
 * Generate predictions for upcoming matches.
 *
 * Fetches live odds from The Odds API (if key configured),
 * loads trained model, generates value bets.
 *
 * Usage:
 *   npx tsx scripts/predict-today.ts
 *   npx tsx scripts/predict-today.ts --league EPL
 */

import { fetchUpcomingOdds, checkQuota } from '../data/scrapers/odds-api.js';
import { LEAGUES, ODDS_API_KEY } from '../src/config.js';
import { detectValueBets } from '../betting/value-detector.js';
import type { LeagueCode } from '../src/types/index.js';

async function main() {
  console.log('=== BetAI v2 - Today\'s Predictions ===\n');
  console.log(`Date: ${new Date().toISOString().slice(0, 10)}`);

  // Check API quota
  if (ODDS_API_KEY) {
    const quota = await checkQuota();
    if (quota) {
      console.log(`Odds API: ${quota.remaining} requests remaining\n`);
    }
  } else {
    console.log('⚠️  ODDS_API_KEY not set. Using mock data.\n');
    console.log('To get live odds, set ODDS_API_KEY environment variable.');
    console.log('Free tier: 500 requests/month at https://the-odds-api.com/\n');
  }

  // Parse league filter
  const leagueArg = process.argv.find(a => a.startsWith('--league'));
  let targetLeagues: LeagueCode[] = Object.keys(LEAGUES) as LeagueCode[];
  if (leagueArg) {
    const val = leagueArg.includes('=') ? leagueArg.split('=')[1] : process.argv[process.argv.indexOf(leagueArg) + 1];
    if (val) {
      targetLeagues = [val.toUpperCase() as LeagueCode];
    }
  }

  console.log(`Leagues: ${targetLeagues.join(', ')}\n`);

  // Fetch odds for each league
  for (const leagueCode of targetLeagues) {
    const league = LEAGUES[leagueCode];
    console.log(`\n--- ${league.name} ---`);

    const odds = await fetchUpcomingOdds(leagueCode);

    if (odds.length === 0) {
      console.log('No upcoming matches with odds found.');
      continue;
    }

    // For each match, simulate model prediction and find value bets
    // In production, you'd load the trained model here
    for (const match of odds) {
      console.log(`\n${match.homeTeam} vs ${match.awayTeam}`);
      console.log(`Kickoff: ${new Date(match.commenceTime).toLocaleString()}`);

      if (!match.h2h) {
        console.log('  No odds available');
        continue;
      }

      // Simple baseline model (in production, use XGBoost)
      // Here we just show the framework
      const impliedHome = 1 / match.h2h.home;
      const impliedDraw = 1 / match.h2h.draw;
      const impliedAway = 1 / match.h2h.away;
      const overround = impliedHome + impliedDraw + impliedAway;

      // Model: slightly favor home (baseline)
      // Real model would use features
      const modelHome = (impliedHome / overround) * 1.02; // 2% home boost
      const modelDraw = (impliedDraw / overround) * 0.98;
      const modelAway = (impliedAway / overround) * 1.00;
      const modelTotal = modelHome + modelDraw + modelAway;

      const modelProbs = {
        home: modelHome / modelTotal,
        draw: modelDraw / modelTotal,
        away: modelAway / modelTotal,
      };

      console.log(`  Market: H ${match.h2h.home.toFixed(2)} | D ${match.h2h.draw.toFixed(2)} | A ${match.h2h.away.toFixed(2)}`);
      console.log(`  Model:  H ${(modelProbs.home * 100).toFixed(0)}% | D ${(modelProbs.draw * 100).toFixed(0)}% | A ${(modelProbs.away * 100).toFixed(0)}%`);

      // Detect value bets
      const valueBets = detectValueBets(
        `${match.homeTeam}_${match.awayTeam}`,
        modelProbs,
        match.h2h,
        0.02, // 2% min edge for baseline model
      );

      if (valueBets.length > 0) {
        console.log(`  💰 VALUE BETS:`);
        for (const vb of valueBets) {
          console.log(`     ${vb.selection}: edge ${(vb.edge * 100).toFixed(1)}%, stake ${(vb.recommendedStake * 100).toFixed(1)}% bankroll`);
        }
      } else {
        console.log(`  No value bets found (edge < 2%)`);
      }
    }
  }

  console.log('\n' + '='.repeat(50));
  console.log('[Note: This uses a simple baseline model.]');
  console.log('[Train XGBoost for real predictions: npx tsx scripts/train-model.ts]');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
