#!/usr/bin/env tsx
/**
 * Download all historical CSV data from football-data.co.uk
 *
 * Usage:
 *   npx tsx scripts/download-data.ts              # All leagues
 *   npx tsx scripts/download-data.ts --league EPL  # Single league
 */

import { downloadAll } from '../data/scrapers/football-data.js';
import type { LeagueCode } from '../src/types/index.js';

async function main() {
  console.log('=== BetAI v2 - Data Download ===\n');
  console.log('Source: football-data.co.uk');
  console.log('Leagues: 22 European divisions (Tier 1-3)');
  console.log('Seasons: 2010-11 to 2024-25 (15 seasons)');

  // Parse --league argument
  const leagueArg = process.argv.find(a => a.startsWith('--league'));
  let leagues: LeagueCode[] | undefined;
  if (leagueArg) {
    const idx = process.argv.indexOf(leagueArg);
    const val = leagueArg.includes('=')
      ? leagueArg.split('=')[1]
      : process.argv[idx + 1];
    if (val) {
      leagues = [val.toUpperCase() as LeagueCode];
      console.log(`Filtering: ${leagues.join(', ')}`);
    }
  }

  console.log('');

  const result = await downloadAll(leagues);

  console.log('\n=== Summary ===');
  console.log(`Downloaded: ${result.downloaded}`);
  console.log(`Failed:     ${result.failed}`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
