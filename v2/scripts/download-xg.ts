#!/usr/bin/env npx tsx
/**
 * Download xG data from FBRef (Sports Reference)
 * Public data: https://fbref.com
 * No login required
 */

import { scrapeAll, loadCachedXG } from '../data/scrapers/fbref.js';

async function main() {
  console.log('=== BetAI v2 - Download FBRef xG ===\n');

  const startYear = 2015;
  const endYear = 2024;

  console.log(`Downloading xG data: ${startYear} to ${endYear}`);
  console.log('Source: fbref.com (public, no login)\n');

  try {
    const matches = await scrapeAll(undefined, startYear, endYear);
    console.log(`\n✅ Downloaded ${matches.length} matches with xG data`);
  } catch (error) {
    console.error('\n❌ Download failed, attempting to load cached data...\n');
    const cached = await loadCachedXG();
    console.log(`\n✅ Loaded ${cached.length} matches from cache`);

    if (cached.length === 0) {
      console.error('No cached data available. Please check network and try again.');
      process.exit(1);
    }
  }
}

main().catch(console.error);
