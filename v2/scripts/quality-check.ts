#!/usr/bin/env tsx
/**
 * Quality Check: Analyze all downloaded CSVs for data completeness,
 * odds coverage, closing odds availability, and segmentation readiness.
 */

import { loadAllMatches } from '../data/scrapers/football-data.js';
import { LEAGUES, getLeagueTier } from '../src/config.js';
import type { LeagueCode } from '../src/types/index.js';
import type { Match } from '../src/types/index.js';

interface LeagueStats {
  league: string;
  name: string;
  tier: number;
  totalMatches: number;
  withOdds: number;
  withClosingOdds: number;
  bookmakerCoverage: Record<string, number>;
  seasonRange: string;
  seasonsCount: number;
}

interface OddsBucket {
  range: string;
  min: number;
  max: number;
  count: number;
  homeWinRate: number;
  drawRate: number;
  awayWinRate: number;
  impliedProbAvg: number;
  actualWinRate: number;
  bias: number; // actual - implied (positive = market underestimates)
}

async function main() {
  console.log('=== BetAI v2 - Data Quality Check ===\n');

  const allLeagues = Object.keys(LEAGUES) as LeagueCode[];
  const matches = await loadAllMatches(allLeagues);

  console.log(`\nTotal matches loaded: ${matches.length}\n`);

  // ─── 1. Per-league breakdown ───
  console.log('═══════════════════════════════════════════════════════════════════════════');
  console.log('1. PER-LEAGUE BREAKDOWN');
  console.log('═══════════════════════════════════════════════════════════════════════════\n');

  const byLeague = new Map<string, Match[]>();
  for (const m of matches) {
    const arr = byLeague.get(m.league) || [];
    arr.push(m);
    byLeague.set(m.league, arr);
  }

  const leagueStats: LeagueStats[] = [];

  for (const [code, leagueMatches] of byLeague) {
    const league = LEAGUES[code];
    if (!league) continue;

    const withOdds = leagueMatches.filter(m => m.oddsHome && m.oddsDraw && m.oddsAway).length;
    const withClosing = leagueMatches.filter(m => m.closingOddsHome && m.closingOddsDraw && m.closingOddsAway).length;

    // Bookmaker coverage
    const bkCoverage: Record<string, number> = {};
    for (const m of leagueMatches) {
      if (m.oddsByBookmaker) {
        for (const bk of Object.keys(m.oddsByBookmaker)) {
          bkCoverage[bk] = (bkCoverage[bk] || 0) + 1;
        }
      }
    }

    const seasons = new Set(leagueMatches.map(m => m.season));
    const sortedSeasons = [...seasons].sort();

    leagueStats.push({
      league: code,
      name: league.name,
      tier: getLeagueTier(code),
      totalMatches: leagueMatches.length,
      withOdds,
      withClosingOdds: withClosing,
      bookmakerCoverage: bkCoverage,
      seasonRange: `${sortedSeasons[0]} to ${sortedSeasons[sortedSeasons.length - 1]}`,
      seasonsCount: seasons.size,
    });
  }

  // Sort by tier then matches
  leagueStats.sort((a, b) => a.tier - b.tier || b.totalMatches - a.totalMatches);

  // Print table
  console.log(pad('League', 25) + pad('Tier', 5) + pad('Matches', 9) + pad('w/Odds', 9) + pad('%Odds', 8) + pad('w/Closing', 11) + pad('%Closing', 10) + 'Seasons');
  console.log('-'.repeat(105));

  let totalMatches = 0, totalWithOdds = 0, totalWithClosing = 0;

  for (const s of leagueStats) {
    const pctOdds = s.totalMatches > 0 ? ((s.withOdds / s.totalMatches) * 100).toFixed(1) : '0';
    const pctClosing = s.totalMatches > 0 ? ((s.withClosingOdds / s.totalMatches) * 100).toFixed(1) : '0';
    console.log(
      pad(s.name, 25) +
      pad(String(s.tier), 5) +
      pad(String(s.totalMatches), 9) +
      pad(String(s.withOdds), 9) +
      pad(`${pctOdds}%`, 8) +
      pad(String(s.withClosingOdds), 11) +
      pad(`${pctClosing}%`, 10) +
      `${s.seasonsCount} (${s.seasonRange})`
    );
    totalMatches += s.totalMatches;
    totalWithOdds += s.withOdds;
    totalWithClosing += s.withClosingOdds;
  }

  console.log('-'.repeat(105));
  console.log(
    pad('TOTAL', 25) +
    pad('', 5) +
    pad(String(totalMatches), 9) +
    pad(String(totalWithOdds), 9) +
    pad(`${((totalWithOdds / totalMatches) * 100).toFixed(1)}%`, 8) +
    pad(String(totalWithClosing), 11) +
    pad(`${((totalWithClosing / totalMatches) * 100).toFixed(1)}%`, 10)
  );

  // ─── 2. Bookmaker coverage ───
  console.log('\n═══════════════════════════════════════════════════════════════════════════');
  console.log('2. BOOKMAKER COVERAGE (matches with full H/D/A odds)');
  console.log('═══════════════════════════════════════════════════════════════════════════\n');

  const globalBk: Record<string, number> = {};
  for (const m of matches) {
    if (m.oddsByBookmaker) {
      for (const bk of Object.keys(m.oddsByBookmaker)) {
        globalBk[bk] = (globalBk[bk] || 0) + 1;
      }
    }
  }

  const sortedBk = Object.entries(globalBk).sort((a, b) => b[1] - a[1]);
  for (const [bk, count] of sortedBk) {
    const pct = ((count / totalMatches) * 100).toFixed(1);
    console.log(`  ${pad(bk, 8)} ${pad(String(count), 8)} matches  (${pct}%)`);
  }

  // ─── 3. Closing odds by season ───
  console.log('\n═══════════════════════════════════════════════════════════════════════════');
  console.log('3. CLOSING ODDS AVAILABILITY BY SEASON');
  console.log('═══════════════════════════════════════════════════════════════════════════\n');

  const bySeason = new Map<string, { total: number; closing: number }>();
  for (const m of matches) {
    const s = m.season;
    const entry = bySeason.get(s) || { total: 0, closing: 0 };
    entry.total++;
    if (m.closingOddsHome && m.closingOddsDraw && m.closingOddsAway) entry.closing++;
    bySeason.set(s, entry);
  }

  const sortedSeasons = [...bySeason.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  console.log(pad('Season', 12) + pad('Matches', 10) + pad('w/Closing', 12) + '%Closing');
  console.log('-'.repeat(50));
  for (const [season, data] of sortedSeasons) {
    const pct = data.total > 0 ? ((data.closing / data.total) * 100).toFixed(1) : '0';
    console.log(pad(season, 12) + pad(String(data.total), 10) + pad(String(data.closing), 12) + `${pct}%`);
  }

  // ─── 4. Odds range segmentation readiness ───
  console.log('\n═══════════════════════════════════════════════════════════════════════════');
  console.log('4. SEGMENTATION READINESS (matches with odds, by home odds bucket)');
  console.log('═══════════════════════════════════════════════════════════════════════════\n');

  const matchesWithOdds = matches.filter(m => m.oddsHome && m.oddsDraw && m.oddsAway);

  const buckets: OddsBucket[] = [
    { range: '1.00-1.50', min: 1.00, max: 1.50, count: 0, homeWinRate: 0, drawRate: 0, awayWinRate: 0, impliedProbAvg: 0, actualWinRate: 0, bias: 0 },
    { range: '1.50-2.00', min: 1.50, max: 2.00, count: 0, homeWinRate: 0, drawRate: 0, awayWinRate: 0, impliedProbAvg: 0, actualWinRate: 0, bias: 0 },
    { range: '2.00-2.50', min: 2.00, max: 2.50, count: 0, homeWinRate: 0, drawRate: 0, awayWinRate: 0, impliedProbAvg: 0, actualWinRate: 0, bias: 0 },
    { range: '2.50-3.50', min: 2.50, max: 3.50, count: 0, homeWinRate: 0, drawRate: 0, awayWinRate: 0, impliedProbAvg: 0, actualWinRate: 0, bias: 0 },
    { range: '3.50+',     min: 3.50, max: 999,  count: 0, homeWinRate: 0, drawRate: 0, awayWinRate: 0, impliedProbAvg: 0, actualWinRate: 0, bias: 0 },
  ];

  for (const m of matchesWithOdds) {
    const odds = m.oddsHome!;
    for (const b of buckets) {
      if (odds >= b.min && odds < b.max) {
        b.count++;
        if (m.ftResult === 'H') b.homeWinRate++;
        if (m.ftResult === 'D') b.drawRate++;
        if (m.ftResult === 'A') b.awayWinRate++;
        b.impliedProbAvg += 1 / odds;
        break;
      }
    }
  }

  console.log(pad('Odds Range', 14) + pad('Matches', 10) + pad('H Win%', 9) + pad('D%', 8) + pad('A Win%', 9) + pad('Implied%', 10) + pad('Actual%', 10) + 'Bias');
  console.log('-'.repeat(85));

  for (const b of buckets) {
    if (b.count === 0) continue;
    const hPct = (b.homeWinRate / b.count * 100).toFixed(1);
    const dPct = (b.drawRate / b.count * 100).toFixed(1);
    const aPct = (b.awayWinRate / b.count * 100).toFixed(1);
    const impliedAvg = (b.impliedProbAvg / b.count * 100).toFixed(1);
    const actualPct = (b.homeWinRate / b.count * 100).toFixed(1);
    const bias = (b.homeWinRate / b.count - b.impliedProbAvg / b.count) * 100;
    const biasStr = bias >= 0 ? `+${bias.toFixed(2)}%` : `${bias.toFixed(2)}%`;
    console.log(
      pad(b.range, 14) +
      pad(String(b.count), 10) +
      pad(`${hPct}%`, 9) +
      pad(`${dPct}%`, 8) +
      pad(`${aPct}%`, 9) +
      pad(`${impliedAvg}%`, 10) +
      pad(`${actualPct}%`, 10) +
      biasStr
    );
  }

  // ─── 5. Segmentation by bet type (H/D/A) ───
  console.log('\n═══════════════════════════════════════════════════════════════════════════');
  console.log('5. SEGMENTATION BY BET TYPE (all matches with odds)');
  console.log('═══════════════════════════════════════════════════════════════════════════\n');

  const betTypes = [
    { label: 'Home (H)', getOdds: (m: Match) => m.oddsHome, result: 'H' as const },
    { label: 'Draw (D)', getOdds: (m: Match) => m.oddsDraw, result: 'D' as const },
    { label: 'Away (A)', getOdds: (m: Match) => m.oddsAway, result: 'A' as const },
  ];

  for (const bt of betTypes) {
    const withOdds = matchesWithOdds.filter(m => bt.getOdds(m));
    const wins = withOdds.filter(m => m.ftResult === bt.result).length;
    const totalImplied = withOdds.reduce((sum, m) => sum + 1 / bt.getOdds(m)!, 0);
    const winRate = (wins / withOdds.length * 100).toFixed(2);
    const avgImplied = (totalImplied / withOdds.length * 100).toFixed(2);
    const bias = (wins / withOdds.length - totalImplied / withOdds.length) * 100;
    const biasStr = bias >= 0 ? `+${bias.toFixed(2)}%` : `${bias.toFixed(2)}%`;
    console.log(`  ${pad(bt.label, 12)} ${pad(String(withOdds.length) + ' matches', 16)} Win: ${winRate}%  Implied: ${avgImplied}%  Bias: ${biasStr}`);
  }

  // ─── 6. Tier summary ───
  console.log('\n═══════════════════════════════════════════════════════════════════════════');
  console.log('6. TIER SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════════════════\n');

  for (const tier of [1, 2, 3] as const) {
    const tierMatches = matches.filter(m => getLeagueTier(m.league) === tier);
    const withOdds = tierMatches.filter(m => m.oddsHome && m.oddsDraw && m.oddsAway).length;
    const withClosing = tierMatches.filter(m => m.closingOddsHome).length;
    console.log(`  Tier ${tier}: ${tierMatches.length} matches | ${withOdds} with odds (${(withOdds/tierMatches.length*100).toFixed(1)}%) | ${withClosing} with closing (${(withClosing/tierMatches.length*100).toFixed(1)}%)`);
  }

  // ─── 7. Segmentation power check ───
  console.log('\n═══════════════════════════════════════════════════════════════════════════');
  console.log('7. SEGMENTATION POWER (odds_bucket x bet_type = 15 segments)');
  console.log('═══════════════════════════════════════════════════════════════════════════\n');

  const MIN_THRESHOLD = 500;
  let segOk = 0, segWarn = 0;

  console.log(pad('Segment', 22) + pad('Count', 8) + 'Status');
  console.log('-'.repeat(45));

  for (const b of buckets) {
    for (const bt of betTypes) {
      const segMatches = matchesWithOdds.filter(m => {
        const odds = bt.getOdds(m);
        return odds && odds >= b.min && odds < b.max;
      });
      const status = segMatches.length >= MIN_THRESHOLD ? 'OK' : 'LOW';
      if (segMatches.length >= MIN_THRESHOLD) segOk++; else segWarn++;
      console.log(pad(`${b.range} x ${bt.label}`, 22) + pad(String(segMatches.length), 8) + status);
    }
  }

  console.log('-'.repeat(45));
  console.log(`\nSegments OK (>=${MIN_THRESHOLD}): ${segOk}/15 | Low: ${segWarn}/15`);
  console.log(`\nMinimum for 95% CI: 384 samples per segment`);
}

function pad(str: string, len: number): string {
  return str.padEnd(len);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
