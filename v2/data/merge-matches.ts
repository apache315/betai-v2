/**
 * Merge & Deduplicate Match Sources
 *
 * Combines v2 football-data.co.uk matches with archive matches.
 * For overlapping periods (5 primary leagues, 2015-2025),
 * v2 matches take priority (they have Pinnacle odds).
 *
 * Archive Elo ratings are preserved on v2 matches when a matching
 * archive row exists.
 */

import type { Match } from '../src/types/index.js';

/**
 * Merge two match arrays, deduplicating by match ID.
 * v2 matches take priority over archive matches.
 * Archive Elo ratings are transferred to v2 matches when available.
 */
export function mergeMatchSources(
  v2Matches: Match[],
  archiveMatches: Match[],
): Match[] {
  // Build a map of v2 match IDs for fast lookup
  const v2Map = new Map<string, Match>();
  for (const m of v2Matches) {
    m.source = 'football-data';
    v2Map.set(m.id, m);
  }

  // Build archive map for Elo transfer
  const archiveMap = new Map<string, Match>();
  for (const m of archiveMatches) {
    archiveMap.set(m.id, m);
  }

  // Transfer archive Elo to v2 matches where available
  let eloTransferred = 0;
  for (const [id, v2Match] of v2Map) {
    const archiveMatch = archiveMap.get(id);
    if (archiveMatch && archiveMatch.archiveHomeElo != null) {
      v2Match.archiveHomeElo = archiveMatch.archiveHomeElo;
      v2Match.archiveAwayElo = archiveMatch.archiveAwayElo;
      eloTransferred++;
    }
  }

  // From archive, keep only matches NOT already in v2
  const uniqueArchive = archiveMatches.filter(m => !v2Map.has(m.id));

  // Merge and sort by date
  const merged = [...v2Matches, ...uniqueArchive];
  merged.sort((a, b) => a.date.getTime() - b.date.getTime());

  console.log(`Merge: ${v2Matches.length} v2 + ${uniqueArchive.length} archive-only = ${merged.length} total`);
  console.log(`  Duplicates removed: ${archiveMatches.length - uniqueArchive.length}`);
  console.log(`  Elo transferred to v2 matches: ${eloTransferred}`);

  return merged;
}
