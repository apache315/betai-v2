#!/usr/bin/env tsx
/**
 * Run backtest on historical data.
 *
 * Uses walk-forward validation: for each month, train on all prior data,
 * predict on that month, track bets and P&L.
 *
 * Usage:
 *   npx tsx scripts/backtest.ts
 *   npx tsx scripts/backtest.ts --start 2023-01-01 --end 2024-12-31
 */

import { join } from 'path';
import { fileURLToPath } from 'url';
import { createReadStream } from 'fs';
import { createInterface } from 'readline';
import { stat } from 'fs/promises';
import { runBacktest, printBacktestReport } from '../backtest/engine.js';
import type { MatchFeatures } from '../src/types/index.js';

/**
 * Stream-parse a JSON array file where each element is on its own line.
 * Handles files too large for JSON.parse() in one shot.
 */
async function loadFeaturesStreaming(filePath: string): Promise<MatchFeatures[]> {
  const fileInfo = await stat(filePath);
  const sizeMB = fileInfo.size / 1e6;
  console.log(`  File size: ${sizeMB.toFixed(0)} MB — using streaming parser`);

  const features: MatchFeatures[] = [];
  const rl = createInterface({
    input: createReadStream(filePath, { encoding: 'utf-8' }),
    crlfDelay: Infinity,
  });

  let count = 0;
  for await (const line of rl) {
    const trimmed = line.trim().replace(/,\s*$/, '');
    if (!trimmed || trimmed === '[' || trimmed === ']' || trimmed === '[]') continue;

    try {
      const match = JSON.parse(trimmed) as MatchFeatures;
      features.push(match);
      count++;
      if (count % 50000 === 0) {
        console.log(`    Loaded ${count} matches...`);
      }
    } catch {
      // Skip malformed lines
    }
  }

  console.log(`    Total: ${count} matches`);
  return features;
}

const __dirname = join(fileURLToPath(import.meta.url), '..');

async function main() {
  console.log('=== BetAI v2 - Backtest ===\n');

  // Parse arguments
  const args = process.argv.slice(2);
  let startDate: Date | undefined;
  let endDate: Date | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--start' && args[i + 1]) {
      startDate = new Date(args[i + 1]);
    }
    if (args[i] === '--end' && args[i + 1]) {
      endDate = new Date(args[i + 1]);
    }
  }

  // Load features
  const featuresPath = join(__dirname, '..', 'data', 'processed', 'features.json');
  console.log(`Loading features from: ${featuresPath}`);

  let features: MatchFeatures[];
  try {
    features = await loadFeaturesStreaming(featuresPath);
  } catch (err) {
    console.error('Failed to load features. Run build-features.ts first.');
    process.exit(1);
  }

  console.log(`Loaded ${features.length} matches with features`);

  // Enhanced model using Glicko-2, Fatigue, and Style features
  // Still not XGBoost, but demonstrates the new feature value

  console.log('\nGenerating predictions (enhanced model with Glicko-2 + Fatigue + Style)...');

  const predictions = new Map<string, { home: number; draw: number; away: number }>();

  for (const match of features) {
    const f = match.features;

    // 1. Market implied (strongest signal)
    let mktHome = f['odds_implied_home'] || 0.45;
    let mktDraw = f['odds_implied_draw'] || 0.27;
    let mktAway = f['odds_implied_away'] || 0.28;

    // 2. Glicko-2 based probabilities (if available)
    const glickoHome = f['glicko_home_win_prob'] || mktHome;
    const glickoDraw = f['glicko_draw_prob'] || mktDraw;
    const glickoAway = f['glicko_away_win_prob'] || mktAway;

    // 3. Form adjustment
    const formAdj = (f['diff_ppg_5'] || 0) * 0.03;

    // 4. Fatigue adjustment
    // Positive fatigue_diff = home is LESS fatigued (advantage)
    const fatigueDiff = f['fatigue_diff'] || 0;
    const fatigueAdj = fatigueDiff * 0.05;  // 5% per unit of fatigue difference

    // 5. Thursday-Sunday squeeze (Europa League effect)
    const thuSunHome = f['fatigue_home_thu_sun_squeeze'] || 0;
    const thuSunAway = f['fatigue_away_thu_sun_squeeze'] || 0;
    const thuSunAdj = (thuSunAway - thuSunHome) * 0.03;  // 3% disadvantage if squeezed

    // 6. Style matchup adjustment
    const styleHomeWR = f['style_matchup_home_wr'] || 0.45;
    const styleAdj = (styleHomeWR - 0.45) * 0.1;  // Small adjustment based on style matchup history

    // Blend: 50% market + 30% Glicko + 20% adjustments
    let home = mktHome * 0.5 + glickoHome * 0.3 + 0.2 * (mktHome + formAdj + fatigueAdj + thuSunAdj + styleAdj);
    let draw = mktDraw * 0.5 + glickoDraw * 0.3 + 0.2 * mktDraw;
    let away = mktAway * 0.5 + glickoAway * 0.3 + 0.2 * (mktAway - formAdj - fatigueAdj - thuSunAdj - styleAdj);

    // Ensure non-negative
    home = Math.max(0.05, home);
    draw = Math.max(0.05, draw);
    away = Math.max(0.05, away);

    // Normalize
    const total = home + draw + away;
    predictions.set(match.matchId, {
      home: home / total,
      draw: draw / total,
      away: away / total,
    });
  }

  console.log(`Generated ${predictions.size} predictions`);

  // Run backtest
  console.log('\nRunning backtest...');
  const result = runBacktest(features, predictions, {
    startDate,
    endDate,
    minEdge: 0.03, // 3% edge for this simple model (it's mostly market-based)
    kellyFraction: 0.25,
  });

  // Print report
  printBacktestReport(result);

  console.log('\n[Note: This uses an enhanced heuristic model with Glicko-2 + Fatigue + Style.]');
  console.log('[For optimal results, train the full XGBoost model: npx tsx scripts/train-model.ts]');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
