#!/usr/bin/env tsx
/**
 * Train GNN model and run Bayesian ensemble.
 *
 * Pipeline:
 *   1. Train GNN → export team embeddings
 *   2. Run Bayesian ensemble (Market + XGBoost + GNN)
 *   3. Evaluate vs each source individually
 *
 * Usage:
 *   npx tsx scripts/train-gnn.ts
 *   npx tsx scripts/train-gnn.ts --validate-only
 */

import { spawn } from 'child_process';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { stat } from 'fs/promises';

const __dirname = join(fileURLToPath(import.meta.url), '..');

function runPython(script: string, args: string[]): Promise<number> {
  return new Promise((resolve, reject) => {
    const proc = spawn('python', [script, ...args], {
      stdio: 'inherit',
      cwd: join(__dirname, '..', 'ml'),
    });
    proc.on('close', (code) => resolve(code ?? 1));
    proc.on('error', reject);
  });
}

async function main() {
  console.log('=== BetAI v2 - GNN + Bayesian Ensemble Pipeline ===\n');

  const featuresPath = join(__dirname, '..', 'data', 'processed', 'features.json');
  const gnnEmbeddingsPath = join(__dirname, '..', 'data', 'processed', 'gnn_embeddings.json');
  const selectedFeaturesPath = join(__dirname, '..', 'data', 'processed', 'selected_features.json');
  const ensembleResultsPath = join(__dirname, '..', 'data', 'processed', 'ensemble_results.json');

  const validateOnly = process.argv.includes('--validate-only');
  const skipGnn = process.argv.includes('--skip-gnn');

  // Step 1: Train GNN
  if (!skipGnn) {
    console.log('--- Step 1: Train GNN ---\n');
    const gnnScript = join(__dirname, '..', 'ml', 'gnn_model.py');
    const gnnArgs = [
      '--data', featuresPath,
      '--output', gnnEmbeddingsPath,
      '--epochs', '50',
    ];
    if (validateOnly) gnnArgs.push('--validate-only');

    const gnnCode = await runPython(gnnScript, gnnArgs);
    if (gnnCode !== 0) {
      console.error('\nGNN training failed');
      process.exit(1);
    }
  } else {
    console.log('--- Skipping GNN (using existing embeddings) ---\n');
  }

  // Step 2: Run Bayesian Ensemble
  console.log('\n--- Step 2: Bayesian Ensemble ---\n');

  // Check if selected features exist
  let hasSelectedFeatures = false;
  try {
    await stat(selectedFeaturesPath);
    hasSelectedFeatures = true;
  } catch {}

  const ensembleScript = join(__dirname, '..', 'ml', 'bayesian_ensemble.py');
  const ensembleArgs = [
    '--features', featuresPath,
    '--gnn-embeddings', gnnEmbeddingsPath,
    '--output', ensembleResultsPath,
  ];
  if (hasSelectedFeatures) {
    ensembleArgs.push('--selected-features', selectedFeaturesPath);
  }

  const ensembleCode = await runPython(ensembleScript, ensembleArgs);
  if (ensembleCode !== 0) {
    console.error('\nEnsemble evaluation failed');
    process.exit(1);
  }

  console.log('\n=== Pipeline Complete ===');
  console.log(`GNN embeddings: ${gnnEmbeddingsPath}`);
  console.log(`Ensemble results: ${ensembleResultsPath}`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
