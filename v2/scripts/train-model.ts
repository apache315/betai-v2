#!/usr/bin/env tsx
/**
 * Train XGBoost model.
 *
 * Requires Python with: xgboost, scikit-learn, optuna, pandas, numpy
 *   pip install xgboost scikit-learn optuna pandas numpy
 *
 * Usage:
 *   npx tsx scripts/train-model.ts
 *   npx tsx scripts/train-model.ts --validate-only
 */

import { join } from 'path';
import { fileURLToPath } from 'url';
import { trainModel } from '../ml/model.js';

const __dirname = join(fileURLToPath(import.meta.url), '..');

async function main() {
  console.log('=== BetAI v2 - Train Model ===\n');

  const featuresPath = join(__dirname, '..', 'data', 'processed', 'features.json');
  const modelPath = join(__dirname, '..', 'data', 'processed', 'model.json');

  const validateOnly = process.argv.includes('--validate-only');

  // Check for --selected-features flag
  const sfIdx = process.argv.indexOf('--selected-features');
  const selectedFeatures = sfIdx !== -1 && process.argv[sfIdx + 1]
    ? process.argv[sfIdx + 1]
    : undefined;

  console.log(`Features: ${featuresPath}`);
  console.log(`Output:   ${modelPath}`);
  console.log(`Mode:     ${validateOnly ? 'Validation only' : 'Full training'}`);
  if (selectedFeatures) console.log(`Selected: ${selectedFeatures}`);
  console.log('');

  const result = await trainModel(featuresPath, modelPath, {
    validateOnly,
    nSplits: 5,
    selectedFeatures,
  });

  if (result.success) {
    console.log('\n✅ Training completed');
    console.log(`   Validation Brier: ${result.validationBrier.toFixed(4)}`);
  } else {
    console.error('\n❌ Training failed:', result.error);
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
