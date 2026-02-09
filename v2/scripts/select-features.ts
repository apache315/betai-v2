#!/usr/bin/env tsx
/**
 * Run Boruta + Feature Ablation selection
 *
 * Usage:
 *   npx tsx scripts/select-features.ts
 *   npx tsx scripts/select-features.ts --method boruta
 *   npx tsx scripts/select-features.ts --method ablation
 */

import { spawn } from 'child_process';
import { join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = join(fileURLToPath(import.meta.url), '..');

async function main() {
  const args = process.argv.slice(2);
  let method = 'both';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--method' && args[i + 1]) {
      method = args[i + 1];
    }
  }

  const featuresPath = join(__dirname, '..', 'data', 'processed', 'features.json');
  const outputPath = join(__dirname, '..', 'data', 'processed', 'selected_features.json');

  const pyArgs = [
    join(__dirname, '..', 'ml', 'feature_selection.py'),
    '--data', featuresPath,
    '--output', outputPath,
    '--method', method,
  ];

  console.log('=== BetAI v2 - Feature Selection ===\n');
  console.log(`Method: ${method}`);
  console.log(`Input: ${featuresPath}`);
  console.log(`Output: ${outputPath}\n`);

  return new Promise<void>((resolve, reject) => {
    const proc = spawn('python', pyArgs, {
      stdio: 'inherit',
      cwd: join(__dirname, '..', 'ml'),
    });

    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Feature selection failed with code ${code}`));
      } else {
        resolve();
      }
    });
  });
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
