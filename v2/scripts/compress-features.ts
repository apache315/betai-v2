#!/usr/bin/env npx tsx
/**
 * Compress features.json to Parquet for faster loading
 */

import { readFileSync } from 'fs';
import { writeFileSync } from 'fs/promises';
import { join } from 'path';

async function main() {
  console.log('=== Compressing features to Parquet ===\n');

  const featuresPath = 'd:\\BetAI\\v2\\data\\processed\\features.json';
  const outputPath = 'd:\\BetAI\\v2\\data\\processed\\features.parquet';

  try {
    console.log('Reading features.json...');
    const json = readFileSync(featuresPath, 'utf-8');
    const features = JSON.parse(json);

    console.log(`Loaded ${features.length} features`);

    // Create a CSV for now (easier than Parquet without dependencies)
    const csvPath = 'd:\\BetAI\\v2\\data\\processed\\features.csv';
    console.log('Converting to CSV...');

    const headers = Object.keys(features[0]).sort();
    const csv = [
      headers.join(','),
      ...features.map((f: any) =>
        headers
          .map((h) => {
            const val = f[h];
            if (val === null || val === undefined) return '';
            if (typeof val === 'string' && val.includes(',')) {
              return `"${val.replace(/"/g, '""')}"`;
            }
            if (typeof val === 'object') {
              return `"${JSON.stringify(val).replace(/"/g, '""')}"`;
            }
            return val;
          })
          .join(',')
      ),
    ].join('\n');

    console.log(`CSV size: ${(csv.length / 1024 / 1024).toFixed(1)} MB`);
    writeFileSync(csvPath, csv, 'utf-8');

    console.log(`\n✅ Saved to: ${csvPath}`);
    console.log('Use this CSV file for training instead of features.json');
  } catch (error: any) {
    console.error(`\n❌ Error: ${error.message}`);
    process.exit(1);
  }
}

main();
