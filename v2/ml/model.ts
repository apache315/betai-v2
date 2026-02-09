/**
 * XGBoost Model Wrapper
 *
 * Calls Python script for training and inference.
 * Model is serialized as JSON for portability.
 */

import { spawn } from 'child_process';
import { readFile, writeFile } from 'fs/promises';
import { join } from 'path';
import { fileURLToPath } from 'url';
import type { MatchFeatures } from '../src/types/index.js';

const __dirname = join(fileURLToPath(import.meta.url), '..');

export interface ModelPrediction {
  matchId: string;
  probHome: number;
  probDraw: number;
  probAway: number;
  confidence: number;
}

export interface TrainResult {
  success: boolean;
  modelPath: string;
  validationBrier: number;
  cvResults: any[];
  error?: string;
}

/**
 * Train XGBoost model via Python script
 */
export async function trainModel(
  featuresPath: string,
  outputPath: string,
  options: { validateOnly?: boolean; nSplits?: number; selectedFeatures?: string } = {},
): Promise<TrainResult> {
  const args = [
    join(__dirname, 'train.py'),
    '--data', featuresPath,
    '--output', outputPath,
  ];

  if (options.validateOnly) args.push('--validate-only');
  if (options.nSplits) args.push('--n-splits', String(options.nSplits));
  if (options.selectedFeatures) args.push('--selected-features', options.selectedFeatures);

  return new Promise((resolve) => {
    console.log(`[model] Training: python ${args.join(' ')}`);

    const proc = spawn('python', args, {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      const text = data.toString();
      stdout += text;
      process.stdout.write(text);
    });

    proc.stderr.on('data', (data) => {
      const text = data.toString();
      stderr += text;
      process.stderr.write(text);
    });

    proc.on('close', async (code) => {
      if (code !== 0) {
        resolve({
          success: false,
          modelPath: outputPath,
          validationBrier: -1,
          cvResults: [],
          error: stderr || `Process exited with code ${code}`,
        });
        return;
      }

      // Read metadata
      try {
        const metaPath = outputPath.replace('.json', '_meta.json');
        const meta = JSON.parse(await readFile(metaPath, 'utf-8'));
        resolve({
          success: true,
          modelPath: outputPath,
          validationBrier: meta.validation_brier,
          cvResults: [],
        });
      } catch {
        resolve({
          success: true,
          modelPath: outputPath,
          validationBrier: -1,
          cvResults: [],
        });
      }
    });
  });
}

/**
 * Save features to JSON for Python training
 */
export async function saveFeatures(
  features: MatchFeatures[],
  outputPath: string,
): Promise<void> {
  await writeFile(outputPath, JSON.stringify(features, null, 2), 'utf-8');
  console.log(`[model] Saved ${features.length} matches to ${outputPath}`);
}

/**
 * Python inference script for batch prediction
 */
const PREDICT_SCRIPT = `
import sys
import json
import xgboost as xgb
import numpy as np

def main():
    model_path = sys.argv[1]
    features_json = sys.argv[2]

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Load features
    data = json.loads(features_json)
    X = np.array([list(m['features'].values()) for m in data])

    # Predict
    probs = model.predict_proba(X)

    results = []
    for i, m in enumerate(data):
        results.append({
            'matchId': m['matchId'],
            'probHome': float(probs[i][0]),
            'probDraw': float(probs[i][1]),
            'probAway': float(probs[i][2]),
        })

    print(json.dumps(results))

if __name__ == '__main__':
    main()
`;

/**
 * Run batch prediction on features
 */
export async function predict(
  modelPath: string,
  features: MatchFeatures[],
): Promise<ModelPrediction[]> {
  const featuresJson = JSON.stringify(features);

  return new Promise((resolve, reject) => {
    const proc = spawn('python', ['-c', PREDICT_SCRIPT, modelPath, featuresJson], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });

    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Prediction failed: ${stderr}`));
        return;
      }

      try {
        const results: any[] = JSON.parse(stdout);
        resolve(results.map(r => ({
          matchId: r.matchId,
          probHome: r.probHome,
          probDraw: r.probDraw,
          probAway: r.probAway,
          confidence: computeConfidence(r.probHome, r.probDraw, r.probAway),
        })));
      } catch (err) {
        reject(new Error(`Failed to parse prediction output: ${stdout}`));
      }
    });
  });
}

/**
 * Confidence = how "peaked" the probability distribution is.
 * Max prob - uniform (0.33) / (1 - 0.33)
 */
function computeConfidence(h: number, d: number, a: number): number {
  const maxProb = Math.max(h, d, a);
  return (maxProb - 0.333) / (1 - 0.333);
}
