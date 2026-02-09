#!/bin/bash

# COMPREHENSIVE TEST SUITE
# Runs all diagnostic and calibration steps

echo "===================================================================="
echo "STEP 1: Train model WITHOUT market data (Closing Line Test)"
echo "===================================================================="
python d:\BetAI\v2\ml\train_residual_no_market.py

echo ""
echo "===================================================================="
echo "STEP 2: Apply Platt Scaling (Calibration)"
echo "===================================================================="
python d:\BetAI\v2\ml\apply_calibration.py

echo ""
echo "===================================================================="
echo "STEP 3: Run Backtest with Trained Model"
echo "===================================================================="
echo "Running with EDGE >= 8% filter..."
cd d:\BetAI\v2
npm run backtest -- --minEdge=0.08 --kellyFraction=0.25

echo ""
echo "===================================================================="
echo "STEP 4: Comprehensive Diagnostics"
echo "===================================================================="
echo "Testing:"
echo "  - Brier on high-edge bets only"
echo "  - Segmentation analysis"
echo "  - Calibration quality"
echo ""
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py

echo ""
echo "===================================================================="
echo "TESTS COMPLETE"
echo "===================================================================="
echo ""
echo "Decision Tree:"
echo "  IF Brier on 8%+ edge bets < Market Brier:"
echo "    ✅ PASS - Model has real edge, ready for live"
echo "  ELSE:"
echo "    ❌ FAIL - Need data improvements (real xG, injuries, lineups)"
echo ""
