# 🎯 Implementazione Completa: Diagnostica & Calibrazione

## Cosa Ho Fatto

Ho tradotto la tua analisi esperta in **4 strumenti diagnostici** + **2 processi di calibrazione** per testare sistematicamente se il modello può essere salvato o serve ricominciare da dati migliori.

---

## 1️⃣ **CLOSING LINE TEST** (ml/train_residual_no_market.py)
Il test più critico: allena il modello SENZA le quote di mercato come feature.

**Logica**:
- Se il modello dipende solo da quote → R² crolla < 0.05 → ricomincia con real xG
- Se il modello ha insight vero → R² rimane >= 0.10 → procedi

**Esegui**:
```bash
python d:\BetAI\v2\ml\train_residual_no_market.py
```

**Output**: Metrica decisionale R² = X.XX

---

## 2️⃣ **PLATT SCALING CALIBRATION** (ml/apply_calibration.py)
Applica regressione logistica ai residui del modello per correggere distorsioni sistematiche.

**Effetto**: Brier improvement tipico 5-15% senza riallenare il modello

**Esegui**:
```bash
python d:\BetAI\v2\ml\apply_calibration.py
```

**Output**: `ml/models/platt_calibrator.pkl`

---

## 3️⃣ **GENERATE CALIBRATED PREDICTIONS** (scripts/generate-predictions.py)
Crea prediction file per il backtest usando il modello + calibrator + edge filter.

**Feature**: Integra automaticamente il Platt calibrator

**Esegui**:
```bash
python d:\BetAI\v2\scripts\generate-predictions.py
```

**Output**: `backtest/predictions_calibrated.json`

---

## 4️⃣ **COMPREHENSIVE DIAGNOSTICS** (scripts/comprehensive-diagnostics.py)
Suite di test che valuta TUTTE le dimensioni:

**Test Inclusi**:
- ✅ Brier overall vs Market Brier
- ✅ Brier su high-edge bets (3%, 5%, 8%, 10%, 12%)
- ✅ **TEST CRITICO**: Brier edge>=8% < Market Brier? → EDGE REALE
- ✅ Segmentazione per lega (dove funziona?)
- ✅ Segmentazione per quote range (low/mid/long)
- ✅ Calibrazione check (Platt curve, ECE)

**Esegui**:
```bash
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py
```

**Output**: Report completo con decisione PASS/FAIL

---

## 5️⃣ **BACKTEST CON EDGE FILTER** (npm run backtest)
Esegui backtest con il nuovo parametro `--minEdge=0.08` (da 5% a 8%).

**Scopo**: 
- Riduce i bet da 60-90% a ~8-15% (realistico)
- Testa se Brier migliora su subset selezionato

**Esegui**:
```bash
cd d:\BetAI\v2
npm run backtest -- --minEdge=0.08 --kellyFraction=0.25
```

---

## 📊 Flow Completo

```
┌─────────────────────────────────────────────┐
│ INPUT: features.json + trained model        │
└────────────────┬────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Closing Line Test? │ (no market data)
        │ R² >= 0.10?        │
        └────┬───────────┬───┘
             │           │
           YES           NO
             │           │
             ▼           ▼
        ┌─────────┐  STOP - Ricomincia
        │Procedi  │  con real xG
        └────┬────┘
             │
             ▼
      ┌─────────────────┐
      │ Platt Scaling   │ (calibrazione)
      │ Calibration     │
      └────────┬────────┘
               │
               ▼
      ┌────────────────────┐
      │ Generate Predictions│
      │ (calibrated)        │
      └────────┬────────────┘
               │
               ▼
      ┌────────────────────┐
      │ Backtest minEdge=8%│
      │ (edge filter)      │
      └────────┬────────────┘
               │
               ▼
      ┌────────────────────┐
      │ Comprehensive Test │
      │ (Brier, calib,seg) │
      └────────┬────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
    ✅ PASS      ❌ FAIL
    LIVE READY    DATA UPGRADE
```

---

## 🎯 Decisione Finale

Dopo i test, scegli UNA di queste:

### ✅ SCENARIO A: PASS - Deploy to Live
- Brier edge>=8% < Market Brier
- CLV >= +0.5%
- R² chiusura linea >= 0.10

→ **Deploy live, monitora 2 settimane, riaddestra settimanalmente**

### ⚠️ SCENARIO B: WEAK - Conditional Deployment
- Marginal Brier improvement (+0.003 to 0.010)
- CLV >= +0.2% ma < +0.5%
- Funziona 1 lega minore

→ **Riduci puntate 50%, edge filter 10%, monitora 1 mese**

### ❌ SCENARIO C: FAIL - Data Upgrade Required
- Brier overall > Market Brier
- R² chiusura linea < 0.05
- CLV <= 0%

→ **Stop live, acquista Real xG data ($100-2000), riaddestra, ritest**

---

## 📋 Prossimi Passaggi Immediati

**ORA**:
```bash
# Test 1: Chiudi il modello
python d:\BetAI\v2\ml\train_residual_no_market.py

# Scrivi il risultato R²: ___________
```

**Se R² >= 0.10**:
```bash
# Test 2: Calibrazione
python d:\BetAI\v2\ml\apply_calibration.py

# Test 3: Predictions
python d:\BetAI\v2\scripts\generate-predictions.py

# Test 4: Backtest
cd d:\BetAI\v2
npm run backtest -- --minEdge=0.08

# Test 5: Diagnostica COMPLETA
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py

# Scrivi i risultati e ti dico PASS/FAIL
```

**Se R² < 0.05**:
```bash
# ❌ STOP - Il modello è puro rumore
# Procedi a: Data Upgrade (Real xG API)
```

---

## 📚 Documentazione Completa

| File | Contenuto |
|------|-----------|
| [CALIBRATION_README.md](CALIBRATION_README.md) | Quick start + interpretazione risultati |
| [DIAGNOSTICS_PROTOCOL.md](DIAGNOSTICS_PROTOCOL.md) | Protocollo dettagliato + checklist |
| `ml/train_residual_no_market.py` | Codice closing line test |
| `ml/apply_calibration.py` | Codice calibrazione Platt |
| `scripts/generate-predictions.py` | Codice prediction generation |
| `scripts/comprehensive-diagnostics.py` | Codice test diagnostica |

---

## ⚡ TL;DR

Ho implementato il tuo piano diagnostico in 5 step automatizzati:

1. **Closing line test** → Scopri se il modello ha insight o dipende solo da quote
2. **Platt scaling** → Calibra le probabilità
3. **Edge filter** → Riduci bet da 90% a 10% (realistico)
4. **Comprehensive diagnostics** → Testa Brier, calibrazione, segmentazione
5. **Decisione finale** → PASS (live ready) / WEAK (conditional) / FAIL (upgrade data)

**Tempo**: 15-20 minuti per eseguire tutto

**Prossimo**: Condividi il risultato di `python ml/train_residual_no_market.py` (la metrica R²) e procediamo di conseguenza.

