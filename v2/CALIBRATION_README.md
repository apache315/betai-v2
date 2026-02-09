# Fase di Calibrazione & Validazione

## Panoramica
Hai identificato correttamente che il modello è **miscalibrato**: replica il mercato con distorsione sistematica anziché trovare un edge reale.

Questo protocollo testa se il modello può essere salvato con calibrazione o se serve rifare dati/features.

---

## 🚀 Come Procedere (5 Step)

### Step 1: Chiudi il Modello (Closing Line Test)
Allena il modello **SENZA quote di mercato** come feature per verificare se ha insight vero:

```bash
python d:\BetAI\v2\ml\train_residual_no_market.py
```

**Gate Check**:
- ✅ Se R² >= 0.10 → Il modello ha insight indipendente
- ❌ Se R² < 0.05 → Il modello dipende completamente da market data (quit qui, riprendi con real xG)

---

### Step 2: Calibrazione con Platt Scaling
Applica una regressione logistica ai residui del modello per correggere distorsioni sistematiche:

```bash
python d:\BetAI\v2\ml\apply_calibration.py
```

**Cosa aspettarsi**: Brier improvement 5-15%
**Output**: `ml/models/platt_calibrator.pkl`

---

### Step 3: Genera Predictions Calibrate
Crea il file di predictions usando il modello + calibrator:

```bash
python d:\BetAI\v2\scripts\generate-predictions.py
```

**Output**: `backtest/predictions_calibrated.json`

---

### Step 4: Backtest con Edge Filter
Esegui backtest con il nuovo filtro edge (8% minimum):

```bash
cd d:\BetAI\v2
npm run backtest -- --minEdge=0.08 --kellyFraction=0.25
```

**Parametri**:
- `--minEdge=0.08`: Solo bets con edge >= 8% (da default 5%)
- `--kellyFraction=0.25`: Kelly 1/4 frazionato (conservative)

---

### Step 5: Diagnostica Completa
Analizza il backtest con tutti i filtri:

```bash
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py
```

**Testa**:
- Brier su bets ad alto edge (8%, 10%, 12%)
- Segmentazione per lega
- Calibrazione (Platt curve, ECE)
- Rapporto vs Market Brier

---

## 📊 Interpretazione Risultati

### Scenario 1: ✅ PASS (Model salvabile)

```
Closing Line R² (no market quotes): 0.12-0.15
  ✅ Il modello ha insight indipendente

Brier su edge >= 8%: 0.195
Market Brier: 0.210
  ✅ Model Brier < Market Brier per +0.015 improvement

ECE (calibration error): 0.022
  ✅ Ben calibrato dopo Platt scaling

Action: ✅ PROCEED TO LIVE DEPLOYMENT
```

**Step successivi**:
- Deploy su API live
- Monitora CLV giornaliero per 2 settimane
- Riaddestra settimanalmente con nuovi dati
- Aumenta slowly la stake size se performance rimane positiva

---

### Scenario 2: ⚠️ WEAK (Model potenzialmente salvabile)

```
Closing Line R² (no market quotes): 0.08-0.10
  ⚠️  Insight debole, dipende ancora da market data

Brier su edge >= 8%: 0.205
Market Brier: 0.210
  ⚠️  Marginal +0.005 improvement (under noise)

CLV medio: +0.2%
  ⚠️  Sopra zero ma inaffidabile

Action: ⚠️ CONDITIONAL - Riduci puntate del 50%
```

**Step successivi**:
- Aumenta edge filter a 10%
- Riduci Kelly fraction al 15%
- Monitora per 1 mese
- Se rimane profittevole → upgrade a real xG data
- Se diventa negativo → stop e riprendi con dati migliori

---

### Scenario 3: ❌ FAIL (Model non salvabile, riprendi dati)

```
Closing Line R² (no market quotes): < 0.05
  ❌ Il modello è puro rumore senza market data

Brier su edge >= 8%: 0.215
Market Brier: 0.210
  ❌ Model PEGGIO del mercato (-0.005)

% value bets @ 8%+ edge: 4%
  ❌ Non ci sono sufficienti bet profittevoli

Action: ❌ STOP - Upgrade data necessario
```

**Step successivi**:
1. Acquista real xG data da StatsBomb/Wyscout API
2. Integra injuries real-time da Sportsmonks
3. Scrapa confirmed lineups da Flashscore
4. Riaddestra modello completo
5. Ritest diagnostica

**Timeline**: 2-3 settimane per integration

---

## 📋 Checklist di Successo

```
✅ READY FOR LIVE:
  [ ] Brier su edge >= 8% < Market Brier
  [ ] CLV medio >= +0.5% per >50 bet/month
  [ ] Hit rate >= 50% in almeno 2 leghe
  [ ] Closing Line R² >= 0.10 (no market data)
  [ ] ECE <= 0.03 (ben calibrato)

⚠️  CONDITIONAL (ridotto 50%):
  [ ] Brier improvement marginale (+0.003 to +0.010)
  [ ] CLV medio >= +0.2%
  [ ] Funziona bene 1 lega minore
  [ ] Yield positivo ma sotto +1%

❌ NOT READY (ripeti con dati migliori):
  [ ] Brier peggio del mercato
  [ ] Closing Line R² < 0.05
  [ ] CLV medio <= 0
  [ ] >80% partite forzate a perdere (no edge)
```

---

## 🎯 Metriche Critiche Monitorate

| Metrica | Goal | Red Flag |
|---------|------|----------|
| **Brier Edge>=8%** | < Market Brier | > Market Brier |
| **Closing Line R²** | >= 0.10 | < 0.05 |
| **ECE (Calibration)** | < 0.03 | > 0.05 |
| **CLV medio** | >= +0.5% | < 0% |
| **% Value Bets** | 8-15% | < 5% o > 30% |
| **Hit Rate** | >= 50% | < 45% |

---

## 🔧 Opzioni se Test FALLISCE

### Opzione A: Real xG Data (Consigliato)
- **Fonte**: StatsBomb API ($100-2000/anno)
- **Miglioramento**: Brier -0.020 to -0.040
- **Difficulty**: ⭐⭐ (moderato)
- **Timeline**: 1-2 settimane

### Opzione B: Injuries Real-time
- **Fonte**: Sportsmonks API
- **Miglioramento**: Brier -0.008 to -0.012
- **Difficulty**: ⭐ (easy)
- **Timeline**: 3-5 giorni

### Opzione C: Confirmed Lineups
- **Fonte**: Flashscore/FotMob scraping
- **Miglioramento**: Brier -0.005 to -0.008
- **Difficulty**: ⭐⭐⭐ (scraping fragile)
- **Timeline**: 1 week

### Opzione D: Feature Engineering Avanzata
- Squad rotation patterns
- Travel distance + fatigue
- Head-to-head correlations
- Manager tactical profiles
- **Miglioramento**: Brier -0.005 to -0.015
- **Difficulty**: ⭐⭐⭐⭐
- **Timeline**: 2-3 settimane

---

## ⚡ Quick Start

Esegui TUTTO in una volta:

```bash
# Step 1: Chiudi il modello
python d:\BetAI\v2\ml\train_residual_no_market.py

# Step 2: Calibrazione
python d:\BetAI\v2\ml\apply_calibration.py

# Step 3: Predictions
python d:\BetAI\v2\scripts\generate-predictions.py

# Step 4: Backtest
cd d:\BetAI\v2
npm run backtest -- --minEdge=0.08

# Step 5: Diagnostica
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py
```

**Tempo totale**: 15-20 minuti

---

## 📝 Note Importanti

**Non deployare live finché**:
- ❌ Brier overall > Market Brier
- ❌ Closing Line R² < 0.05
- ❌ > 50% delle partite flaggate "value"
- ❌ CLV medio < 0%

**Sempre usa**:
- Kelly fraction 25% inizialmente (non 100%)
- Max stake 5% per partita
- Edge filter >= 8% (non 5%)
- Reserve 20% del bankroll per volatilità

**Monitora settimanalmente**:
- CLV medio (deve essere >= +0.5%)
- Hit rate per lega
- Brier score vs market
- Numero e distribuzione di bet

---

## 📚 File Creati

| File | Scopo |
|------|-------|
| `ml/train_residual_no_market.py` | Closing line test (insight vero?) |
| `ml/apply_calibration.py` | Platt scaling calibration |
| `scripts/generate-predictions.py` | Predictions calibrate per backtest |
| `scripts/comprehensive-diagnostics.py` | Analisi completa del backtest |
| `DIAGNOSTICS_PROTOCOL.md` | Protocollo dettagliato |

---

**Prossimo**: Esegui i 5 step sopra e condividi i risultati della diagnostica. Ti dirò se sei READY o serve upgrade dati.
