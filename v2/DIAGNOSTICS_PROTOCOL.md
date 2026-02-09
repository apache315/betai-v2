# Azioni Diagnostiche: Fase Calibrazione & Validazione

## Situazione Attuale
Il tuo modello mostra segni classici di **overfitting da ML betting**:
- ❌ Brier PEGGIO del mercato (-0.008 tipicamente)
- ⚠️ 60-90% delle partite flaggate come "value bets" (irrealistico)
- ⚠️ CLV debole (+0.3-0.7%) - sotto la soglia +1% di affidabilità
- ❌ Lo yield positivo è probabilmente falso positivo da overfitting

**Diagnosi**: Il modello replica il mercato con distorsione sistematica + amplifica il rumore per generare CLV marginale.

---

## Azioni Implementate

### 1. Script di Diagnostica Completa ✅
**File**: `scripts/comprehensive-diagnostics.py`

**Testa**:
- [x] Brier score overall vs market Brier
- [x] Brier SOLO su value bets (edge >= 3%, 5%, 8%, 10%, 12%)
- [x] **TEST CRITICO**: Se Brier su edge>=8% < Market Brier → EDGE REALE
- [x] Segmentazione per lega (dove funziona il modello?)
- [x] Segmentazione per range di quote (low/mid/long)
- [x] Analisi calibrazione (Platt curve, ECE)

**Esegui**:
```bash
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py
```

---

### 2. Closing Line Test ✅
**File**: `ml/train_residual_no_market.py`

**Cosa fa**: Ritraina il modello SENZA usare le quote di mercato come feature
- Rimuove tutte le colonne: `*odd*`, `*market*`, `*betting*`, `*implied*`
- Se R² crollo drasticamente → il "edge" viene dalle quote (data leak)
- Se R² rimane decente (>0.10) → c'è insight vero nel modello

**Perché**: Testa se il modello generalizza senza quote o dipende solo dai dati del mercato

**Esegui**:
```bash
python d:\BetAI\v2\ml\train_residual_no_market.py
```

**Interpretazione**:
- R² no-market < 0.05 → ❌ **NIENTE EDGE**: Ricomincia con dati migliori (real xG)
- R² no-market >= 0.10 → ✅ **EDGE POTENZIALE**: Procedi con calibrazione

---

### 3. Calibrazione con Platt Scaling ✅
**File**: `ml/apply_calibration.py`

**Cosa fa**: Applica regressione logistica ai residui del modello per recalibrare le probabilità
- Corregge distorsioni sistematiche senza riallenare il modello completo
- Tempo: 5 minuti, rischio minimo
- Effetto: Tipicamente migliora Brier del 5-15%

**Esegui**:
```bash
python d:\BetAI\v2\ml\apply_calibration.py
```

**Output**: `ml/models/platt_calibrator.pkl` (applica al modello prima di usarlo)

---

## Protocollo di Test Completo

### Fase 1: Diagnostica Pre-Calibrazione (2-3 minuti)
```bash
# Controlla diagnostica senza calibrazione
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py
```

**Cosa aspettarsi**:
```
Brier overall: 0.2XXX
Brier market:  0.2XXX
Model better?: NO (overfitting)

Edge >= 8%: 2000 bets (9.3%)
  Brier: 0.2YYY
  vs Market: STILL WORSE ❌
```

### Fase 2: Chiudi il Modello (5 minuti)
```bash
python d:\BetAI\v2\ml\train_residual_no_market.py
```

**Gate Check**: Se R² senza quote < 0.05 → **STOP** - Riprendi con real xG

### Fase 3: Applica Calibrazione (2 minuti)
```bash
python d:\BetAI\v2\ml\apply_calibration.py
```

### Fase 4: Ri-Backtest (5 minuti)
```bash
cd d:\BetAI\v2
npm run backtest -- --minEdge=0.08
```

### Fase 5: Diagnostica Post-Calibrazione
```bash
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py
```

**Decisione finale**:
- ✅ **PASS**: Brier su edge>=8% < Market Brier → **Pronto per live**
- ⚠️ **WEAK**: Brier su edge>=8% ≈ Market Brier ±0.001 → Riduci bet di 50%
- ❌ **FAIL**: Brier su edge>=8% > Market Brier → **Riprendi con real xG + injuries**

---

## Prossimi Step se Test FALLISCE

### Opzione 1: Real xG Data
**Costo**: $100-2000/anno via StatsBomb o Wyscout API
**Miglioramento atteso**: Brier -0.020 to -0.040
**Timeline**: 2 settimane per integrazione

### Opzione 2: Aggiungi Injuries Data
**Fonte**: Sportsmonks API (free trial o premium)
**Miglioramento atteso**: Brier -0.008 to -0.012
**Timeline**: 3-5 giorni

### Opzione 3: Confirmed Lineups
**Fonte**: Flashscore o FotMob scraping
**Miglioramento atteso**: Brier -0.005 to -0.008
**Timeline**: 1 week

### Opzione 4: Feature Engineering Avanzata
- Squad rotation patterns (starting XI correlation)
- Travel distance + fatigue (days rest)
- Head-to-head historical trends (ultimi 10 scontri)
- Manager tactical patterns (pressing, possession style)
- Stadium effect (home advantage reale vs distorsione)

---

## Criteri di Successo

### ✅ READY FOR LIVE se:
1. Brier su edge >= 8% < Market Brier (almeno -0.002)
2. CLV medio >= +0.5% su 50+ bet/mese
3. Hit rate >= 50% su almeno 2 leghe
4. Yield positivo anche se tolgo il 10% di bet random (robustezza)
5. Chiusura linea test: R² senza quote >= 0.10

### ⚠️ CONDITIONAL if:
1. Brier migliora su edge >= 10% (più selettivo)
2. CLV >= +0.3% con yield >= +2%
3. Funziona bene su 1 lega (non multilighe)
4. Ricavi prevedibilmente > spese trading

### ❌ STOP & REWORK se:
1. Brier senza quote crolla < 0.05 (dipendi da market data)
2. Nessun miglioramento dopo calibrazione
3. CLV < +0.2% anche su edge >= 10%
4. >80% dei bet forzati a perdere (no edge, solo rumore)

---

## Metriche da Monitorare

```
Dashboard Monitoraggio:

Metrica              | Target    | Azione se Fallisce
---------------------|-----------|------------------
Brier high-edge      | < 0.200   | Calibrazione
vs Market Brier      |           | + Data mejoramiento
---------------------|-----------|------------------
% value bets (8%+)   | 8-15%     | Se > 20% → filtro too loose
                     |           | Se < 5% → filtro too tight
---------------------|-----------|------------------
CLV medio            | >= +0.5%  | Se < 0.3% → recalibrate
                     |           | Se < 0% → modello broken
---------------------|-----------|------------------
Closing Line R² (no quote) | >= 0.10 | Se < 0.05 → ripeti con real xG
---------------------|-----------|------------------
ECE (calibration err)| < 0.03    | Se > 0.05 → Platt scaling needed
```

---

## Timeline Consigliato

**Domani (Giorno 1)**:
- [ ] Esegui Fase 1-5 completa (20 minuti)
- [ ] Documenta risultati
- [ ] Decidi PASS/FAIL

**Se PASS (Giorno 2)**:
- [ ] Deploy in production con live API
- [ ] Monitora CLV giornaliero
- [ ] Riaddestra settimanalmente con nuovi dati

**Se FAIL (Giorno 2-14)**:
- [ ] Scegli opzione upgrade (real xG consigliato)
- [ ] Integra nuovi dati
- [ ] Riaddestra modello
- [ ] Ritest diagnostica

---

## Note Importanti

**Non deployare live se**:
- [ ] Brier overall > Market Brier ❌
- [ ] > 50% delle partite flaggate "value" ❌
- [ ] CLV medio < 0% ❌
- [ ] R² chiusura linea < 0.05 ❌

**Tieni sempre**:
- Kelly fraction al 25% inizialmente (ridotto da standard 100%)
- Max stake al 5% per partita
- Filtro minEdge >= 8% (non 5%)
- Reserve 20% del bankroll per volatilità

---

## Comandi Rapidi

```bash
# TEST COMPLETO
bash d:\BetAI\v2\scripts\run-all-diagnostics.sh

# Solo diagnostica
python d:\BetAI\v2\scripts\comprehensive-diagnostics.py

# Solo calibrazione
python d:\BetAI\v2\ml\apply_calibration.py

# Solo closing-line test
python d:\BetAI\v2\ml\train_residual_no_market.py

# Backtest con filtri
npm run backtest -- --minEdge=0.08 --kellyFraction=0.25
```

---

**Prossimo passo**: Esegui `run-all-diagnostics.sh` e condividi i risultati.
