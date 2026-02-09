# AUDIT REPORT - BetAI Over 2.5 Model

## Executive Summary

**STATUS: PROBLEMI CRITICI IDENTIFICATI**

Il modello ha problemi di calibrazione significativi sul test set. I risultati del backtest sono SOVRASTIMATI.

---

## 1. DATA LEAKAGE CHECK

| Feature | Status | Note |
|---------|--------|------|
| xG Home/Away | OK | Pre-match predictions (corr con risultato: 0.15) |
| O25/U25 | OK | Pre-match edge values (corr con risultato: 0.03) |
| Rolling features | OK | Shift applicato correttamente |
| H2H features | OK | Solo dati storici |
| ELO | OK | Pre-match ratings |

**CONCLUSIONE: Nessun data leakage rilevato nelle features.**

---

## 2. PROBLEMI CRITICI

### 2.1 Formula Poisson SBAGLIATA

La formula Poisson usata per stimare le quote non corrisponde alla realta:

| Gross xG | Poisson P(O25) | Reale P(O25) | Errore |
|----------|----------------|--------------|--------|
| 2.0 | 32% | 42% | -10pp |
| 2.5 | 46% | 48% | -2pp |
| 3.0 | 58% | 52% | +6pp |
| 3.5 | 68% | 57% | +11pp |
| 4.0 | 76% | 63% | +13pp |

**IMPATTO**: Le quote stimate nel backtest sono sbagliate, quindi i risultati ROI non sono affidabili.

### 2.2 Modello OVERCONFIDENT

Calibrazione sul TEST SET (ultimi 6 mesi):

| P_model | Reale | Gap |
|---------|-------|-----|
| 50-60% | 49.8% | -5pp |
| 60-70% | 54.2% | -11pp |
| 70-80% | 60.3% | -15pp |

**IMPATTO**: Quando il modello predice 70%, la realta e' 60%. Il modello e' troppo sicuro di se.

### 2.3 Backtest su Training Data

Le analisi ROI iniziali includevano il training set. I risultati corretti (solo test set):

| Fascia Quote | WR | ROI Reale | N |
|--------------|-----|-----------|---|
| 1.5-2.0 | 57.9% | +1.6% | 2217 |
| 2.0-2.5 | 57.5% | +20.4% | 259 |

**NOTA**: ROI +20% su quote 2.0-2.5 e' ancora interessante, ma il sample e' piccolo (n=259).

---

## 3. COSA FUNZIONA

1. **No data leakage** - Le features sono tutte pre-match
2. **Split temporale** - Correttamente implementato
3. **Ensemble stacking** - Architettura solida (XGB + LGB + LR)
4. **Edge su quote alte** - Sembra esserci valore su quote 2.0+

---

## 4. COSA NON FUNZIONA

1. **Formula Poisson** - Non riflette la realta del calcio
2. **Calibrazione** - Modello overconfident del 10-15%
3. **Quote stimate** - Non sono quote reali di mercato
4. **ROI gonfiato** - I backtest precedenti erano su tutto il dataset

---

## 5. RACCOMANDAZIONI

### Immediate (Prima di usare in produzione)

1. **Ri-calibrare il modello** con Isotonic/Platt scaling
2. **Usare quote REALI** dal file Excel invece di stimarle
3. **Alzare soglie** a P > 65% per compensare overconfidence
4. **Paper trading** minimo 2 mesi prima di soldi reali

### Lungo termine

1. Raccogliere quote reali di apertura/chiusura
2. Testare su campionato specifico (es. solo Serie A)
3. Aggiungere features di form recente piu granulari
4. Considerare modello separato per casa/trasferta

---

## 6. RISULTATI BACKTEST CORRETTI

Dopo aver applicato:
- Quote empiriche (non Poisson)
- Solo test set (ultimi 6 mesi)
- Modello calibrato

| Parametri | Scommesse | WR | ROI | Drawdown |
|-----------|-----------|-----|-----|----------|
| P>55%, Edge>3% | 1,847 | 57.2% | +1.1% | 44% |
| P>50%, Edge>5% | 1,920 | 56.2% | +1.5% | 41% |

### Per Fascia di Quota

| Fascia | ROI | Raccomandazione |
|--------|-----|-----------------|
| 1.0-1.7 | -5% | NO BET |
| **1.7-2.0** | **+4%** | ZONA OTTIMALE |
| 2.0-2.5 | 0% | Marginale |

**ROI realistico: +1-2% (prima delle commissioni)**
**ROI dopo commissioni book (3-5%): circa 0% o negativo**

---

## 7. CONCLUSIONE

Il modello ha potenziale ma richiede:
- Ri-calibrazione delle probabilita
- Quote reali invece di stimate
- Test con paper trading esteso

**NON USARE IN PRODUZIONE SENZA QUESTE CORREZIONI.**
