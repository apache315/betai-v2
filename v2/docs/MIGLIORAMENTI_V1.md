# MIGLIORAMENTI CALCIO-PRED - Analisi e Roadmap

> Documento generato dall'audit completo del codebase.
> Ogni sezione descrive: **Problema** | **Soluzione** | **File coinvolti** | **Motivazione**

---

## Indice

- [P0 - Fix Critici Matematici](#p0---fix-critici-matematici)
  - [1. Formula Empirica 1X2 errata](#1-formula-empirica-1x2-errata)
  - [2. Double Counting sui Lambda](#2-double-counting-sui-lambda)
  - [3. Confidence calcolata due volte](#3-confidence-calcolata-due-volte)
- [P1 - Miglioramenti Modello](#p1---miglioramenti-modello)
  - [4. Injuries Impact disabilitato](#4-injuries-impact-disabilitato)
  - [5. Under/Over Empirico usa distribuzione sbagliata](#5-underover-empirico-usa-distribuzione-sbagliata)
  - [6. ML Service non e' vero ML](#6-ml-service-non-e-vero-ml)
  - [7. Parametri hardcoded senza cross-validation](#7-parametri-hardcoded-senza-cross-validation)
- [P2 - Code Quality](#p2---code-quality)
  - [8. Type safety: any ovunque nei metodi critici](#8-type-safety-any-ovunque-nei-metodi-critici)
  - [9. Under/Over e BTTS: README incoerente col codice](#9-underover-e-btts-readme-incoerente-col-codice)
  - [10. 80+ file non organizzati nella root](#10-80-file-non-organizzati-nella-root)
- [P3 - Infrastruttura Test](#p3---infrastruttura-test)
  - [11. Zero test automatizzati](#11-zero-test-automatizzati)

---

## P0 - Fix Critici Matematici

### 1. Formula Empirica 1X2 errata

**Problema**

In `empiric.ts:170-197`, il calcolo delle probabilita' 1X2 usa:

```typescript
prob1 = homeStrength * (1 - awayStrength)
prob2 = awayStrength * (1 - homeStrength)
probX = avgDrawRate
```

Questa formula non ha fondamento probabilistico. Se entrambe le squadre vincono il 70%
delle partite (home in casa, away in trasferta):
- `prob1 = 0.70 * (1 - 0.70) = 0.21` → sottostima pesante della vittoria casalinga
- `prob2 = 0.70 * (1 - 0.70) = 0.21` → stessa sottostima
- Il pareggio diventa dominante per normalizzazione

Il risultato e' che le previsioni 1X2 dell'empirico sono sistematicamente schiacciate
verso il pareggio, distorcendo il blending col Poisson.

**Soluzione: Bradley-Terry Model**

Sostituire con un modello Bradley-Terry, standard nella letteratura di ranking sportivo:

```typescript
// Forza relativa: rapporto tra win rate di ciascuna squadra
const homeRating = homeStrength / (homeStrength + awayStrength);
const awayRating = awayStrength / (homeStrength + awayStrength);

// Probabilita' finali con draw rate empirica
const combatProb = 1 - avgDrawRate;
prob1 = homeRating * combatProb;
prob2 = awayRating * combatProb;
probX = avgDrawRate;
```

Con squadre al 70%: `homeRating = 0.70/1.40 = 0.50`, prob1 = 0.50 * 0.73 = 0.365.
Molto piu' realistico.

**File coinvolti**: `api/src/services/prediction/empiric.ts` (metodo `calculate1X2`)

**Motivazione**: Il Bradley-Terry e' il modello di riferimento per il confronto tra
competitori con forza relativa. E' usato da FIFA, ELO, e tutti i sistemi di ranking.
Produce probabilita' proporzionali alla forza relativa delle squadre, non al prodotto
delle debolezze avversarie.

---

### 2. Double Counting sui Lambda

**Problema**

In `engine.ts:177-290`, il lambda Poisson (gol attesi) viene moltiplicato in cascata
per 4 fattori:

```
lambdaBase (dallo storico, che gia' include forma/H2H/lega)
  *= formFactor        (0.65-1.40)
  *= h2hFactor         (0.85-1.15)
  *= leagueCoefficient (0.70-1.05)
```

Il problema e' che `lambdaBase` viene calcolato dallo storico delle partite, che
**gia' riflette implicitamente** la forma recente (le ultime partite pesano di piu'),
il rendimento nei derby (inclusi nell'archivio), e il livello della lega (i gol sono
gia' calibrati sul campionato giocato).

Risultato: gli stessi segnali vengono contati 2-3 volte. Una squadra in ottima forma
nel suo campionato riceve un boost composto:
- Lambda storico gia' alto (ultime partite pesate di piu')
- formFactor = 1.40 (forma eccellente)
- h2hFactor = 1.15 (dominante negli scontri diretti)
- leagueCoeff = 1.00

Lambda finale: `1.5 * 1.40 * 1.15 * 1.00 = 2.415` (gol attesi irrealistici)

**Soluzione: Adjustment additivo con dampening**

Cambiare da moltiplicativo a additivo con dampening logaritmico:

```typescript
// Calcola adjustment complessivo come somma limitata
const formAdj = (homeForm.formFactor - 1.0);     // es: +0.40
const h2hAdj = (h2hStats.h2hFactor.home - 1.0);  // es: +0.15
const leagueAdj = (leagueStrength.coefficient - 1.0); // es: 0.00

// Dampening: limita l'adjustment totale a ±30% del lambda base
const totalAdj = formAdj + h2hAdj + leagueAdj;
const dampenedAdj = Math.sign(totalAdj) * Math.min(Math.abs(totalAdj), 0.30);

poissonResult.lambdaHome *= (1 + dampenedAdj);
```

Con lo stesso esempio: `1.5 * (1 + 0.30) = 1.95` — realistico.

**File coinvolti**: `api/src/services/prediction/engine.ts` (metodo `calculatePrediction`,
sezione "Apply Form/H2H/League")

**Motivazione**: I modelli predittivi sportivi ben calibrati (Dixon-Coles originale,
Football-Data.co.uk) applicano adjustment limitati perche' il lambda base gia'
cattura la maggior parte della varianza. Il compounding moltiplicativo senza
dampening porta a lambda estremi che peggiorano la calibrazione del modello.

---

### 3. Confidence calcolata due volte

**Problema**

In `engine.ts:411-469`, la confidence viene calcolata due volte consecutivamente
con gli stessi input:

```typescript
// Prima chiamata (linea 411-426)
let confidenceOverall = confidenceCalculator.calculate(
  homeHistory, awayHistory, injuries, lineups, ...
).overall;

// ... applicazione boost ...

// Seconda chiamata (linea 452-468) - IDENTICA alla prima
const confidence = {
  ...confidenceCalculator.calculate(
    homeHistory, awayHistory, injuries, lineups, ...  // stessi input!
  ),
  overall: confidenceOverall,  // sovrascrive con il valore modificato
};
```

La seconda chiamata ricalcola tutti i sotto-fattori (dataAvailability, recency,
stability, lineupStatus, injuryImpact) solo per poi sovrascrivere `overall` con
il valore gia' calcolato e modificato. Spreco computazionale puro.

**Soluzione**

Chiamare una sola volta e applicare i boost sul risultato:

```typescript
const confidenceResult = confidenceCalculator.calculate(
  homeHistory, awayHistory, injuries, lineups, ...
);
let confidenceOverall = confidenceResult.overall;

// Applica boost (market calibration, league strength)
if (calibrationResult?.confidenceBoost > 0) {
  confidenceOverall = Math.min(1.0, confidenceOverall + calibrationResult.confidenceBoost);
}
confidenceOverall *= leagueStrength.confidenceFactor;

const confidence = { ...confidenceResult, overall: confidenceOverall };
```

**File coinvolti**: `api/src/services/prediction/engine.ts`

**Motivazione**: Eliminare codice duplicato che spreca risorse e rende il flusso
difficile da seguire. Il risultato funzionale e' identico.

---

## P1 - Miglioramenti Modello

### 4. Injuries Impact disabilitato

**Problema**

In `engine.ts:221-230`, l'intera sezione di impatto infortuni e' commentata:

```typescript
let injuriesAnalysis: any = null;
/*
const injuriesAnalysis = await injuriesService.analyzeMatchInjuriesImpact(
  input.homeTeamId, input.awayTeamId, input.fixtureId, input.season
);
*/
```

Gli infortuni vengono fetchati (linea 95) e usati per la confidence (linea 414),
ma **non influenzano le probabilita' 1X2 ne' il lambda Poisson**. Per un sistema
di predizione pre-match, ignorare chi gioca e' un errore grave.

Un attaccante top (Haaland, Mbappe) assente puo' ridurre il lambda atteso della
squadra del 15-25%. Questo non viene catturato.

**Soluzione**

Implementare `analyzeMatchInjuriesImpact` nel servizio infortuni Sportsmonks:

```typescript
// Logica proposta:
// 1. Fetch giocatori infortunati/squalificati per entrambe le squadre
// 2. Classificare per ruolo: GK, DEF, MID, ATT
// 3. Applicare severity weight:
//    - Titolare ATT assente: attackFactor = 0.85 (-15%)
//    - Titolare MID assente: attackFactor = 0.93, defenseFactor = 0.95
//    - Titolare DEF assente: defenseFactor = 0.88 (-12%)
//    - Titolare GK assente: defenseFactor = 0.82 (-18%)
// 4. Cumulare se piu' giocatori assenti (con dampening)
// 5. Restituire { home: { attackFactor, defenseFactor }, away: {...} }
```

Il factor viene poi applicato al lambda in engine.ts (gia' predisposto con il codice
commentato alle linee 234-260).

**File coinvolti**:
- `api/src/services/sportsmonks/injuries.ts` (aggiungere `analyzeMatchInjuriesImpact`)
- `api/src/services/prediction/engine.ts` (decommentare e connettere)

**Motivazione**: L'assenza di giocatori chiave e' il singolo fattore piu' predittivo
per le deviazioni dal lambda atteso. Tutti i modelli professionali (Opta, StatsBomb)
lo integrano come primo adjustment.

---

### 5. Under/Over Empirico usa distribuzione sbagliata

**Problema**

In `empiric.ts:220-231`, il calcolo Under/Over usa una approssimazione con CDF normale:

```typescript
const over = this.normalCDF(threshold, expectedGoals, Math.sqrt(avgVariance));
```

Ma il motore Poisson (`poisson.ts:461-487`) calcola Under/Over dalla matrice Poisson
esatta. Quando il blender combina i due risultati, sta mescolando probabilita'
calcolate con distribuzioni diverse (Normale vs Poisson), creando inconsistenza.

Per bassi expected goals (<2.0), la Normale e la Poisson divergono significativamente
(la Poisson e' asimmetrica, la Normale no).

**Soluzione**

Usare la CDF di Poisson anche nell'empirico:

```typescript
private calculateUnderOverPoisson(expectedGoals: number): { [key: string]: { under: number; over: number } } {
  const thresholds = [0.5, 1.5, 2.5, 3.5, 4.5];
  const underOver: { [key: string]: { under: number; over: number } } = {};

  thresholds.forEach(threshold => {
    // Somma P(X <= floor(threshold)) usando Poisson CDF
    let cdfValue = 0;
    const maxK = Math.floor(threshold);
    for (let k = 0; k <= maxK; k++) {
      cdfValue += (Math.pow(expectedGoals, k) * Math.exp(-expectedGoals)) / this.factorial(k);
    }
    underOver[threshold.toString()] = { under: cdfValue, over: 1 - cdfValue };
  });

  return underOver;
}
```

**File coinvolti**: `api/src/services/prediction/empiric.ts`

**Motivazione**: Coerenza statistica. Se entrambi i motori usano la stessa distribuzione
di base (Poisson), il blending produce risultati matematicamente coerenti. La Poisson
e' il modello naturale per eventi contabili rari (gol nel calcio).

---

### 6. ML Service non e' vero ML

**Problema**

Il file `ml-algorithm.service.ts` e `ml-prediction.service.ts` sono etichettati come
"Machine Learning" ma non contengono nessun algoritmo di apprendimento automatico:

- Nessun training su dati storici
- Nessun gradient descent, nessuna loss function
- Nessuna feature selection automatica
- Nessuna cross-validation
- Pesi fissi hardcoded (H2H 10%, Form 35%, Stats 30%, xG 25%)

E' un secondo motore statistico a pesi fissi, parallelo al motore Empirico+Poisson.

**Soluzione (Futura - v2.0)**

Per il next step reale verso ML:

1. **Feature Engineering**: Estrarre 30-50 feature da ogni partita (lambda, form, H2H,
   xG, injuries count, league position, days of rest, home/away streak, etc.)
2. **Gradient Boosting**: Usare XGBoost o LightGBM per predire P(1), P(X), P(2)
3. **Training pipeline**: Split temporale (train su stagioni passate, test su corrente)
4. **Calibrazione Platt**: Calibrare le probabilita' in output con isotonic regression
5. **Ensemble**: Combinare il Poisson/Empirico corrente con il modello ML come
   stacking ensemble

Questo richiede un progetto separato. Per ora, rinominare il servizio da "ML" a
"StatisticalPredictor" per evitare confusione.

**File coinvolti**:
- `api/src/services/ml-prediction/ml-algorithm.service.ts` (rinominare)
- `api/src/services/ml-prediction.service.ts` (rinominare)
- `api/src/services/prediction/engine.ts` (aggiornare import/commenti)

**Motivazione**: Trasparenza. Chiamare "ML" un sistema a pesi fissi genera false
aspettative e rende difficile capire quando si introduce vero ML.

---

### 7. Parametri hardcoded senza cross-validation

**Problema**

Il sistema ha decine di soglie e parametri hardcoded:

| Parametro | Valore | File |
|-----------|--------|------|
| Dixon-Coles RHO dinamico | 0.05-0.18 | `poisson.ts:55-85` |
| Form factor range | 0.65-1.40 | `form-momentum.ts` |
| H2H factor range | 0.85-1.15 | `engine.ts:908-910` |
| Seasonal draw boost Q1 | +30% | `ml-algorithm.service.ts:86` |
| Empiric/Poisson blend | 55/45 | `config/index.ts` |
| xG blend weight | 25% | `config/index.ts` |
| GIOCALA threshold | 80% + conf 0.60 | `config/index.ts` |
| Time decay factor | 0.95 | `config/index.ts` |

Nessuno di questi e' stato ottimizzato con cross-validation temporale. I risultati
di backtest mostrano alta volatilita' (da -92% ROI in Q1 2025 a +683% in Nov 2025),
suggerendo possibile overfitting ai periodi di test recenti.

**Soluzione (Futura)**

1. Implementare grid search con cross-validation temporale (walk-forward):
   - Train: 6 mesi rolling window
   - Test: 1 mese successivo
   - Metrica: Brier Score (non accuracy, non ROI)
2. Documentare i risultati per ogni combinazione di parametri
3. Usare i parametri che minimizzano il Brier Score su almeno 3 fold temporali

**File coinvolti**: Nuovo script `api/src/scripts/optimize-parameters.ts`

**Motivazione**: Un modello con parametri ottimizzati su un periodo e testato su
un altro e' la base minima per validare che non si sta facendo overfitting.

---

## P2 - Code Quality

### 8. Type safety: `any` ovunque nei metodi critici

**Problema**

I metodi piu' importanti di `engine.ts` usano `any` per i parametri:

- `classifyAllMarkets(final: any, ...)` (linea 952)
- `buildPredictionResponse(blendedResult: any, confidence: any, strength: any, ...)`
  (linea 1031) con 17 parametri posizionali, molti `any`
- `calculateH2HStats` restituisce un oggetto complesso senza interfaccia

Questo rende impossibile per il compilatore TypeScript catturare errori di tipo
e rende il refactoring pericoloso.

**Soluzione**

Definire interfacce precise per ogni struttura dati:

```typescript
interface ClassifyMarketsInput {
  prob1: number; probX: number; prob2: number;
  underOver: Record<string, { under: number; over: number }>;
  btts: { yes: number; no: number };
  doubleChance: Record<string, number>;
  exactGoals?: Record<string, number>;
}

interface MarketStrengths {
  strength1X2: PredictionStrength;
  strengthOver05: PredictionStrength;
  // ... etc
}
```

Per `buildPredictionResponse`, convertire i 17 parametri posizionali in un singolo
oggetto parametro con interfaccia tipizzata.

**File coinvolti**: `api/src/services/prediction/engine.ts`, `api/src/types/index.ts`

**Motivazione**: TypeScript senza tipi e' solo JavaScript piu' lento da scrivere.
La type safety e' il punto centrale dell'adozione di TS.

---

### 9. Under/Over e BTTS: README incoerente col codice

**Problema**

Il README.md dichiara:
```
- Under/Over (not supported)
- BTTS (not supported)
```

Ma il codice calcola, salva nel DB (schema Prisma con tutti i campi Under/Over e BTTS),
e mostra nel frontend entrambi i mercati. Il README e' fuorviante.

**Soluzione**

Aggiornare il README per riflettere lo stato reale: Under/Over e BTTS sono calcolati
e mostrati, anche se meno validati del mercato 1X2.

**File coinvolti**: `README.md`

**Motivazione**: Documentazione accurata. Un utente che legge "not supported" potrebbe
ignorare dati utili o dubitare dell'affidabilita' del sistema.

---

### 10. 80+ file non organizzati nella root

**Problema**

La root del progetto contiene 80+ file markdown di documentazione, script di backtest,
report JSON/TXT, guide utente. Non c'e' struttura:

```
calcio-pred/
├── BACKTESTING_README.md
├── BACKTEST_ANALYSIS_AND_FIXES.md
├── BACKTEST_CACHE_OPTIMIZATION.md
├── BACKTEST_FIX_ACCURATEZZA_REPORT.md
├── ... (80+ file simili)
├── backtest-month.mjs
├── backtest-multiple.js
├── analyze-1x2-rating.mjs
├── ... (20+ script)
```

Impossibile capire cosa e' attuale e cosa obsoleto.

**Soluzione**

Riorganizzare in:
```
calcio-pred/
├── docs/
│   ├── architecture/     # Docs architettura
│   ├── guides/           # Guide utente
│   ├── reports/          # Report backtest
│   └── changelog/        # Fix e miglioramenti
├── scripts/
│   ├── backtest/         # Script backtest
│   ├── analysis/         # Script analisi
│   └── optimization/     # Script ottimizzazione
```

**File coinvolti**: Tutti i file .md e .mjs/.js nella root

**Motivazione**: Manutenibilita'. Un nuovo sviluppatore non puo' orientarsi con 80+
file nella root.

---

## P3 - Infrastruttura Test

### 11. Zero test automatizzati

**Problema**

Il progetto non ha:
- Nessun file `*.test.ts` o `*.spec.ts`
- Nessun test runner configurato (jest, vitest)
- Nessun script `test` nel package.json
- Nessuna pipeline CI/CD

Per un sistema dove l'accuratezza delle predizioni e' il prodotto, l'assenza di test
e' il rischio piu' alto. Un refactoring che cambia un segno in una formula puo'
invertire tutte le predizioni senza che nessuno se ne accorga fino al prossimo
backtest manuale.

**Soluzione**

1. Setup Vitest (veloce, compatibile con TypeScript nativo)
2. Test unitari per:
   - `PoissonEngine.calculate()` con input noti → output atteso
   - `EmpiricEngine.calculate()` con input noti → output atteso
   - `Blender.blend()` con input noti → output atteso
   - `ConfidenceCalculator.calculate()` con input noti → output atteso
   - `StrengthClassifier.classify1X2()` per ogni soglia
   - Dixon-Coles: verifica che la matrice normalizzata sommi a 1.0
   - Under/Over: verifica monotonia (Over 0.5 >= Over 1.5 >= ... >= Over 4.5)
3. Test di integrazione per `PredictionEngine.calculatePrediction()` con fixture mock

**File coinvolti**:
- `api/package.json` (aggiungere vitest)
- `api/vitest.config.ts` (nuovo)
- `api/src/services/prediction/__tests__/` (nuova cartella)

**Motivazione**: I test sono la rete di sicurezza che permette di fare refactoring
senza paura. Senza test, ogni modifica e' un rischio non misurato.

---

## Riepilogo Priorita'

| # | Issue | Priorita' | Impatto | Effort |
|---|-------|-----------|---------|--------|
| 1 | Formula empirica 1X2 | P0 | Alto - distorce tutte le predizioni | Basso |
| 2 | Double counting lambda | P0 | Alto - lambda irrealistici | Medio |
| 3 | Confidence duplicata | P0 | Medio - spreco + confusione | Basso |
| 4 | Injuries disabilitato | P1 | Alto - ignora chi gioca | Medio |
| 5 | Under/Over distribuzione | P1 | Medio - inconsistenza motori | Basso |
| 6 | ML non e' ML | P1 | Basso - naming misleading | Basso |
| 7 | Parametri non validati | P1 | Alto - possibile overfitting | Alto |
| 8 | Type safety any | P2 | Medio - rischio refactoring | Medio |
| 9 | README incoerente | P2 | Basso - documentazione | Basso |
| 10 | File root disorganizzati | P2 | Basso - manutenibilita' | Basso |
| 11 | Zero test | P3 | Critico - nessuna safety net | Alto |
