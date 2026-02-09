# BetAI v2

Sistema ML per predizioni calcistiche con **230k+ partite**, **modello residuale XGBoost** e **Bayesian Ensemble**.

## Caratteristiche

- **230,000+ partite** da 38 leghe (2000-2025) tramite football-data.co.uk + archive storico
- **49 features selezionate** (da 137 candidate) tramite Boruta + Ablation
- **Modello residuale**: predice correzioni alle probabilita' del mercato, non probabilita' assolute
- **Bayesian Ensemble**: combina XGBoost, GNN e modello residuale
- **Glicko-2 + Elo** ratings per forza squadre
- **Walk-forward validation** con 5 fold cronologici (no look-ahead bias)
- **Kelly Criterion** (1/4 frazionale) per bet sizing
- **CLV tracking** come metrica principale

## Risultati

### Modello Residuale (Walk-Forward, 5 Fold)

| Fold | Model Brier | Market Brier | Improvement | CLV |
|------|------------|-------------|-------------|-----|
| 1 | 0.6057 | 0.6049 | -0.0008 | +0.58% |
| 2 | 0.6046 | 0.6055 | +0.0009 | +0.74% |
| 3 | 0.5958 | 0.5967 | +0.0009 | +0.62% |
| 4 | 0.6008 | 0.6020 | +0.0012 | +0.62% |
| 5 | 0.5966 | 0.5978 | +0.0012 | +0.52% |
| **Media** | **0.6007** | **0.6014** | **+0.0007** | **+0.62%** |

Il modello batte il mercato in 4/5 fold con CLV positivo in tutti e 5.

## Leghe

### Primarie (5 leghe, dati completi con quote Pinnacle)

| Lega | Codice | Stagioni |
|------|--------|----------|
| Premier League | EPL | 2015-2025 |
| Serie A | SERIE_A | 2015-2025 |
| La Liga | LA_LIGA | 2015-2025 |
| Bundesliga | BUNDESLIGA | 2015-2025 |
| Ligue 1 | LIGUE_1 | 2015-2025 |

### Archive (38 leghe, dati storici con Elo pre-calcolati)

230,000+ partite dal 2000 al 2025 usate per training.

## Quick Start

```bash
# 1. Installa dipendenze
npm install
pip install -r requirements.txt

# 2. Scarica dati storici
npm run download

# 3. Costruisci features (con archive: 230k partite)
npm run features:full

# Oppure solo 5 leghe primarie (~18k partite)
npm run features:v2only

# 4. Seleziona features (Boruta + Ablation)
npm run select-features

# 5. Addestra modello residuale
python ml/train_residual.py \
  --data data/processed/features.json \
  --selected-features data/processed/selected_features.json \
  --output ml/models/residual

# 6. Backtest
npm run backtest

# 7. Predizioni di oggi
npm run predict
```

## Struttura

```
v2/
├── data/
│   ├── scrapers/
│   │   ├── football-data.ts    # CSV download (5 leghe)
│   │   ├── archive-loader.ts   # Parser archive 230k partite
│   │   ├── understat.ts        # xG data
│   │   └── odds-api.ts         # Quote live
│   ├── merge-matches.ts        # Deduplicazione v2 + archive
│   ├── raw/                    # CSV scaricati (gitignore)
│   └── processed/              # Features JSON (gitignore)
├── ml/
│   ├── features.ts             # 137 features (49 selezionate)
│   ├── train_residual.py       # Training residuale (principale)
│   ├── train.py                # Training XGBoost diretto
│   ├── bayesian_ensemble.py    # Bayesian Ensemble
│   ├── gnn_model.py            # Graph Neural Network
│   ├── feature_selection.py    # Boruta + Ablation
│   ├── glicko2.ts              # Rating Glicko-2
│   ├── fatigue.ts              # Indice fatica + viaggi
│   ├── style-clustering.ts     # K-Means stili gioco
│   └── model.ts                # Wrapper XGBoost
├── betting/
│   ├── kelly.ts                # Kelly Criterion
│   └── value-detector.ts       # Value bet detection + CLV
├── backtest/
│   └── engine.ts               # Walk-forward backtest
└── scripts/
    ├── build-features.ts       # Pipeline features
    ├── select-features.ts      # Feature selection
    ├── train-model.ts          # Script training
    ├── train-gnn.ts            # Training GNN
    ├── download-data.ts        # Download dati
    ├── backtest.ts             # Backtest
    └── predict-today.ts        # Predizioni giornaliere
```

## Architettura ML

### Approccio Residuale

Invece di predire H/D/A da zero (dove il modello non batte mai il mercato), il modello predice **correzioni** alle probabilita' implicite delle quote:

```
prob_finale = prob_mercato + correzione_modello
```

- 3 regressori XGBoost separati (Home/Draw/Away)
- Target: `one_hot(risultato) - prob_mercato`
- Output normalizzato a somma 1, clipped [0.01, 0.99]
- Hyperparameter tuning con Optuna (30 trial per fold)

### Feature Engineering (49 features selezionate)

| Categoria | Features | Descrizione |
|-----------|----------|-------------|
| Form | 10 | Win/draw/loss rate, PPG (5 e 10 partite) |
| Glicko-2 | 11 | Rating, RD, volatilita', probabilita' |
| Elo | 6 | Elo archive, differenza, prob. vittoria |
| Fatigue | 16 | Riposo, congestione, viaggi |
| Style | 16 | Cluster K-Means, matchup storico |
| League | 3 | Tier, paese, lega primaria |
| H2H | 6 | Head-to-head storico |
| Seasonal | 5 | Posizione, punti, GD |

### Feature Selection

1. **Boruta**: identifica features significativamente migliori di feature shadow randomizzate
2. **Ablation**: rimuove features una alla volta, tiene solo quelle che migliorano il Brier score
3. Risultato: 137 -> 49 features

## Metriche

| Metrica | Obiettivo | Descrizione |
|---------|-----------|-------------|
| **Brier Score** | < market | Calibrazione probabilita' (multiclass) |
| **CLV** | > 0% | Closing Line Value |
| **Accuracy 1X2** | > 45% | Precisione classificazione |
| **Value Hit Rate** | > 45% | % value bet vincenti |

## Configurazione

### Quote Live (opzionale)
```bash
export ODDS_API_KEY=your_key_here
npm run predict
```

### Telegram Bot (opzionale)
Secrets GitHub:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Riferimenti

- [Glicko-2 Paper](http://www.glicko.net/glicko/glicko2.pdf) - Mark Glickman
- [football-data.co.uk](https://www.football-data.co.uk/) - Dati storici
- [Fortune's Formula](https://en.wikipedia.org/wiki/Fortune%27s_Formula) - Kelly Criterion
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient Boosting

## License

MIT
