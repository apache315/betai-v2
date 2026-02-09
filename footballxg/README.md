# FootballXG - Modelli Betting su dati xG

Modelli di betting basati su dati [footballxg.com](https://footballxg.com).

## Struttura

```
footballxg/
├── BETTING_GUIDE.md          # GUIDA OPERATIVA PER BETTER
├── AUDIT_REPORT.md           # Report tecnico problemi/soluzioni
├── README.md
│
├── data/
│   ├── source/               # File Excel originale
│   │   └── Footballxg.com - *.xlsx
│   └── processed/            # Dataset processati
│       └── xg_full_dataset.parquet
│
├── markets/
│   ├── common/               # Utility condivise
│   │   └── utils.py
│   ├── over25/               # Over 2.5 (RACCOMANDATO)
│   │   ├── train.py
│   │   ├── backtest.py
│   │   └── models/
│   ├── btts/                 # BTTS (NON RACCOMANDATO)
│   │   └── train.py
│   └── match_odds/           # 1X2 (NON SVILUPPATO)
│
└── docs_xg.pdf               # Documentazione footballxg.com
```

## Quick Start

```bash
# 1. Training modello Over 2.5
cd footballxg
python markets/over25/train.py

# 2. Backtest
python markets/over25/backtest.py
```

## Risultati

| Mercato | AUC | ROI | Raccomandato |
|---------|-----|-----|--------------|
| **Over 2.5** | 0.58 | +1-4% | **SI** |
| BTTS | 0.55 | ~0% | NO |
| 1X2 | - | - | NO |

## File Importanti

| File | Descrizione |
|------|-------------|
| **BETTING_GUIDE.md** | Leggi prima di scommettere! |
| **AUDIT_REPORT.md** | Problemi tecnici identificati |
| **markets/over25/** | Unico modello funzionante |

## Dati

- **Fonte**: footballxg.com
- **Periodo**: 2021-2024
- **Partite**: ~66,000
- **Leghe**: 54

## Regole Operative

1. **Quote**: solo 1.70-2.00
2. **Probabilita**: 50-60% (non >65%)
3. **Stake**: 1-2% fisso
4. **Edge minimo**: 3%

## Note

- Modello overconfident alle alte probabilita
- ROI marginale dopo commissioni bookmaker
- Consigliato paper trading prima di soldi reali
