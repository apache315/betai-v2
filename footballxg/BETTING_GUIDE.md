# Guida Operativa per il Better

## TL;DR - Cosa Funziona

| Mercato | Usabile? | ROI Atteso | Zona Quote |
|---------|----------|------------|------------|
| **Over 2.5** | Si | +1-4% | 1.70-2.00 |
| BTTS | No | ~0% | - |
| 1X2 | No | - | - |

---

## Modello Over 2.5 - UNICO RACCOMANDATO

### Regole Operative

```
SCOMMETTI SE:
1. Probabilita modello: 50-60%
2. Edge: > 3%
3. Quota: 1.70 - 2.00
4. Stake: 1-2% fisso del bankroll

NON SCOMMETTERE SE:
- Probabilita > 65% (modello overconfident)
- Quota < 1.70 (ROI negativo)
- Quota > 2.50 (sample troppo piccolo)
```

### Performance Backtest (6 mesi, 7726 partite)

| Parametri | Bet | Win Rate | ROI |
|-----------|-----|----------|-----|
| P>55%, Edge>3% | 1,847 | 57.2% | +1.1% |
| P>50%, Edge>5% | 1,920 | 56.2% | +1.5% |

### ROI per Fascia di Quota

| Quota | ROI | Azione |
|-------|-----|--------|
| 1.00-1.70 | **-5%** | NO BET |
| **1.70-2.00** | **+4%** | BET |
| 2.00-2.50 | 0% | Marginale |
| 2.50+ | N/A | Sample piccolo |

---

## Formule Chiave

### Edge (Expected Value)
```
Edge = (Probabilita * Quota) - 1

Esempio:
- P = 55%, Quota = 1.90
- Edge = (0.55 * 1.90) - 1 = 0.045 = +4.5%
- Se Edge > 0, hai valore
```

### Break-Even Win Rate
```
BE = 1 / Quota

Esempio:
- Quota 1.90 → BE = 52.6%
- Devi vincere piu del 52.6% per profitto
```

### Kelly Criterion (Stake Sizing)
```
Kelly% = (P * b - q) / b
dove: b = quota-1, q = 1-P

Consiglio: usa 15-25% del Kelly (quarter Kelly)
```

### ROI
```
ROI = (Win_Rate * Quota_Media - 1) * 100

Esempio:
- WR = 57%, Quota = 1.85
- ROI = (0.57 * 1.85 - 1) * 100 = +5.5%
```

---

## Soglie Critiche

| Parametro | Valore | Motivo |
|-----------|--------|--------|
| P minima | 50% | Sotto non c'e' segnale |
| P massima | 65% | Sopra modello overconfident |
| Edge minimo | 3% | Sotto non copre commissioni |
| Quota minima | 1.70 | Sotto ROI negativo |
| Quota massima | 2.00 | Sopra sample piccolo |
| Stake | 1-2% | Flat, mai Kelly puro |

---

## Problemi Noti

### 1. Modello Overconfident
- Quando predice 70% → realta e' 60%
- Quando predice 60% → realta e' 55%
- **SOLUZIONE**: Non fidarti di P > 65%

### 2. Quote Stimate (non reali)
- Usiamo quote empiriche da dati storici
- Non sono quote reali dei bookmaker
- **SOLUZIONE**: Paper trading prima di soldi veri

### 3. Commissioni Book
- I book prendono 3-5%
- ROI +2% diventa ~0% dopo commissioni
- **SOLUZIONE**: Cercare book con margini bassi

---

## Workflow Consigliato

### 1. Setup (1 volta)
```bash
cd BetAI/footballxg
python markets/over25/train.py  # Allena modello
```

### 2. Daily (ogni giorno)
```python
# Carica modello e dati nuovi
# Filtra: P=50-60%, Edge>3%, Quota=1.70-2.00
# Scommetti 1-2% su ogni segnale
```

### 3. Tracking
- Registra ogni bet (data, match, quota, risultato)
- Calcola ROI settimanale
- Se ROI < -5% per 2 settimane → STOP

---

## Metriche da Monitorare

| Metrica | Target | Alert |
|---------|--------|-------|
| Win Rate | > 54% | < 50% |
| ROI | > 0% | < -3% |
| Drawdown | < 20% | > 30% |
| Sharpe | > 0.5 | < 0 |

---

## FAQ

**Q: Posso usare Kelly puro?**
A: NO. Kelly puro causa drawdown 40-80%. Usa sempre 15-25% del Kelly.

**Q: Perche' non quote alte (>2.5)?**
A: Sample troppo piccolo nel backtest. Non statisticamente significativo.

**Q: Quanto bankroll iniziale?**
A: Minimo 500 EUR con stake 1%. Meglio 1000 EUR.

**Q: Quante bet al giorno?**
A: Circa 10-15 con i filtri consigliati. Non forzare se non ci sono segnali.

**Q: BTTS funziona?**
A: NO. AUC 0.55, ROI ~0%. Non vale il rischio.

---

## Contatti e Supporto

Per problemi tecnici o domande, controlla:
- `AUDIT_REPORT.md` - Analisi dettagliata problemi
- `markets/over25/backtest.py` - Codice backtest
- `markets/common/utils.py` - Formule implementate
