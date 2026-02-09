# BetAI v2 - Roadmap: Salto di Qualita

## Principi guida (da esperto di settore)
1. Separare stima di probabilita da decisione di bet
2. Pipeline a 2 livelli (Probability Engine -> Betting Engine)
3. Metriche di betting, non da data scientist
4. Segmentazione dei mercati
5. Il modello deve imparare dove il mercato e biased

---

## 1. Segmentazione mercati + backtest segmentato
**Priorita: ALTA | File coinvolti: backtest/engine.ts, betting/value-detector.ts**

- Segmentare per range quota: [1.0-1.5], [1.5-2.0], [2.0-2.5], [2.5-3.5], [3.5+]
- Segmentare per lega e tier
- Segmentare per tipo bet (H, D, A)
- Report di backtest con breakdown per segmento
- Soglie di edge dinamiche per segmento (non 5% fisso ovunque)

## 2. Metriche di betting come obiettivo di ottimizzazione
**Priorita: ALTA | File coinvolti: ml/train_residual.py**

- Cambiare obiettivo Optuna da Brier Score a CLV / yield flat stake
- Brier come constraint (non peggiorare), non come target
- Tracciare per ogni fold: yield, profit factor, drawdown duration, CLV per segmento
- Expected Calibration Error solo sui bin dove effettivamente si scommette

## 3. Analisi storica bias di mercato
**Priorita: ALTA | File coinvolti: nuovo script di analisi**

- Per ogni segmento (lega x range_quota x tipo_bet):
  - Calcolare P_implied media vs frequenza reale
  - Identificare bias sistematici (favourite-longshot, home, draw)
- Usare finestre temporali (pre-2015, 2015-2020, 2020-2025) per verificare stabilita del bias
- Output: mappa di bias per segmento con confidence interval

## 4. Kelly adattivo
**Priorita: MEDIA | File coinvolti: betting/kelly.ts**

- Kelly fraction variabile in base a:
  - Confidenza del modello (Glicko-2 phi, ensemble disagreement)
  - Segmento di mercato (piu aggressivo dove il modello ha edge storico)
  - Dimensione del campione nel segmento
- Limiti di esposizione per lega/giornata

## 5. Integrazione dati movimenti quote
**Priorita: MEDIA | File coinvolti: data/scrapers/, nuovo modulo**

- Valutare OddAlerts API (https://documenter.getpostman.com/view/17615275/2s935uG1WF)
- Tracciare: opening odds, steam moves, closing odds
- Feature aggiuntive: direzione movimento, volatilita quote, spread bookmaker
- CLV reale (quota giocata vs closing line)

## 6. Pipeline 2 livelli formale
**Priorita: MEDIA | File coinvolti: architettura generale**

### Livello 1 - Probability Engine
- Input: features del match
- Output: P(H), P(D), P(A) + confidence interval
- Modelli: XGBoost residuale, GNN, Glicko-2
- Completamente indipendente dalle quote di mercato

### Livello 2 - Betting Engine
- Input: probabilita L1 + quote mercato + metadati segmento
- Analisi: segmento di mercato, bias atteso, qualita segnale
- Output: bet si/no, stake (Kelly adattivo), timing
- Gestione: bankroll, esposizione, correlazione tra bet

---

## Validazione dati (analisi preliminare necessaria)

### Dati disponibili
- 230k+ match archivio (2000-2025, 38 leghe)
- ~18k match 5 leghe primarie (2015-2025)
- 193k match con closing odds

### Limiti da verificare
- Quanti match per segmento fine (lega x range x tipo)?
- Stabilita temporale dei bias (pre-2015 vs post-2015)
- Qualita odds nell'archivio vs football-data.co.uk



Soluzione: 
Historical Football Betting Odds Dataset: Comprehensive Source Analysis and Implementation Strategy
Report Date: January 30, 2026

Executive Summary
A comprehensive European football betting odds dataset spanning 2010–2025 with closing odds from multiple bookmakers is achievable at zero cost using freely accessible sources. Your goal of constructing 130,000+ historical matches across 25+ leagues with multi-bookmaker closing odds meets strong data availability conditions. The optimal approach combines football-data.co.uk's established archive (your current primary source) with supplemental scraping from OddsPortal for gap-filling, yielding sufficient volume for statistically robust 2-dimensional segmentation analysis across 15+ segments.

Current Market for Historical Betting Odds Data
The market for football betting odds datasets has fragmented across three distinct tiers: free/accessible archives, premium APIs with recent data, and specialized scraping tools. Unlike 2015–2020 when Kaggle's European Soccer Database dominated, the landscape has shifted toward distributed sources with varying temporal coverage.

Football-data.co.uk remains the institutional gold standard for European historical odds, covering 22 divisions across 11 European nations with data extending back to 1993/94. The site maintains daily updates and provides closing odds from multiple bookmakers, including historical averages and best prices. Since 2019/20, the site has systematically collected two odds snapshots: post-opening odds and closing odds ('C' suffix in column headers). This dual approach allows analysis of odds movement, critical for market-making research. The resource is free, downloadable as CSV or Excel formats, and explicitly designed for quantitative betting system analysis—making it fundamentally aligned with your use case.

However, a critical issue emerged on July 23, 2025: Pinnacle's public odds API, historically the most reliable pre-closing odds source, became unreliable with systematic delivery delays. Football-data.co.uk has consequently deprecated Pinnacle from market average calculations. This does not eliminate pre-2020 data availability but signals quality degradation for the period 2012–2020 when Pinnacle was the primary closing odds reference.

Primary Data Sources for Football Betting Odds: Features and Accessibility 
Data Source Evaluation: Coverage, Accessibility, and Pragmatism
Football-data.co.uk: The Foundation

Football-data.co.uk directly addresses your stated expansion opportunity. You currently download from five primary European leagues; the site hosts 22 divisions in structured CSV format. The 16 additional world-premier divisions (added post-2020) extend to 38+ total leagues beyond your current scope. For 2010–2026, this archive provides:

European leagues: Full results and odds for English Premier League, Championship, League One, League Two, Scottish Premier and League One, German Bundesliga and 2.Bundesliga, Spanish La Liga and Segunda División, Italian Serie A and Serie B, French Ligue 1 and Ligue 2, Dutch Eredivisie and Eerste Divisie, Belgian Jupiler Pro League, Portuguese Primeira Liga, Greek Super League, Turkish Super Lig, and additional domestic divisions.

Closing odds: Available since 2019/20 with 5–6 bookmaker snapshots per match. Pre-2019/20, "closing odds" are technically pre-closing or market averages depending on the season.

File structure: Individual season ZIP archives or complete season files. CSV format facilitates direct pipeline integration.

No API rate limits, authentication, or maintenance overhead: The resource requires simple HTTP downloads and batch processing.

The simplest expansion strategy is downloading all 22 European divisions from 2010–2026. Based on your 380 matches/year × 5 league × 16-year calculation, expanding to 22 divisions yields:

Top tier (5 leagues): ~30,400 matches (2010–2025)

Mid-tier (12 leagues): ~57,000 matches

Lower-tier (5 additional divisions): ~45,000 matches

Total: ~132,400 matches

Estimated Match Volume by League Tier (2010-2025) vs. Segmentation Threshold 
This far exceeds your 7,500-match minimum per 15-segment configuration. Statistical significance is achieved at the 95% confidence level with approximately 384 samples per segment; 6,500–8,800 samples per segment (the average across your dataset) provides robust power for hypothesis testing and eliminates sampling noise.

OddsPortal.com: High-Granularity Supplement

OddsPortal maintains the most extensive real-time and historical odds database globally, with 20+ bookmakers per match and all leagues covered. Unlike football-data.co.uk, OddsPortal is not directly downloadable; data must be scraped. However, recent tooling maturation has substantially reduced barrier-to-entry.

OddsHarvester, an open-source Python application released in January 2025 and actively maintained, automates OddsPortal scraping with:

Pagination and dynamic content handling (automatic scrolling/page load)

Proxy rotation for detection avoidance

CLI and programmatic interfaces

Export to CSV with team ID mapping and timezone handling

Community discussion confirming multi-year scraping feasibility

A practical scenario: deploy OddsHarvester to extract 2010–2019 closing odds for leagues underrepresented in football-data.co.uk or to validate Pinnacle-source data. Estimated runtime: 2–4 weeks for a comprehensive multi-league historical pull with proxy rotation and throttling.

OddsPortal's advantage lies in bookmaker diversity—access to 20+ sportsbooks per match enables book-specific analysis and line-movement research. For instance, identifying systematic closure odds from Pinnacle, Bet365, and Betfair across your dataset would support hedging-cost analysis or market-efficiency testing.

Kaggle Datasets: Partial Reference

Kaggle's "European Soccer Database" (25,000+ matches, 10 bookmakers, 2008–2016) remains accessible but outdated for contemporary analysis. Its primary value is as a validation dataset or for feature engineering (player names, team IDs, match lineups), not as a primary odds source for 2010–2026.

Newer Kaggle uploads (Top 5 Euro Leagues 2023–24) contain current odds but lack temporal depth. In aggregate, Kaggle is a secondary resource for specific supplemental needs, not a primary data backbone.

SportMonks API and The Odds API: Premium/Recent Data

SportMonks (€39–€69/month) and The Odds API (paid plans) both offer API-driven access to recent historical odds. SportMonks supports 180+ bookmakers with pre-match and 7-day post-match windows; The Odds API provides snapshots since June 2020 at 5–10 minute intervals. Both are suitable for automated real-time analysis or forward-looking backtesting but unsuitable for 2010–2019 historical research at acceptable cost.

If your analysis includes 2024–2026 proprietary bookmaker tracking or real-time odds-feed integration for a commercial product, SportMonks becomes relevant. For historical dataset construction, the cost exceeds the benefit of free alternatives.

Implementation Roadmap: Phased Data Acquisition
Phase 1: Foundation (Weeks 1–4)

Download complete football-data.co.uk archive for all 22 European divisions, seasons 2010/11–2025/26

Extract into local database or CSV staging area

Inspect data structure:

Verify closing odds availability by league and season

Identify columns: match date, teams, result, Pinnacle H/D/A odds, market averages, best/worst prices

Quantify null/missing values

Initial data quality report:

Matches per league per season

Bookmaker coverage distribution (e.g., % of matches with 5+ vs. 3–4 bookmakers)

Odds range validation (e.g., typical H/D/A margins)

Phase 2: Augmentation and Validation (Weeks 5–8)

Identify gaps in closing odds coverage:

Leagues with sparse pre-2019/20 closing odds

Seasons with incomplete Pinnacle data

Deploy OddsHarvester for targeted scraping:

Priority: 2010–2019 period for top 5 leagues

Secondary: mid-tier leagues for any missing seasons

Normalize scraped data to football-data.co.uk schema:

Standardize bookmaker names (account for brand mergers/changes)

Align date/time formats and timezones

Deduplicate on (date, home_team, away_team, bookmaker)

Validation pipeline:

Odds plausibility check (exclude p < 0.01 for any outcome)

Cross-source reconciliation: compare football-data Pinnacle odds vs. OddsPortal Pinnacle

Statistical outlier detection (e.g., odds >10 or <1.01)

Phase 3: Integration and Analysis Preparation (Weeks 9–12)

Merge football-data.co.uk + OddsHarvester data into unified master dataset

Create segmentation schema:

Define odds-range bins (e.g., [1.0–1.5], [1.5–2.0], [2.0–3.0], [3.0–5.0], [5.0+] for home win)

Define bet-type segments (1X2 only vs. including O/U, BTTS, AH)

Generate statistical summaries:

Matches per segment per league-tier

Mean/median closing odds per segment

Bookmaker consensus vs. outliers

Prepare export: single CSV with enriched columns (segment ID, bookmaker consensus, best/worst prices, implied probability)

Data Volume Achievable and Statistical Power
Your requirement specifies 15 segments (5 odds-range buckets × 3 bet types, or alternative 2D taxonomy) with minimum 500 matches per segment for statistical significance. The proposed dataset structure substantially exceeds this threshold:

Scenario: 22-league, 16-year span

Total matches: 132,400

Segments: 15 (conservative)

Matches per segment (uniform distribution): 8,827

Required for 95% CI, Cohen's d = 0.2: ~384 samples per segment

This achieves >4× minimum power. In practice, distribution will be non-uniform (top-tier leagues over-represented), further concentrating statistical power in high-volume segments relevant to professional betting models.

For hypothesis testing (e.g., "closing odds from Bet365 exhibit 3% systematic bias vs. Pinnacle"), the sample size supports detection of effects as small as 0.5–1% with high power. For segmentation-based strategy evaluation, you have sufficient granularity to detect interaction effects between league tier, odds range, and season.

Critical Data Quality Issues and Mitigation
Pinnacle API Deprecation (July 2025)

Football-data.co.uk no longer considers Pinnacle odds reliable for market-average calculations as of late July 2025. If your analysis depends on Pinnacle as a "ground truth" closing odds reference (as is standard in the literature—Pinnacle closing odds are widely treated as the market's true probability estimate), pre-2020 Pinnacle data from football-data.co.uk should be validated against an independent source.

Mitigation: Cross-reference 2015–2020 Pinnacle closing odds from OddsPortal scraping if available, or substitute market-average odds (best price across 5–6 bookmakers) as a proxy. Academic literature supports market-consensus odds as a reasonable efficient-market estimate when Pinnacle is unavailable.

Scraping Reliability and Terms of Service

OddsPortal does not explicitly prohibit historical scraping, but anti-bot detection is active. OddsHarvester includes proxy rotation; however, risks remain:

Rate-limiting: Scraping all historical data for 22 leagues may trigger detection

Data completeness: Odds may be sparse for lower-league, lower-importance matches

Maintenance: OddsPortal structure updates may break scraper selectors (mitigated by OddsHarvester's community maintenance)

Mitigation: Implement incremental scraping with exponential backoff and randomized request timing. Test on a single league/season before full-scale deployment. Monitor for 403/429 responses and pause-resume logic.

Bookmaker Consolidation and Brand Evolution

Bookmakers merge, rebrand, or cease operations. Bet365 acquired certain entities; Interwetten changed ownership. Odds labeled under different names may represent the same underlying book.

Mitigation: Maintain a mapping table of bookmaker brand aliases (e.g., "Interwetten" = "IW" = legacy label X). Standardize on a canonical name for analysis. Football-data.co.uk uses consistent abbreviations (B365, BW, IW, PS, PH, etc.); adhere to these.

Recommended Dataset Export Format and Downstream Use
Master CSV Schema

text
match_id, date, league, home_team, away_team, home_goals, away_goals, 
closing_H, closing_D, closing_A, 
pinnacle_H, pinnacle_D, pinnacle_A,
bet365_H, bet365_D, bet365_A, [... additional bookmakers],
best_H, best_D, best_A,
avg_H, avg_D, avg_A,
num_bookmakers, 
segment_id, 
data_source (football_data | odds_harvester)
This structure enables:

Segmentation analysis: Group by segment_id; calculate win rate, ROI, closing-line value

Odds consensus: Compare individual-bookmaker odds to bet365_* and avg_* benchmarks

Source comparison: Filter by data_source; validate overlaps

Lineage tracking: Audit trail for data governance and research reproducibility

Cost and Timeline Summary
Phase	Duration	Cost	Outputs
Phase 1	4 weeks	€0	50–70k matches from football-data.co.uk; data schema and QA report
Phase 2	4 weeks	€0 (self-hosted scraping)	+30–50k augmented OddsPortal matches; validation report; bookmaker reconciliation table
Phase 3	4 weeks	€0	Master CSV (130k+ matches); segmentation summary; statistical power analysis
Total	12 weeks	€0	Production-ready dataset ready for model development
Optional enhancements:

SportMonks subscription (€50/month × 3): €150 for advanced real-time odds and post-2024 data refinement

BrightData or Bright Proxy service (€30–100/month): Commercial-grade scraping reliability if self-hosted OddsHarvester proves unstable

Conclusion
Your intuition to expand beyond five primary leagues is data-justified. Football-data.co.uk's 22-division coverage, combined with supplemental OddsPortal scraping, delivers 130,000+ historical matches with closing odds from 5–20+ bookmakers per match—well exceeding statistical requirements for 2-dimensional segmentation analysis. The approach is cost-effective (zero up-front), leverages public data, and avoids API dependencies or licensing complications.

The critical path is Phase 1 (data download and inspection), which can commence immediately. Subsequent phases are contingent on gap analysis results; you may find that football-data.co.uk alone suffices if pre-2020 data quality is acceptable. Plan 12 weeks from initiation to a production-ready master dataset, with Phase 1 serving as a decision point for whether OddsHarvester augmentation is necessary.

Given your background in data infrastructure and API integration, the technical execution is straightforward. The largest unknown is Pinnacle data reliability post-July 2025; validate 2015–2020 samples early in Phase 1 to inform downstream analysis methodology.



Quality Check - Risultati
Numeri chiave
107,661 match caricati da 21 leghe
107,506 (99.9%) con odds di almeno un bookmaker
91,588 (85.1%) con closing odds
Bet365 copre il 99.8% dei match, Pinnacle l'85.5%
Closing odds per stagione
2010-11 / 2011-12: 0% closing (solo opening odds)
2012-13 in poi: 94-100% closing odds
2019-20+: 100% closing odds
Quindi hai 13 stagioni con closing odds (2012-2025), non solo dal 2019. Meglio del previsto.

Segmentazione: cosa funziona
12 dei 15 segmenti (odds_bucket × bet_type) superano la soglia di 500 match. I 3 segmenti "LOW" sono i Draw con quota bassa (1.00-2.50), che ha senso: quote draw sotto 2.50 quasi non esistono nel calcio.

Bias iniziale rilevato
I dati mostrano gia' un pattern interessante:

Bet Type	Implied Prob	Actual Win%	Bias
Home	45.36%	43.98%	-1.38%
Draw	27.39%	26.40%	-0.99%
Away	30.99%	29.62%	-1.36%
Il bias negativo su tutti e tre indica l'overround dei bookmaker (margine ~3.7%). Ma la distribuzione non e' uniforme - l'home bias e' il piu' forte, il che suggerisce che i bookmaker sovrastimano sistematicamente le vittorie casalinghe.

Favourite-Longshot Bias: CONFERMATO
Il pattern classico e' chiaro nei dati:

Fascia Quota	Fair Prob	Actual	Bias	Direzione
1.00-1.50 (favoriti)	73.3%	76.0%	+2.74%	Mercato sottostima i favoriti
1.50-2.00	55.1%	55.8%	+0.75%	Lieve sottostima
2.00-3.50	31-43%	~uguale	~0%	Zona efficiente
3.50-5.00	24.5%	24.0%	-0.54%	Mercato sovrastima i longshot
5.00+	13.9%	13.1%	-0.84%	Sovrastima piu' forte
I favoriti forti (quota < 1.50) vincono piu' di quanto il mercato prevede. I longshot vincono meno. Questo e' il favourite-longshot bias documentato in letteratura.

4 segmenti con bias sfruttabile (significativo + yield positivo)
Segmento	N	Bias	Yield	p-value
Tier 2, quote 1.00-1.50, Home	4,476	+4.48%	+1.45%	0.0000
2015-2019, quote 1.00-1.50, Home	3,538	+2.91%	+0.61%	0.0001
2020-2025, quote 1.00-1.50, Away	1,184	+3.15%	+0.38%	0.0166
2020-2025, quote 1.50-2.00, Away	3,332	+2.30%	+0.17%	0.0078
Il segnale piu' forte: favoriti casalinghi nel Tier 2 (Championship, Serie B, Ligue 2, ecc.). Il mercato li sottostima di 4.5%, con yield positivo a flat stake. Questo ha senso: meno liquidita' nei campionati minori = piu' inefficienza.

Stabilita' temporale
Il bias sui favoriti e' stabile in tutti e 3 i periodi (2010-14, 2015-19, 2020-25), il che suggerisce un pattern strutturale, non un artefatto.