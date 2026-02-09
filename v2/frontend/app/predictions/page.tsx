'use client';

import { useState, useEffect } from 'react';

interface Prediction {
  id: string;
  homeTeam: string;
  awayTeam: string;
  league: string;
  date: string;
  kickoff: string;
  probHome: number;
  probDraw: number;
  probAway: number;
  confidence: number;
  valueBet?: {
    market: string;
    selection: string;
    edge: number;
    odds: number;
    stake: number;
  };
  glickoRatingDiff: number;
  fatigueDiff: number;
}

// Demo data - in production this comes from API
const demoPredictions: Prediction[] = [
  {
    id: '1',
    homeTeam: 'Arsenal',
    awayTeam: 'Chelsea',
    league: 'EPL',
    date: '2025-01-28',
    kickoff: '20:45',
    probHome: 0.52,
    probDraw: 0.24,
    probAway: 0.24,
    confidence: 0.72,
    valueBet: {
      market: '1X2',
      selection: 'Home Win',
      edge: 0.08,
      odds: 1.95,
      stake: 2.5,
    },
    glickoRatingDiff: 85,
    fatigueDiff: 0.1,
  },
  {
    id: '2',
    homeTeam: 'Man United',
    awayTeam: 'Liverpool',
    league: 'EPL',
    date: '2025-01-28',
    kickoff: '17:30',
    probHome: 0.28,
    probDraw: 0.26,
    probAway: 0.46,
    confidence: 0.58,
    glickoRatingDiff: -120,
    fatigueDiff: -0.15,
  },
  {
    id: '3',
    homeTeam: 'Juventus',
    awayTeam: 'Inter',
    league: 'Serie A',
    date: '2025-01-28',
    kickoff: '20:45',
    probHome: 0.38,
    probDraw: 0.30,
    probAway: 0.32,
    confidence: 0.45,
    glickoRatingDiff: 25,
    fatigueDiff: 0.05,
  },
];

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate API fetch
    setTimeout(() => {
      setPredictions(demoPredictions);
      setLoading(false);
    }, 500);
  }, []);

  const filteredPredictions = predictions.filter((p) => {
    if (filter === 'all') return true;
    if (filter === 'value') return p.valueBet !== undefined;
    return p.league === filter;
  });

  const valueBetsCount = predictions.filter((p) => p.valueBet).length;

  if (loading) {
    return <div className="text-center py-20">Caricamento predizioni...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Predizioni</h1>
          <p className="text-gray-400 mt-1">
            {new Date().toLocaleDateString('it-IT', {
              weekday: 'long',
              year: 'numeric',
              month: 'long',
              day: 'numeric'
            })}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-green-400 font-semibold">
            💰 {valueBetsCount} Value Bets
          </span>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-2 flex-wrap">
        {['all', 'value', 'EPL', 'Serie A', 'La Liga'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-2 rounded-lg transition ${
              filter === f
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            {f === 'all' ? 'Tutte' : f === 'value' ? '💰 Value Bets' : f}
          </button>
        ))}
      </div>

      {/* Predictions List */}
      <div className="space-y-4">
        {filteredPredictions.map((pred) => (
          <PredictionCard key={pred.id} prediction={pred} />
        ))}
      </div>

      {filteredPredictions.length === 0 && (
        <div className="text-center py-10 text-gray-400">
          Nessuna predizione trovata per questo filtro.
        </div>
      )}
    </div>
  );
}

function PredictionCard({ prediction }: { prediction: Prediction }) {
  const {
    homeTeam, awayTeam, league, kickoff,
    probHome, probDraw, probAway,
    confidence, valueBet, glickoRatingDiff, fatigueDiff
  } = prediction;

  const confidenceClass =
    confidence >= 0.7 ? 'confidence-high' :
    confidence >= 0.5 ? 'confidence-medium' :
    'confidence-low';

  return (
    <div className={`card ${valueBet ? 'value-bet border-green-500' : ''}`}>
      <div className="flex flex-col lg:flex-row lg:items-center gap-4">
        {/* Match Info */}
        <div className="flex-1">
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-2">
            <span>{league}</span>
            <span>•</span>
            <span>{kickoff}</span>
            {valueBet && (
              <span className="bg-green-600 text-white px-2 py-0.5 rounded text-xs">
                VALUE BET
              </span>
            )}
          </div>
          <div className="text-xl font-semibold">
            {homeTeam} vs {awayTeam}
          </div>
        </div>

        {/* Probabilities */}
        <div className="flex gap-4 items-center">
          <ProbabilityBlock label="1" value={probHome} color="blue" />
          <ProbabilityBlock label="X" value={probDraw} color="gray" />
          <ProbabilityBlock label="2" value={probAway} color="red" />
        </div>

        {/* Confidence & Factors */}
        <div className="flex flex-col items-end gap-1">
          <div className={`text-lg font-semibold ${confidenceClass}`}>
            {(confidence * 100).toFixed(0)}% conf.
          </div>
          <div className="text-xs text-gray-400">
            Glicko: {glickoRatingDiff > 0 ? '+' : ''}{glickoRatingDiff}
          </div>
          <div className="text-xs text-gray-400">
            Fatigue: {fatigueDiff > 0 ? '+' : ''}{(fatigueDiff * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Value Bet Details */}
      {valueBet && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="flex flex-wrap gap-4 items-center text-sm">
            <div>
              <span className="text-gray-400">Selezione:</span>{' '}
              <span className="font-semibold">{valueBet.selection}</span>
            </div>
            <div>
              <span className="text-gray-400">Quota:</span>{' '}
              <span className="font-semibold">{valueBet.odds.toFixed(2)}</span>
            </div>
            <div>
              <span className="text-gray-400">Edge:</span>{' '}
              <span className="text-green-400 font-semibold">
                +{(valueBet.edge * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-gray-400">Stake suggerito:</span>{' '}
              <span className="font-semibold">{valueBet.stake.toFixed(1)}% bankroll</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ProbabilityBlock({
  label,
  value,
  color
}: {
  label: string;
  value: number;
  color: 'blue' | 'gray' | 'red';
}) {
  const colorClasses = {
    blue: 'bg-blue-600',
    gray: 'bg-gray-600',
    red: 'bg-red-600',
  };

  return (
    <div className="text-center min-w-[60px]">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-lg font-bold">{(value * 100).toFixed(0)}%</div>
      <div className="w-full h-1 bg-gray-700 rounded-full mt-1">
        <div
          className={`h-full rounded-full ${colorClasses[color]}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}
