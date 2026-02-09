'use client';

import { useState, useEffect } from 'react';

interface TeamRating {
  team: string;
  league: string;
  rating: number;
  deviation: number;
  volatility: number;
  style: string;
  recentForm: string;
  fatigueIndex: number;
}

// Demo ratings data
const demoRatings: TeamRating[] = [
  { team: 'Man City', league: 'EPL', rating: 1720, deviation: 45, volatility: 0.05, style: 'Possession', recentForm: 'WWWDW', fatigueIndex: 0.15 },
  { team: 'Arsenal', league: 'EPL', rating: 1685, deviation: 52, volatility: 0.06, style: 'Possession', recentForm: 'WDWWW', fatigueIndex: 0.20 },
  { team: 'Liverpool', league: 'EPL', rating: 1678, deviation: 48, volatility: 0.05, style: 'Counter', recentForm: 'WWWWL', fatigueIndex: 0.25 },
  { team: 'Chelsea', league: 'EPL', rating: 1595, deviation: 65, volatility: 0.08, style: 'Possession', recentForm: 'WLDWW', fatigueIndex: 0.30 },
  { team: 'Newcastle', league: 'EPL', rating: 1580, deviation: 58, volatility: 0.06, style: 'Direct', recentForm: 'DWWDL', fatigueIndex: 0.35 },
  { team: 'Tottenham', league: 'EPL', rating: 1565, deviation: 70, volatility: 0.09, style: 'Counter', recentForm: 'LDWWL', fatigueIndex: 0.20 },
  { team: 'Man United', league: 'EPL', rating: 1545, deviation: 75, volatility: 0.10, style: 'Counter', recentForm: 'DLLWD', fatigueIndex: 0.15 },
  { team: 'Aston Villa', league: 'EPL', rating: 1540, deviation: 62, volatility: 0.07, style: 'Direct', recentForm: 'WWDLD', fatigueIndex: 0.40 },
  { team: 'Inter', league: 'Serie A', rating: 1695, deviation: 50, volatility: 0.05, style: 'Counter', recentForm: 'WWWWD', fatigueIndex: 0.25 },
  { team: 'Napoli', league: 'Serie A', rating: 1640, deviation: 68, volatility: 0.08, style: 'Possession', recentForm: 'DWWLW', fatigueIndex: 0.20 },
  { team: 'Real Madrid', league: 'La Liga', rating: 1710, deviation: 48, volatility: 0.05, style: 'Possession', recentForm: 'WWWDW', fatigueIndex: 0.30 },
  { team: 'Barcelona', league: 'La Liga', rating: 1680, deviation: 55, volatility: 0.06, style: 'Possession', recentForm: 'WDWWL', fatigueIndex: 0.25 },
];

export default function RatingsPage() {
  const [ratings, setRatings] = useState<TeamRating[]>([]);
  const [sortBy, setSortBy] = useState<'rating' | 'deviation' | 'fatigue'>('rating');
  const [leagueFilter, setLeagueFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setRatings(demoRatings);
      setLoading(false);
    }, 500);
  }, []);

  const sortedRatings = [...ratings]
    .filter((r) => leagueFilter === 'all' || r.league === leagueFilter)
    .sort((a, b) => {
      if (sortBy === 'rating') return b.rating - a.rating;
      if (sortBy === 'deviation') return a.deviation - b.deviation;
      return a.fatigueIndex - b.fatigueIndex;
    });

  const leagues = [...new Set(ratings.map((r) => r.league))];

  if (loading) {
    return <div className="text-center py-20">Caricamento rating...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Rating Squadre (Glicko-2)</h1>
        <p className="text-gray-400 mt-1">
          Rating con incertezza e stile di gioco
        </p>
      </div>

      {/* Legend */}
      <div className="card">
        <h3 className="font-semibold mb-3">Legenda Glicko-2</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-blue-400 font-semibold">Rating:</span>{' '}
            Forza della squadra (1500 = media)
          </div>
          <div>
            <span className="text-orange-400 font-semibold">Deviation (σ):</span>{' '}
            Incertezza (basso = affidabile)
          </div>
          <div>
            <span className="text-purple-400 font-semibold">Volatility:</span>{' '}
            Quanto è imprevedibile
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 items-center">
        <div className="flex gap-2">
          <span className="text-gray-400">Lega:</span>
          <select
            value={leagueFilter}
            onChange={(e) => setLeagueFilter(e.target.value)}
            className="bg-gray-700 rounded px-3 py-1"
          >
            <option value="all">Tutte</option>
            {leagues.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </div>
        <div className="flex gap-2">
          <span className="text-gray-400">Ordina per:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-gray-700 rounded px-3 py-1"
          >
            <option value="rating">Rating</option>
            <option value="deviation">Affidabilità</option>
            <option value="fatigue">Meno affaticati</option>
          </select>
        </div>
      </div>

      {/* Ratings Table */}
      <div className="card overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="table-header">
              <th className="p-3 text-left rounded-tl-lg">#</th>
              <th className="p-3 text-left">Squadra</th>
              <th className="p-3 text-left">Lega</th>
              <th className="p-3 text-right">Rating</th>
              <th className="p-3 text-right">σ (Dev)</th>
              <th className="p-3 text-center">Stile</th>
              <th className="p-3 text-center">Forma</th>
              <th className="p-3 text-right rounded-tr-lg">Fatigue</th>
            </tr>
          </thead>
          <tbody>
            {sortedRatings.map((team, index) => (
              <tr key={team.team} className="table-row">
                <td className="p-3 text-gray-400">{index + 1}</td>
                <td className="p-3 font-semibold">{team.team}</td>
                <td className="p-3 text-gray-400">{team.league}</td>
                <td className="p-3 text-right">
                  <span className={`font-bold ${getRatingColor(team.rating)}`}>
                    {team.rating}
                  </span>
                </td>
                <td className="p-3 text-right">
                  <span className={getDeviationColor(team.deviation)}>
                    ±{team.deviation}
                  </span>
                </td>
                <td className="p-3 text-center">
                  <StyleBadge style={team.style} />
                </td>
                <td className="p-3 text-center">
                  <FormDisplay form={team.recentForm} />
                </td>
                <td className="p-3 text-right">
                  <FatigueBar value={team.fatigueIndex} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function getRatingColor(rating: number): string {
  if (rating >= 1650) return 'text-green-400';
  if (rating >= 1550) return 'text-yellow-400';
  return 'text-gray-400';
}

function getDeviationColor(deviation: number): string {
  if (deviation <= 50) return 'text-green-400';
  if (deviation <= 65) return 'text-yellow-400';
  return 'text-orange-400';
}

function StyleBadge({ style }: { style: string }) {
  const colors: Record<string, string> = {
    'Possession': 'bg-blue-600',
    'Counter': 'bg-orange-600',
    'Direct': 'bg-red-600',
    'Defensive': 'bg-gray-600',
  };

  return (
    <span className={`px-2 py-1 rounded text-xs ${colors[style] || 'bg-gray-600'}`}>
      {style}
    </span>
  );
}

function FormDisplay({ form }: { form: string }) {
  return (
    <div className="flex gap-0.5 justify-center">
      {form.split('').map((result, i) => (
        <span
          key={i}
          className={`w-5 h-5 rounded text-xs flex items-center justify-center ${
            result === 'W' ? 'bg-green-600' :
            result === 'D' ? 'bg-gray-600' :
            'bg-red-600'
          }`}
        >
          {result}
        </span>
      ))}
    </div>
  );
}

function FatigueBar({ value }: { value: number }) {
  const percentage = value * 100;
  const color = value < 0.2 ? 'bg-green-500' :
                value < 0.35 ? 'bg-yellow-500' :
                'bg-red-500';

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-xs text-gray-400">{percentage.toFixed(0)}%</span>
    </div>
  );
}
