'use client';

import { useState, useEffect } from 'react';

interface DashboardStats {
  totalMatches: number;
  totalFeatures: number;
  leagues: string[];
  lastUpdate: string;
  backtestROI: number;
  backtestBrier: number;
  valueBetsToday: number;
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In production, this would fetch from API
    // For now, show demo data
    setStats({
      totalMatches: 3728,
      totalFeatures: 128,
      leagues: ['EPL', 'Serie A', 'La Liga', 'Bundesliga', 'Ligue 1'],
      lastUpdate: new Date().toISOString().split('T')[0],
      backtestROI: -41.0,
      backtestBrier: 0.57,
      valueBetsToday: 3,
    });
    setLoading(false);
  }, []);

  if (loading) {
    return <div className="text-center py-20">Caricamento...</div>;
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-gray-400 mt-2">
          Sistema ML per predizioni calcistiche con XGBoost + Glicko-2 + Fatigue + Style
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          label="Partite Analizzate"
          value={stats?.totalMatches.toLocaleString() || '0'}
          icon="📊"
        />
        <StatCard
          label="Features per Match"
          value={stats?.totalFeatures.toString() || '0'}
          icon="🧠"
        />
        <StatCard
          label="Brier Score"
          value={stats?.backtestBrier.toFixed(2) || '0'}
          icon="🎯"
          subtext="(< 0.20 = ottimo)"
        />
        <StatCard
          label="Value Bets Oggi"
          value={stats?.valueBetsToday.toString() || '0'}
          icon="💰"
          highlight
        />
      </div>

      {/* Feature Overview */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Feature Engineering</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <FeatureGroup
            title="Glicko-2 Rating"
            features={['Rating squadra', 'Deviazione (incertezza)', 'Volatilità', 'Probabilità Glicko']}
            color="blue"
          />
          <FeatureGroup
            title="Fatigue & Travel"
            features={['Giorni riposo', 'Match ultimi 7/14gg', 'Thu-Sun squeeze', 'km percorsi']}
            color="orange"
          />
          <FeatureGroup
            title="Style Clustering"
            features={['4 cluster K-Means', 'Possession/Counter/Direct/Defensive', 'Matchup history']}
            color="purple"
          />
        </div>
      </div>

      {/* Leagues */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Leghe Supportate</h2>
        <div className="flex flex-wrap gap-3">
          {stats?.leagues.map((league) => (
            <span
              key={league}
              className="px-4 py-2 bg-gray-700 rounded-full text-sm"
            >
              {league}
            </span>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Azioni Rapide</h2>
        <div className="flex flex-wrap gap-4">
          <a href="/predictions" className="btn-primary">
            📈 Vedi Predizioni Oggi
          </a>
          <a href="/backtest" className="btn-primary">
            📊 Risultati Backtest
          </a>
          <a href="/ratings" className="btn-primary">
            ⭐ Rating Squadre
          </a>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  icon,
  subtext,
  highlight,
}: {
  label: string;
  value: string;
  icon: string;
  subtext?: string;
  highlight?: boolean;
}) {
  return (
    <div className={`card ${highlight ? 'border-green-500 bg-green-900/20' : ''}`}>
      <div className="flex items-center gap-3">
        <span className="text-2xl">{icon}</span>
        <div>
          <div className="stat-value">{value}</div>
          <div className="stat-label">{label}</div>
          {subtext && <div className="text-xs text-gray-500">{subtext}</div>}
        </div>
      </div>
    </div>
  );
}

function FeatureGroup({
  title,
  features,
  color,
}: {
  title: string;
  features: string[];
  color: 'blue' | 'orange' | 'purple';
}) {
  const colorClasses = {
    blue: 'border-blue-500 bg-blue-900/20',
    orange: 'border-orange-500 bg-orange-900/20',
    purple: 'border-purple-500 bg-purple-900/20',
  };

  return (
    <div className={`p-4 rounded-lg border ${colorClasses[color]}`}>
      <h3 className="font-semibold mb-2">{title}</h3>
      <ul className="text-sm text-gray-300 space-y-1">
        {features.map((f, i) => (
          <li key={i}>• {f}</li>
        ))}
      </ul>
    </div>
  );
}
