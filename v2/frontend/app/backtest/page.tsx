'use client';

import { useState, useEffect } from 'react';

interface BacktestResult {
  period: string;
  totalMatches: number;
  totalBets: number;
  accuracy: number;
  brierScore: number;
  roi: number;
  profit: number;
  maxDrawdown: number;
  sharpe: number;
  clvMean: number;
  clvPositiveRate: number;
  monthlyResults: Array<{
    month: string;
    bets: number;
    roi: number;
    clv: number;
  }>;
  marketBreakdown: Array<{
    market: string;
    bets: number;
    winRate: number;
    roi: number;
  }>;
}

// Demo backtest results
const demoBacktest: BacktestResult = {
  period: '2023-01 to 2024-12',
  totalMatches: 3728,
  totalBets: 326,
  accuracy: 0.549,
  brierScore: 0.5697,
  roi: -41.0,
  profit: -409.99,
  maxDrawdown: 0.677,
  sharpe: -1.19,
  clvMean: 2.0,
  clvPositiveRate: 1.0,
  monthlyResults: [
    { month: '2023-01', bets: 16, roi: 29.8, clv: 2.0 },
    { month: '2023-02', bets: 15, roi: 21.7, clv: 2.0 },
    { month: '2023-03', bets: 15, roi: -71.8, clv: 2.0 },
    { month: '2023-04', bets: 27, roi: 62.0, clv: 2.0 },
    { month: '2023-05', bets: 25, roi: -34.3, clv: 2.0 },
    { month: '2023-09', bets: 14, roi: 71.8, clv: 2.0 },
    { month: '2023-12', bets: 34, roi: 42.2, clv: 2.0 },
    { month: '2024-04', bets: 18, roi: -72.9, clv: 2.0 },
    { month: '2024-11', bets: 16, roi: -43.4, clv: 2.0 },
    { month: '2024-12', bets: 24, roi: 17.5, clv: 2.0 },
  ],
  marketBreakdown: [
    { market: '1X2_H', bets: 22, winRate: 40.9, roi: 7.6 },
    { market: '1X2_D', bets: 132, winRate: 15.2, roi: -8.2 },
    { market: '1X2_A', bets: 172, winRate: 25.6, roi: -8.9 },
  ],
};

export default function BacktestPage() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setResult(demoBacktest);
      setLoading(false);
    }, 500);
  }, []);

  if (loading || !result) {
    return <div className="text-center py-20">Caricamento risultati backtest...</div>;
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Backtest Results</h1>
        <p className="text-gray-400 mt-1">
          Periodo: {result.period} | Modello: Enhanced Heuristic (Glicko-2 + Fatigue + Style)
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Accuracy 1X2"
          value={`${(result.accuracy * 100).toFixed(1)}%`}
          status={result.accuracy > 0.5 ? 'good' : 'bad'}
        />
        <MetricCard
          label="Brier Score"
          value={result.brierScore.toFixed(3)}
          status={result.brierScore < 0.6 ? 'neutral' : 'bad'}
          help="< 0.20 = ottimo"
        />
        <MetricCard
          label="ROI"
          value={`${result.roi.toFixed(1)}%`}
          status={result.roi > 0 ? 'good' : 'bad'}
        />
        <MetricCard
          label="Max Drawdown"
          value={`${(result.maxDrawdown * 100).toFixed(1)}%`}
          status={result.maxDrawdown < 0.5 ? 'good' : 'bad'}
        />
      </div>

      {/* CLV Section */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">
          CLV (Closing Line Value) - La Vera Metrica
        </h2>
        <p className="text-gray-400 text-sm mb-4">
          Se il CLV è positivo costantemente, stai battendo il mercato. Il profitto arriverà matematicamente.
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-green-900/30 border border-green-500 rounded-lg p-4">
            <div className="text-3xl font-bold text-green-400">
              +{result.clvMean.toFixed(1)}%
            </div>
            <div className="text-gray-400 text-sm">CLV Medio</div>
          </div>
          <div className="bg-green-900/30 border border-green-500 rounded-lg p-4">
            <div className="text-3xl font-bold text-green-400">
              {(result.clvPositiveRate * 100).toFixed(0)}%
            </div>
            <div className="text-gray-400 text-sm">Bets con CLV positivo</div>
          </div>
        </div>
      </div>

      {/* Monthly Results */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Risultati Mensili</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="table-header">
                <th className="p-3 rounded-tl-lg">Mese</th>
                <th className="p-3">Bets</th>
                <th className="p-3">ROI</th>
                <th className="p-3 rounded-tr-lg">CLV</th>
              </tr>
            </thead>
            <tbody>
              {result.monthlyResults.map((m) => (
                <tr key={m.month} className="table-row">
                  <td className="p-3 font-medium">{m.month}</td>
                  <td className="p-3">{m.bets}</td>
                  <td className={`p-3 font-semibold ${m.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {m.roi >= 0 ? '+' : ''}{m.roi.toFixed(1)}%
                  </td>
                  <td className="p-3 text-green-400">+{m.clv.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Market Breakdown */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Per Mercato</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {result.marketBreakdown.map((m) => (
            <div
              key={m.market}
              className={`p-4 rounded-lg border ${
                m.roi >= 0 ? 'border-green-500 bg-green-900/20' : 'border-gray-700'
              }`}
            >
              <div className="text-lg font-semibold">
                {m.market === '1X2_H' ? '🏠 Home Win' :
                 m.market === '1X2_D' ? '🤝 Draw' :
                 '✈️ Away Win'}
              </div>
              <div className="mt-2 space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Bets:</span>
                  <span>{m.bets}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Win Rate:</span>
                  <span>{m.winRate.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">ROI:</span>
                  <span className={m.roi >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {m.roi >= 0 ? '+' : ''}{m.roi.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Note */}
      <div className="bg-yellow-900/30 border border-yellow-500 rounded-lg p-4">
        <p className="text-yellow-200 text-sm">
          ⚠️ Questi risultati usano un modello euristico. Per risultati ottimali,
          addestrare il modello XGBoost completo con Optuna.
        </p>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  status,
  help,
}: {
  label: string;
  value: string;
  status: 'good' | 'bad' | 'neutral';
  help?: string;
}) {
  const statusColors = {
    good: 'text-green-400',
    bad: 'text-red-400',
    neutral: 'text-yellow-400',
  };

  return (
    <div className="card">
      <div className={`text-2xl font-bold ${statusColors[status]}`}>{value}</div>
      <div className="text-gray-400 text-sm">{label}</div>
      {help && <div className="text-xs text-gray-500 mt-1">{help}</div>}
    </div>
  );
}
