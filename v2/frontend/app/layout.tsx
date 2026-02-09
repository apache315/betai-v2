import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'BetAI v2 - ML Football Predictions',
  description: 'Sistema ML per predizioni calcistiche con XGBoost',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="it">
      <body>
        <div className="min-h-screen bg-gray-900 text-white">
          <nav className="bg-gray-800 border-b border-gray-700">
            <div className="max-w-7xl mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <a href="/" className="text-xl font-bold text-green-400">
                  🎯 BetAI v2
                </a>
                <div className="flex gap-6">
                  <a href="/" className="hover:text-green-400 transition">Dashboard</a>
                  <a href="/predictions" className="hover:text-green-400 transition">Predizioni</a>
                  <a href="/backtest" className="hover:text-green-400 transition">Backtest</a>
                  <a href="/ratings" className="hover:text-green-400 transition">Rating</a>
                </div>
              </div>
            </div>
          </nav>
          <main className="max-w-7xl mx-auto px-4 py-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
