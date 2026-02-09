#!/usr/bin/env python3
"""
Graph Neural Network for Football Match Prediction

Architecture:
- Each TEAM is a node with features (Glicko rating, season stats, form)
- Each MATCH is an edge connecting home and away teams
- GNN learns team EMBEDDINGS (32-dim vectors) that capture:
  - Team strength (like Elo but richer)
  - Playing style (attacking vs defensive)
  - Current form (temporal dynamics)
  - Contextual relationships (who beat whom)

The embeddings are updated after each matchday via message passing:
  "A team's representation is shaped by its opponents' representations"

This captures TRANSITIVE STRENGTH:
  If A beats B (strong) and B beats C (strong), A's embedding
  encodes this chain of information automatically.

Usage:
    python gnn_model.py --data features.json --output gnn_embeddings.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_features(path: str) -> pd.DataFrame:
    import os
    file_size = os.path.getsize(path)

    if file_size > 100_000_000:  # > 100MB: stream line by line
        print(f"  Large file ({file_size / 1e6:.0f} MB), streaming...")
        rows = []
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().rstrip(',')
                if not line or line in ('[]', '[', ']'):
                    continue
                try:
                    match = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = {
                    'matchId': match['matchId'],
                    'date': match['date'],
                    'homeTeam': match['homeTeam'],
                    'awayTeam': match['awayTeam'],
                    'league': match['league'],
                    'result': match['result'],
                    **match['features']
                }
                if match.get('closingOdds'):
                    row['closing_odds_home'] = match['closingOdds']['home']
                    row['closing_odds_draw'] = match['closingOdds']['draw']
                    row['closing_odds_away'] = match['closingOdds']['away']
                rows.append(row)
                count += 1
                if count % 50000 == 0:
                    print(f"    Loaded {count} matches...")
        print(f"    Total: {count} matches")
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    with open(path, 'r') as f:
        data = json.load(f)

    rows = []
    for match in data:
        row = {
            'matchId': match['matchId'],
            'date': match['date'],
            'homeTeam': match['homeTeam'],
            'awayTeam': match['awayTeam'],
            'league': match['league'],
            'result': match['result'],
            **match['features']
        }
        if match.get('closingOdds'):
            row['closing_odds_home'] = match['closingOdds']['home']
            row['closing_odds_draw'] = match['closingOdds']['draw']
            row['closing_odds_away'] = match['closingOdds']['away']
        rows.append(row)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def build_team_index(df: pd.DataFrame) -> dict:
    """Assign integer index to each team."""
    teams = sorted(set(df['homeTeam'].unique()) | set(df['awayTeam'].unique()))
    return {team: idx for idx, team in enumerate(teams)}


# ──────────────────────────────────────────────
# NODE FEATURES (per team, computed from history)
# ──────────────────────────────────────────────

def compute_team_node_features(df: pd.DataFrame, team_idx: dict, match_date: pd.Timestamp, lookback: int = 10) -> torch.Tensor:
    """
    Compute node features for each team based on their recent history.

    Features per team (12 dims):
    - win_rate, draw_rate, loss_rate (last N)
    - goals_scored_avg, goals_conceded_avg (last N)
    - points_per_game (last N)
    - home_win_rate (last N home games)
    - away_win_rate (last N away games)
    - glicko_rating (if available)
    - season_position_proxy (PPG this season)
    - xg_avg (if available)
    - rest_days (days since last game)
    """
    n_teams = len(team_idx)
    n_features = 12
    features = torch.zeros(n_teams, n_features)

    # Filter to matches before current date
    past = df[df['date'] < match_date]

    for team, idx in team_idx.items():
        # Get team's recent matches
        team_matches = past[(past['homeTeam'] == team) | (past['awayTeam'] == team)].tail(lookback)

        if len(team_matches) == 0:
            # Default features for new teams
            features[idx] = torch.tensor([0.33, 0.33, 0.33, 1.3, 1.3, 1.33, 0.5, 0.3, 1500, 1.33, 1.3, 7.0])
            continue

        wins, draws, losses = 0, 0, 0
        goals_for, goals_against = 0, 0
        home_wins, home_games = 0, 0
        away_wins, away_games = 0, 0

        for _, m in team_matches.iterrows():
            is_home = m['homeTeam'] == team
            result = m['result']

            if is_home:
                goals_for += m.get('home_goals_scored_avg_5', 1.3) if 'home_goals_scored_avg_5' in m else 1.3
                goals_against += m.get('home_goals_conceded_avg_5', 1.3) if 'home_goals_conceded_avg_5' in m else 1.3
                home_games += 1
                if result == 'H':
                    wins += 1
                    home_wins += 1
                elif result == 'D':
                    draws += 1
                else:
                    losses += 1
            else:
                goals_for += m.get('away_goals_scored_avg_5', 1.0) if 'away_goals_scored_avg_5' in m else 1.0
                goals_against += m.get('away_goals_conceded_avg_5', 1.3) if 'away_goals_conceded_avg_5' in m else 1.3
                away_games += 1
                if result == 'A':
                    wins += 1
                    away_wins += 1
                elif result == 'D':
                    draws += 1
                else:
                    losses += 1

        n = len(team_matches)
        win_rate = wins / n
        draw_rate = draws / n
        loss_rate = losses / n
        gf_avg = goals_for / n
        ga_avg = goals_against / n
        ppg = (wins * 3 + draws) / n
        home_wr = home_wins / max(home_games, 1)
        away_wr = away_wins / max(away_games, 1)

        # Glicko rating from features (if available)
        last_match = team_matches.iloc[-1]
        if last_match['homeTeam'] == team:
            glicko = last_match.get('glicko_home_rating', 1500)
        else:
            glicko = last_match.get('glicko_away_rating', 1500)

        # xG avg
        xg_avg = 1.3  # default
        if 'home_xg_avg_5' in last_match and not pd.isna(last_match.get('home_xg_avg_5')):
            xg_avg = last_match['home_xg_avg_5'] if last_match['homeTeam'] == team else last_match.get('away_xg_avg_5', 1.3)

        # Rest days
        last_date = team_matches.iloc[-1]['date']
        rest_days = min(14, (match_date - last_date).days)

        features[idx] = torch.tensor([
            win_rate, draw_rate, loss_rate,
            gf_avg, ga_avg, ppg,
            home_wr, away_wr,
            glicko / 2000.0,  # Normalize
            ppg,  # season position proxy
            xg_avg,
            rest_days / 14.0,  # Normalize
        ])

    return features


# ──────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ──────────────────────────────────────────────

def build_match_graph(df: pd.DataFrame, team_idx: dict, match_date: pd.Timestamp, lookback_matches: int = 200) -> Data:
    """
    Build a graph of recent matches.

    Nodes: teams
    Edges: matches (bidirectional)
    Edge features: [home_goals, away_goals, result_indicator, recency_weight]
    """
    past = df[df['date'] < match_date].tail(lookback_matches)

    # Build edge lists
    edge_src = []
    edge_dst = []
    edge_attr = []

    for _, m in past.iterrows():
        home_idx = team_idx.get(m['homeTeam'])
        away_idx = team_idx.get(m['awayTeam'])
        if home_idx is None or away_idx is None:
            continue

        # Days ago (for recency weighting)
        days_ago = max(1, (match_date - m['date']).days)
        recency = 1.0 / (1.0 + days_ago / 30.0)  # Decay over months

        result_h = 1.0 if m['result'] == 'H' else (0.5 if m['result'] == 'D' else 0.0)

        # Bidirectional edges
        # Home -> Away edge
        edge_src.append(home_idx)
        edge_dst.append(away_idx)
        edge_attr.append([result_h, 1.0 - result_h, recency])

        # Away -> Home edge (reversed result)
        edge_src.append(away_idx)
        edge_dst.append(home_idx)
        edge_attr.append([1.0 - result_h, result_h, recency])

    if len(edge_src) == 0:
        # Empty graph fallback
        n_teams = len(team_idx)
        return Data(
            x=torch.zeros(n_teams, 12),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 3),
        )

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Node features
    node_features = compute_team_node_features(df, team_idx, match_date)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


# ──────────────────────────────────────────────
# GNN MODEL
# ──────────────────────────────────────────────

class FootballGNN(nn.Module):
    """
    Graph Attention Network for football team embeddings.

    Architecture:
    1. Input: team node features (12 dims)
    2. GAT Layer 1: 12 -> 64 (4 attention heads)
    3. GAT Layer 2: 64 -> 32 (2 attention heads)
    4. Output: team embeddings (32 dims)

    For match prediction:
    5. Concatenate [home_embedding, away_embedding, context_features]
    6. MLP: 64+context -> 32 -> 3 (H/D/A probabilities)
    """

    def __init__(self, node_features=12, hidden_dim=64, embed_dim=32, context_dim=7, num_heads_1=4, num_heads_2=2, dropout=0.2):
        super().__init__()

        # GAT layers for learning team embeddings
        self.gat1 = GATConv(node_features, hidden_dim // num_heads_1, heads=num_heads_1, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, embed_dim // num_heads_2, heads=num_heads_2, concat=True, dropout=dropout)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

        # Match prediction MLP
        # Input: home_embed(32) + away_embed(32) + context(7) = 71
        mlp_input = embed_dim * 2 + context_dim
        self.match_mlp = nn.Sequential(
            nn.Linear(mlp_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),  # H, D, A
        )

        self.dropout = dropout

    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Forward pass through GAT layers to get team embeddings."""
        x, edge_index = data.x, data.edge_index

        if edge_index.size(1) == 0:
            # No edges - return zero embeddings
            return torch.zeros(x.size(0), 32)

        # GAT Layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT Layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        return x  # [n_teams, embed_dim]

    def predict_match(self, home_embed: torch.Tensor, away_embed: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Predict match outcome from team embeddings + context."""
        # Concatenate embeddings and context
        x = torch.cat([home_embed, away_embed, context], dim=-1)
        logits = self.match_mlp(x)
        return F.softmax(logits, dim=-1)

    def forward(self, data: Data, home_indices: torch.Tensor, away_indices: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Full forward: graph -> embeddings -> predictions."""
        embeddings = self.get_embeddings(data)
        home_embed = embeddings[home_indices]
        away_embed = embeddings[away_indices]
        return self.predict_match(home_embed, away_embed, context)


# ──────────────────────────────────────────────
# CONTEXT FEATURES (match-level, not team-level)
# ──────────────────────────────────────────────

def get_context_features(row: pd.Series) -> list:
    """
    Extract match-level context features (7 dims):
    - odds_implied_home, draw, away (market prior)
    - h2h_draw_rate
    - fatigue_rest_advantage
    - style_attacking_diff
    - style_defensive_diff
    """
    return [
        row.get('odds_implied_home', 0.4),
        row.get('odds_implied_draw', 0.27),
        row.get('odds_implied_away', 0.33),
        row.get('h2h_draw_rate', 0.27),
        row.get('fatigue_rest_advantage', 0.0) / 7.0,  # Normalize
        row.get('style_attacking_diff', 0.0),
        row.get('style_defensive_diff', 0.0),
    ]


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def compute_brier(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_oh = np.zeros((len(y_true), n_classes))
    for i, l in enumerate(y_true):
        y_oh[i, l] = 1
    return np.mean(np.sum((y_prob - y_oh) ** 2, axis=1))


def train_gnn(df: pd.DataFrame, team_idx: dict, n_splits: int = 3, n_epochs: int = 50, lr: float = 0.001):
    """
    Walk-forward training of GNN.

    For each fold:
    1. Build graph from training matches
    2. Train GNN to predict match outcomes
    3. Extract team embeddings for test matches
    4. Evaluate predictions
    """
    print(f"\n=== GNN Walk-Forward Training ({n_splits} folds) ===")

    label_map = {'H': 0, 'D': 1, 'A': 2}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(len(df))

    all_results = []
    all_embeddings = {}  # matchId -> (home_embed, away_embed)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(indices)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        print(f"\nFold {fold + 1}: train={len(train_df)}, test={len(test_df)}")
        test_start = test_df['date'].min().strftime('%Y-%m')
        test_end = test_df['date'].max().strftime('%Y-%m')
        print(f"  Test period: {test_start} to {test_end}")

        # Initialize model
        model = FootballGNN(node_features=12, hidden_dim=64, embed_dim=32, context_dim=7)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        # Training: process matches in temporal batches (monthly)
        train_df_sorted = train_df.sort_values('date')

        # Group by month for batch training
        train_df_sorted['month_key'] = train_df_sorted['date'].dt.to_period('M')
        months = train_df_sorted['month_key'].unique()

        best_loss = float('inf')
        patience = 10
        no_improve = 0

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            for month in months:
                month_matches = train_df_sorted[train_df_sorted['month_key'] == month]
                if len(month_matches) == 0:
                    continue

                # Build graph from all matches BEFORE this month
                month_start = month_matches['date'].min()
                graph = build_match_graph(df, team_idx, month_start, lookback_matches=300)

                # Prepare batch predictions
                home_indices = []
                away_indices = []
                contexts = []
                labels = []

                for _, m in month_matches.iterrows():
                    h_idx = team_idx.get(m['homeTeam'])
                    a_idx = team_idx.get(m['awayTeam'])
                    if h_idx is None or a_idx is None:
                        continue

                    home_indices.append(h_idx)
                    away_indices.append(a_idx)
                    contexts.append(get_context_features(m))
                    labels.append(label_map[m['result']])

                if len(labels) == 0:
                    continue

                home_t = torch.tensor(home_indices, dtype=torch.long)
                away_t = torch.tensor(away_indices, dtype=torch.long)
                context_t = torch.tensor(contexts, dtype=torch.float)
                label_t = torch.tensor(labels, dtype=torch.long)

                # Forward pass
                probs = model(graph, home_t, away_t, context_t)
                loss = F.cross_entropy(torch.log(probs + 1e-8), label_t)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item() * len(labels)
                epoch_correct += (probs.argmax(dim=1) == label_t).sum().item()
                epoch_total += len(labels)

            scheduler.step()

            avg_loss = epoch_loss / max(epoch_total, 1)
            avg_acc = epoch_correct / max(epoch_total, 1)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}, acc={avg_acc:.2%}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        model.load_state_dict(best_state)
        model.eval()

        # Evaluate on test set
        test_probs = []
        test_labels = []
        test_match_ids = []

        with torch.no_grad():
            for _, m in test_df.iterrows():
                h_idx = team_idx.get(m['homeTeam'])
                a_idx = team_idx.get(m['awayTeam'])
                if h_idx is None or a_idx is None:
                    continue

                # Build graph up to this match's date
                graph = build_match_graph(df, team_idx, m['date'], lookback_matches=300)
                embeddings = model.get_embeddings(graph)

                home_embed = embeddings[h_idx].unsqueeze(0)
                away_embed = embeddings[a_idx].unsqueeze(0)
                context = torch.tensor([get_context_features(m)], dtype=torch.float)

                probs = model.predict_match(home_embed, away_embed, context)

                test_probs.append(probs.numpy()[0])
                test_labels.append(label_map[m['result']])
                test_match_ids.append(m['matchId'])

                # Store embeddings
                all_embeddings[m['matchId']] = {
                    'home_embed': embeddings[h_idx].numpy().tolist(),
                    'away_embed': embeddings[a_idx].numpy().tolist(),
                    'gnn_prob_home': float(probs[0][0]),
                    'gnn_prob_draw': float(probs[0][1]),
                    'gnn_prob_away': float(probs[0][2]),
                }

        if len(test_probs) == 0:
            print("  No valid test predictions")
            continue

        test_probs = np.array(test_probs)
        test_labels = np.array(test_labels)

        brier = compute_brier(test_labels, test_probs)
        acc = accuracy_score(test_labels, test_probs.argmax(axis=1))

        print(f"  Test Brier: {brier:.4f}, Accuracy: {acc:.2%}")

        all_results.append({
            'fold': fold + 1,
            'test_period': f"{test_start} to {test_end}",
            'brier': float(brier),
            'accuracy': float(acc),
            'n_test': len(test_labels),
        })

    # Aggregate
    if all_results:
        avg_brier = np.mean([r['brier'] for r in all_results])
        avg_acc = np.mean([r['accuracy'] for r in all_results])
        print(f"\n=== GNN Aggregate Results ===")
        print(f"  Mean Brier: {avg_brier:.4f}")
        print(f"  Mean Accuracy: {avg_acc:.2%}")

    return all_results, all_embeddings, model


# ──────────────────────────────────────────────
# FINAL TRAINING + EMBEDDING EXPORT
# ──────────────────────────────────────────────

def train_final_and_export(df: pd.DataFrame, team_idx: dict, output_path: str, n_epochs: int = 80):
    """Train final GNN on all data and export embeddings for every match."""
    print("\n=== Training Final GNN ===")

    label_map = {'H': 0, 'D': 1, 'A': 2}

    model = FootballGNN(node_features=12, hidden_dim=64, embed_dim=32, context_dim=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    df_sorted = df.sort_values('date')
    df_sorted['month_key'] = df_sorted['date'].dt.to_period('M')
    months = df_sorted['month_key'].unique()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_total = 0

        for month in months:
            month_matches = df_sorted[df_sorted['month_key'] == month]
            if len(month_matches) == 0:
                continue

            month_start = month_matches['date'].min()
            graph = build_match_graph(df, team_idx, month_start, lookback_matches=300)

            home_indices, away_indices, contexts, labels = [], [], [], []
            for _, m in month_matches.iterrows():
                h_idx = team_idx.get(m['homeTeam'])
                a_idx = team_idx.get(m['awayTeam'])
                if h_idx is None or a_idx is None:
                    continue
                home_indices.append(h_idx)
                away_indices.append(a_idx)
                contexts.append(get_context_features(m))
                labels.append(label_map[m['result']])

            if not labels:
                continue

            probs = model(
                graph,
                torch.tensor(home_indices, dtype=torch.long),
                torch.tensor(away_indices, dtype=torch.long),
                torch.tensor(contexts, dtype=torch.float),
            )
            loss = F.cross_entropy(torch.log(probs + 1e-8), torch.tensor(labels, dtype=torch.long))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            epoch_total += len(labels)

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: loss={epoch_loss / max(epoch_total, 1):.4f}")

    # Export embeddings for every match
    print("\n  Exporting embeddings...")
    model.eval()
    all_embeddings = {}

    with torch.no_grad():
        for _, m in df_sorted.iterrows():
            h_idx = team_idx.get(m['homeTeam'])
            a_idx = team_idx.get(m['awayTeam'])
            if h_idx is None or a_idx is None:
                continue

            graph = build_match_graph(df, team_idx, m['date'], lookback_matches=300)
            embeddings = model.get_embeddings(graph)

            home_embed = embeddings[h_idx]
            away_embed = embeddings[a_idx]
            context = torch.tensor([get_context_features(m)], dtype=torch.float)
            probs = model.predict_match(home_embed.unsqueeze(0), away_embed.unsqueeze(0), context)

            all_embeddings[m['matchId']] = {
                'home_embed': home_embed.numpy().tolist(),
                'away_embed': away_embed.numpy().tolist(),
                'gnn_prob_home': float(probs[0][0]),
                'gnn_prob_draw': float(probs[0][1]),
                'gnn_prob_away': float(probs[0][2]),
            }

    # Save
    with open(output_path, 'w') as f:
        json.dump(all_embeddings, f)

    print(f"  Saved {len(all_embeddings)} match embeddings to {output_path}")

    # Save model
    model_path = output_path.replace('.json', '_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"  Saved model to {model_path}")

    return model, all_embeddings


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train GNN for BetAI')
    parser.add_argument('--data', required=True, help='Path to features JSON')
    parser.add_argument('--output', default='gnn_embeddings.json', help='Output embeddings path')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    args = parser.parse_args()

    print("=== BetAI v2 - GNN Training ===\n")

    df = load_features(args.data)
    print(f"Loaded {len(df)} matches")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    team_idx = build_team_index(df)
    print(f"Teams: {len(team_idx)}")

    # Walk-forward validation
    results, embeddings, model = train_gnn(df, team_idx, n_splits=3, n_epochs=args.epochs)

    if args.validate_only:
        print("\n[validate-only mode]")
        # Save validation results
        with open(args.output.replace('.json', '_validation.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return

    # Train final model and export embeddings
    model, all_embeddings = train_final_and_export(df, team_idx, args.output, n_epochs=args.epochs + 30)

    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()
