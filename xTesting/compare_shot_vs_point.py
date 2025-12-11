"""
compare point-level vs shot-level simulation

runs both simulation modes and compares outcomes, statistics, and performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import numpy as np
from typing import Dict

from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator
from simulation.match_engine import MatchSimulator
from simulation.point_engine import PointSimulator
from simulation.serve_model import ServeModel
from simulation.return_model import ReturnModel
from simulation.rally_model import RallyModel
from simulation.charting_loader import ChartingDataLoader


def compare_simulation_modes(player_a: str, player_b: str,
                            surface: str = 'hard',
                            best_of: int = 3,
                            num_sims: int = 100) -> Dict:
    """
    compare point-level vs shot-level simulation

    args:
        player_a: first player name
        player_b: second player name
        surface: court surface
        best_of: 3 or 5
        num_sims: number of matches to simulate

    returns:
        dict with comparison results
    """
    print("=" * 80)
    print(f"COMPARING SIMULATION MODES: {player_a} vs {player_b}")
    print("=" * 80)
    print()

    # load data
    print("loading match data...")
    data_loader = DataLoader(data_dir='data')
    matches = data_loader.load_match_data(years=[2020, 2021, 2022, 2023], tour='atp')

    if matches is None or len(matches) == 0:
        print("using synthetic demo data")
        matches = data_loader._create_synthetic_data()

    # compute player stats
    stats_calc = PlayerStatsCalculator(matches)
    stats_a = stats_calc.get_player_stats(player_a, surface=surface)
    stats_b = stats_calc.get_player_stats(player_b, surface=surface)

    print(f"player stats computed for {player_a} and {player_b}")
    print()

    # print player stats comparison
    print(f"{'Statistic':<25} {player_a:<20} {player_b:<20}")
    print("-" * 80)
    print(f"{'first serve %':<25} {stats_a.first_serve_pct:>6.1%}               {stats_b.first_serve_pct:>6.1%}")
    print(f"{'ace %':<25} {stats_a.ace_pct:>6.1%}               {stats_b.ace_pct:>6.1%}")
    print(f"{'double fault %':<25} {stats_a.df_pct:>6.1%}               {stats_b.df_pct:>6.1%}")
    print(f"{'1st serve win %':<25} {stats_a.first_serve_win_pct:>6.1%}               {stats_b.first_serve_win_pct:>6.1%}")
    print(f"{'2nd serve win %':<25} {stats_a.second_serve_win_pct:>6.1%}               {stats_b.second_serve_win_pct:>6.1%}")
    print()

    # ===================================================================
    # MODE 1: POINT-LEVEL SIMULATION (LEGACY)
    # ===================================================================

    print("=" * 80)
    print("MODE 1: POINT-LEVEL SIMULATION (LEGACY)")
    print("=" * 80)
    print()

    print(f"simulating {num_sims} matches...")
    start_time = time.time()

    legacy_results = {
        'player_a_wins': 0,
        'player_b_wins': 0,
        'total_aces_a': 0,
        'total_aces_b': 0,
        'total_dfs_a': 0,
        'total_dfs_b': 0,
    }

    for i in range(num_sims):
        # create point simulator with new seed for each match
        point_simulator = PointSimulator(
            rng=np.random.default_rng(42 + i),
            use_shot_simulation=False
        )

        sim_legacy = MatchSimulator(
            player_a_stats=stats_a,
            player_b_stats=stats_b,
            best_of=best_of,
            seed=42 + i,
            point_simulator=point_simulator
        )

        result = sim_legacy.simulate_match()

        if result.winner == player_a:
            legacy_results['player_a_wins'] += 1
        else:
            legacy_results['player_b_wins'] += 1

        legacy_results['total_aces_a'] += result.aces_a
        legacy_results['total_aces_b'] += result.aces_b
        legacy_results['total_dfs_a'] += result.double_faults_a
        legacy_results['total_dfs_b'] += result.double_faults_b

    legacy_time = time.time() - start_time

    print(f"completed in {legacy_time:.2f} seconds ({legacy_time / num_sims * 1000:.1f} ms/match)")
    print()

    print("results:")
    print(f"  {player_a} wins: {legacy_results['player_a_wins']}/{num_sims} ({legacy_results['player_a_wins'] / num_sims:.1%})")
    print(f"  {player_b} wins: {legacy_results['player_b_wins']}/{num_sims} ({legacy_results['player_b_wins'] / num_sims:.1%})")
    print(f"  avg aces per match ({player_a}): {legacy_results['total_aces_a'] / num_sims:.1f}")
    print(f"  avg aces per match ({player_b}): {legacy_results['total_aces_b'] / num_sims:.1f}")
    print()

    # ===================================================================
    # MODE 2: SHOT-LEVEL SIMULATION (NEW)
    # ===================================================================

    print("=" * 80)
    print("MODE 2: SHOT-LEVEL SIMULATION (NEW)")
    print("=" * 80)
    print()

    # load trained models (shared across all matches)
    import pickle
    import os

    serve_patterns = {}
    rally_patterns = {}

    # try to load serve patterns
    serve_pkl_path = 'models/serve_patterns.pkl'
    if os.path.exists(serve_pkl_path):
        with open(serve_pkl_path, 'rb') as f:
            serve_data = pickle.load(f)
            serve_patterns = serve_data.get('serve_patterns', {})
            print(f"loaded serve patterns for {len(serve_patterns)} players")

    # try to load rally patterns
    rally_pkl_path = 'models/rally_patterns.pkl'
    if os.path.exists(rally_pkl_path):
        with open(rally_pkl_path, 'rb') as f:
            rally_data = pickle.load(f)
            rally_patterns = rally_data.get('rally_patterns', {})
            print(f"loaded rally patterns for {len(rally_patterns)} players")

    # create shot-level models with trained patterns
    serve_model = ServeModel(serve_patterns=serve_patterns)
    return_model = ReturnModel()
    rally_model = RallyModel(rally_patterns=rally_patterns)

    print(f"simulating {num_sims} matches...")
    start_time = time.time()

    shot_results = {
        'player_a_wins': 0,
        'player_b_wins': 0,
        'total_aces_a': 0,
        'total_aces_b': 0,
        'total_dfs_a': 0,
        'total_dfs_b': 0,
        'total_rally_length': 0,
        'num_points': 0,
    }

    for i in range(num_sims):
        # create point simulator with new seed for each match
        point_simulator_shot = PointSimulator(
            rng=np.random.default_rng(42 + i),
            use_shot_simulation=True,
            serve_model=serve_model,
            return_model=return_model,
            rally_model=rally_model
        )

        sim_shot = MatchSimulator(
            player_a_stats=stats_a,
            player_b_stats=stats_b,
            best_of=best_of,
            seed=42 + i,
            point_simulator=point_simulator_shot
        )

        result = sim_shot.simulate_match()

        if result.winner == player_a:
            shot_results['player_a_wins'] += 1
        else:
            shot_results['player_b_wins'] += 1

        shot_results['total_aces_a'] += result.aces_a
        shot_results['total_aces_b'] += result.aces_b
        shot_results['total_dfs_a'] += result.double_faults_a
        shot_results['total_dfs_b'] += result.double_faults_b

    shot_time = time.time() - start_time

    print(f"completed in {shot_time:.2f} seconds ({shot_time / num_sims * 1000:.1f} ms/match)")
    print()

    print("results:")
    print(f"  {player_a} wins: {shot_results['player_a_wins']}/{num_sims} ({shot_results['player_a_wins'] / num_sims:.1%})")
    print(f"  {player_b} wins: {shot_results['player_b_wins']}/{num_sims} ({shot_results['player_b_wins'] / num_sims:.1%})")
    print(f"  avg aces per match ({player_a}): {shot_results['total_aces_a'] / num_sims:.1f}")
    print(f"  avg aces per match ({player_b}): {shot_results['total_aces_b'] / num_sims:.1f}")
    print()

    # ===================================================================
    # COMPARISON SUMMARY
    # ===================================================================

    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Metric':<35} {'Point-Level':<20} {'Shot-Level':<20} {'Difference':<15}")
    print("-" * 80)

    # win percentages
    legacy_win_pct_a = legacy_results['player_a_wins'] / num_sims
    shot_win_pct_a = shot_results['player_a_wins'] / num_sims
    diff_win_pct = shot_win_pct_a - legacy_win_pct_a

    print(f"{player_a + ' win %':<35} {legacy_win_pct_a:>6.1%}               {shot_win_pct_a:>6.1%}               {diff_win_pct:>+6.1%}")

    # aces
    legacy_aces_a = legacy_results['total_aces_a'] / num_sims
    shot_aces_a = shot_results['total_aces_a'] / num_sims
    diff_aces = shot_aces_a - legacy_aces_a

    print(f"{'avg aces/match (' + player_a + ')':<35} {legacy_aces_a:>6.1f}               {shot_aces_a:>6.1f}               {diff_aces:>+6.1f}")

    # performance
    print()
    print(f"{'performance':<35} {'Point-Level':<20} {'Shot-Level':<20} {'Slowdown':<15}")
    print("-" * 80)
    print(f"{'total time (s)':<35} {legacy_time:>6.2f}               {shot_time:>6.2f}               {shot_time / legacy_time:>6.2f}x")
    print(f"{'time per match (ms)':<35} {legacy_time / num_sims * 1000:>6.1f}               {shot_time / num_sims * 1000:>6.1f}               {(shot_time / num_sims) / (legacy_time / num_sims):>6.2f}x")

    print()
    print("=" * 80)
    print()

    return {
        'legacy': legacy_results,
        'shot': shot_results,
        'legacy_time': legacy_time,
        'shot_time': shot_time,
        'num_sims': num_sims
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare point-level vs shot-level simulation'
    )
    parser.add_argument('--playerA', type=str, default='Jannik Sinner',
                       help='First player name')
    parser.add_argument('--playerB', type=str, default='Carlos Alcaraz',
                       help='Second player name')
    parser.add_argument('--surface', type=str, default='hard',
                       choices=['hard', 'clay', 'grass'],
                       help='Court surface')
    parser.add_argument('--best_of', type=int, default=3, choices=[3, 5],
                       help='Best of 3 or 5 sets')
    parser.add_argument('--n', type=int, default=100,
                       help='Number of matches to simulate')

    args = parser.parse_args()

    # run comparison
    results = compare_simulation_modes(
        player_a=args.playerA,
        player_b=args.playerB,
        surface=args.surface,
        best_of=args.best_of,
        num_sims=args.n
    )

    print("comparison complete!")
