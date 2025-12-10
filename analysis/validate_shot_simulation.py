"""
shot-level simulation validation

validates shot simulation against historical charting data
compares serve placements, rally lengths, and outcome distributions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict

from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator
from simulation.point_engine import PointSimulator
from simulation.serve_model import ServeModel
from simulation.return_model import ReturnModel
from simulation.rally_model import RallyModel
from simulation.charting_loader import ChartingDataLoader, load_charting_data


def validate_serve_placements(num_sims: int = 1000,
                              player_name: str = 'Novak Djokovic',
                              surface: str = 'hard') -> Dict:
    """
    validate serve placement model against historical data

    args:
        num_sims: number of serves to simulate
        player_name: player to validate
        surface: court surface

    returns:
        dict with validation statistics
    """
    print("=" * 70)
    print(f"VALIDATING SERVE PLACEMENTS ({player_name})")
    print("=" * 70)
    print()

    # load charting data to get historical patterns
    loader = ChartingDataLoader()
    loader.load_charting_data(tour='atp')
    serve_patterns = loader.extract_serve_patterns(min_points=30)

    # check if player exists in data
    player_pattern = serve_patterns.get(player_name)

    if player_pattern is None:
        print(f"warning: {player_name} not found in charting data, using default pattern")
        player_pattern = loader.get_default_serve_pattern()
        historical_placements = player_pattern
    else:
        historical_placements = {
            'wide': player_pattern['wide_pct'],
            'T': player_pattern['T_pct'],
            'body': player_pattern['body_pct']
        }

    # create serve model
    serve_model = ServeModel(serve_patterns=serve_patterns)

    # simulate serves
    simulated_placements = defaultdict(int)
    for _ in range(num_sims):
        placement = serve_model.sample_serve_placement(
            player_name=player_name,
            surface=surface,
            is_second_serve=False
        )
        simulated_placements[placement.value] += 1

    # convert to percentages
    sim_pcts = {
        'wide': simulated_placements['W'] / num_sims,
        'T': simulated_placements['T'] / num_sims,
        'body': simulated_placements['B'] / num_sims
    }

    # compute differences
    print(f"serve placement comparison (n={num_sims} serves):")
    print(f"{'Placement':<12} {'Historical':<15} {'Simulated':<15} {'Difference':<15}")
    print("-" * 70)

    for placement in ['wide', 'T', 'body']:
        hist = historical_placements[placement]
        sim = sim_pcts[placement]
        diff = sim - hist

        print(f"{placement.capitalize():<12} {hist:>6.1%}          {sim:>6.1%}          {diff:>+6.1%}")

    print()

    return {
        'historical': historical_placements,
        'simulated': sim_pcts,
        'num_sims': num_sims
    }


def validate_rally_lengths(num_sims: int = 500,
                          surface: str = 'hard') -> Dict:
    """
    validate rally length distributions

    args:
        num_sims: number of points to simulate
        surface: court surface

    returns:
        dict with validation statistics
    """
    print("=" * 70)
    print(f"VALIDATING RALLY LENGTHS (n={num_sims} points)")
    print("=" * 70)
    print()

    # load data
    data_loader = DataLoader(data_dir='data')
    matches = data_loader.load_match_data(years=[2023], tour='atp')

    if matches is None or len(matches) == 0:
        print("warning: no match data available, using synthetic data")
        matches = data_loader._create_synthetic_data()

    # create stats calculator
    stats_calc = PlayerStatsCalculator(matches)

    # get two players
    players = matches['winner_name'].value_counts().head(10).index.tolist()
    if len(players) < 2:
        print("error: not enough player data")
        return {}

    player_a = players[0]
    player_b = players[1]

    # compute stats
    stats_a = stats_calc.get_player_stats(player_a, surface=surface)
    stats_b = stats_calc.get_player_stats(player_b, surface=surface)

    # load trained models
    import pickle
    import os

    serve_patterns = {}
    rally_patterns = {}

    # try to load serve patterns
    serve_pkl_path = 'models/serve_patterns.pkl'
    if os.path.exists(serve_pkl_path):
        print(f"loading trained serve patterns from {serve_pkl_path}")
        with open(serve_pkl_path, 'rb') as f:
            serve_data = pickle.load(f)
            serve_patterns = serve_data.get('serve_patterns', {})
            print(f"loaded serve patterns for {len(serve_patterns)} players")
    else:
        print(f"warning: {serve_pkl_path} not found, using defaults")

    # try to load rally patterns
    rally_pkl_path = 'models/rally_patterns.pkl'
    if os.path.exists(rally_pkl_path):
        print(f"loading trained rally patterns from {rally_pkl_path}")
        with open(rally_pkl_path, 'rb') as f:
            rally_data = pickle.load(f)
            rally_patterns = rally_data.get('rally_patterns', {})
            print(f"loaded rally patterns for {len(rally_patterns)} players")
    else:
        print(f"warning: {rally_pkl_path} not found, using defaults")

    # create shot-level simulator with trained models
    serve_model = ServeModel(serve_patterns=serve_patterns)
    return_model = ReturnModel()
    rally_model = RallyModel(rally_patterns=rally_patterns)

    point_sim = PointSimulator(
        rng=np.random.default_rng(42),
        use_shot_simulation=True,
        serve_model=serve_model,
        return_model=return_model,
        rally_model=rally_model
    )

    # simulate points
    rally_lengths = []
    serve_placements = defaultdict(int)
    aces = 0
    double_faults = 0

    for i in range(num_sims):
        # alternate server
        if i % 2 == 0:
            server_stats = stats_a
            returner_stats = stats_b
            server_name = player_a
            returner_name = player_b
        else:
            server_stats = stats_b
            returner_stats = stats_a
            server_name = player_b
            returner_name = player_a

        # simulate point
        result = point_sim.simulate_point(
            server_stats=server_stats,
            returner_stats=returner_stats,
            server_name=server_name,
            returner_name=returner_name,
            surface=surface
        )

        rally_lengths.append(result.rally_length)

        if result.serve_placement:
            serve_placements[result.serve_placement.value] += 1

        if result.was_ace:
            aces += 1
        if result.was_double_fault:
            double_faults += 1

    # compute statistics
    rally_lengths = np.array(rally_lengths)

    print(f"rally length statistics:")
    print(f"  mean:   {np.mean(rally_lengths):.2f} shots")
    print(f"  median: {np.median(rally_lengths):.1f} shots")
    print(f"  std:    {np.std(rally_lengths):.2f} shots")
    print(f"  min:    {np.min(rally_lengths)} shots")
    print(f"  max:    {np.max(rally_lengths)} shots")
    print()

    # rally length distribution
    print("rally length distribution:")
    unique, counts = np.unique(rally_lengths, return_counts=True)
    for length, count in zip(unique[:15], counts[:15]):  # show first 15
        pct = count / len(rally_lengths)
        bar = '#' * int(pct * 50)
        print(f"  {int(length):2d} shots: {pct:>6.1%} {bar}")

    print()

    # serve placement distribution
    print("serve placement distribution:")
    total_placements = sum(serve_placements.values())
    if total_placements > 0:
        for placement, count in serve_placements.items():
            print(f"  {placement}: {count / total_placements:.1%}")
    print()

    # outcome statistics
    print("point outcomes:")
    print(f"  aces:          {aces} ({aces / num_sims:.1%})")
    print(f"  double faults: {double_faults} ({double_faults / num_sims:.1%})")
    print()

    return {
        'rally_lengths': rally_lengths,
        'serve_placements': dict(serve_placements),
        'aces': aces,
        'double_faults': double_faults,
        'num_sims': num_sims
    }


def compare_point_vs_shot_simulation(num_sims: int = 100,
                                    surface: str = 'hard') -> Dict:
    """
    compare point-level vs shot-level simulation outcomes

    args:
        num_sims: number of matches to simulate
        surface: court surface

    returns:
        dict with comparison statistics
    """
    print("=" * 70)
    print(f"COMPARING POINT-LEVEL VS SHOT-LEVEL SIMULATION")
    print("=" * 70)
    print()

    # load data
    data_loader = DataLoader(data_dir='data')
    matches = data_loader.load_match_data(years=[2023], tour='atp')

    if matches is None or len(matches) == 0:
        matches = data_loader._create_synthetic_data()

    # create stats calculator
    stats_calc = PlayerStatsCalculator(matches)

    # get two players
    players = matches['winner_name'].value_counts().head(10).index.tolist()
    player_a = players[0]
    player_b = players[1]

    # compute stats
    stats_a = stats_calc.get_player_stats(player_a, surface=surface)
    stats_b = stats_calc.get_player_stats(player_b, surface=surface)

    # create both simulators
    point_sim_legacy = PointSimulator(
        rng=np.random.default_rng(42),
        use_shot_simulation=False
    )

    point_sim_shot = PointSimulator(
        rng=np.random.default_rng(42),
        use_shot_simulation=True
    )

    # simulate with both
    print(f"simulating {num_sims} points with each method...")

    legacy_wins_a = 0
    shot_wins_a = 0

    for i in range(num_sims):
        # legacy simulation
        result_legacy = point_sim_legacy.simulate_point(stats_a, stats_b)
        if result_legacy.server_won:
            legacy_wins_a += 1

        # shot simulation
        result_shot = point_sim_shot.simulate_point(
            stats_a, stats_b, player_a, player_b, surface
        )
        if result_shot.server_won:
            shot_wins_a += 1

    print()
    print(f"results ({player_a} serving):")
    print(f"  point-level simulation:  {legacy_wins_a}/{num_sims} ({legacy_wins_a / num_sims:.1%})")
    print(f"  shot-level simulation:   {shot_wins_a}/{num_sims} ({shot_wins_a / num_sims:.1%})")
    print(f"  difference:              {abs(legacy_wins_a - shot_wins_a)} points ({abs(legacy_wins_a - shot_wins_a) / num_sims:.1%})")
    print()

    return {
        'legacy_win_pct': legacy_wins_a / num_sims,
        'shot_win_pct': shot_wins_a / num_sims,
        'num_sims': num_sims
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate shot-level simulation')
    parser.add_argument('--validate', type=str, default='all',
                       choices=['serve', 'rally', 'compare', 'all'],
                       help='What to validate')
    parser.add_argument('--num_sims', type=int, default=500,
                       help='Number of simulations')
    parser.add_argument('--player', type=str, default='Novak Djokovic',
                       help='Player name for serve validation')
    parser.add_argument('--surface', type=str, default='hard',
                       choices=['hard', 'clay', 'grass'],
                       help='Court surface')

    args = parser.parse_args()

    if args.validate in ['serve', 'all']:
        validate_serve_placements(
            num_sims=args.num_sims,
            player_name=args.player,
            surface=args.surface
        )
        print()

    if args.validate in ['rally', 'all']:
        validate_rally_lengths(
            num_sims=args.num_sims,
            surface=args.surface
        )
        print()

    if args.validate in ['compare', 'all']:
        compare_point_vs_shot_simulation(
            num_sims=args.num_sims,
            surface=args.surface
        )
        print()

    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
