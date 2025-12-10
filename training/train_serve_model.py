"""
serve model training pipeline

extracts serve patterns from match charting project data
learns player-specific serve placement tendencies and ace rates
"""

import os
import pickle
from typing import Dict
from simulation.charting_loader import ChartingDataLoader


def train_serve_model(data_dir: str = 'data/charting',
                     tour: str = 'atp',
                     output_path: str = 'models/serve_patterns.pkl',
                     min_points: int = 50) -> Dict:
    """
    train serve placement model from charting data

    args:
        data_dir: directory containing charting csv files
        tour: 'atp' or 'wta'
        output_path: where to save trained serve patterns
        min_points: minimum serve points required per player

    returns:
        dict of serve patterns by player
    """
    print("=" * 70)
    print("SERVE MODEL TRAINING")
    print("=" * 70)
    print()

    # load charting data
    print(f"loading {tour.upper()} match charting project data...")
    loader = ChartingDataLoader(data_dir=data_dir)
    charting_data = loader.load_charting_data(tour=tour)

    if charting_data is None or len(charting_data) == 0:
        print("error: no charting data loaded")
        print("please download match charting project data:")
        print("  git clone https://github.com/JeffSackmann/tennis_MatchChartingProject")
        print(f"  and place csv files in {data_dir}/")
        return {}

    print(f"loaded {len(charting_data)} points from charting data")
    print()

    # extract serve patterns
    print(f"extracting serve patterns (min {min_points} serves per player)...")
    serve_patterns = loader.extract_serve_patterns(min_points=min_points)

    print(f"extracted serve patterns for {len(serve_patterns)} players")
    print()

    # display sample patterns
    print("sample serve patterns:")
    print("-" * 70)

    sample_players = list(serve_patterns.keys())[:10]
    for player in sample_players:
        pattern = serve_patterns[player]
        print(f"\n{player}:")
        print(f"  Placement:  Wide {pattern['wide_pct']:.1%} | T {pattern['T_pct']:.1%} | Body {pattern['body_pct']:.1%}")
        print(f"  Ace by placement: Wide {pattern['ace_pct_by_placement']['W']:.1%} | " +
              f"T {pattern['ace_pct_by_placement']['T']:.1%} | " +
              f"Body {pattern['ace_pct_by_placement']['B']:.1%}")
        print(f"  Total serves: {pattern['total_serves']}")

    print()
    print("=" * 70)

    # compute and display default pattern
    default_pattern = loader.get_default_serve_pattern()
    print("\ndefault serve pattern (average across all players):")
    print(f"  Wide: {default_pattern['wide_pct']:.1%}")
    print(f"  T:    {default_pattern['T_pct']:.1%}")
    print(f"  Body: {default_pattern['body_pct']:.1%}")
    print(f"  Ace rates: W={default_pattern['ace_pct_by_placement']['W']:.1%}, " +
          f"T={default_pattern['ace_pct_by_placement']['T']:.1%}, " +
          f"B={default_pattern['ace_pct_by_placement']['B']:.1%}")

    # save to pickle file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'serve_patterns': serve_patterns,
            'default_pattern': default_pattern,
            'tour': tour,
            'min_points': min_points,
            'num_players': len(serve_patterns)
        }, f)

    print()
    print(f"serve patterns saved to {output_path}")
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return serve_patterns


def load_serve_patterns(path: str = 'models/serve_patterns.pkl') -> Dict:
    """
    load trained serve patterns from file

    args:
        path: path to saved serve patterns

    returns:
        dict with serve patterns and metadata
    """
    if not os.path.exists(path):
        print(f"warning: serve patterns file not found at {path}")
        return None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train serve placement model from charting data')
    parser.add_argument('--data_dir', type=str, default='data/charting',
                       help='Directory containing charting CSV files')
    parser.add_argument('--tour', type=str, default='atp', choices=['atp', 'wta'],
                       help='Tour to train on (atp or wta)')
    parser.add_argument('--output', type=str, default='models/serve_patterns.pkl',
                       help='Output path for trained patterns')
    parser.add_argument('--min_points', type=int, default=50,
                       help='Minimum serve points required per player')

    args = parser.parse_args()

    # train serve model
    serve_patterns = train_serve_model(
        data_dir=args.data_dir,
        tour=args.tour,
        output_path=args.output,
        min_points=args.min_points
    )

    if serve_patterns:
        print(f"\nsuccess! trained serve model with {len(serve_patterns)} players")
    else:
        print("\ntraining failed - check that charting data is available")
