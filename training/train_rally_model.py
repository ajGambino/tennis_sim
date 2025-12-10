"""
rally model training pipeline

extracts rally characteristics from match charting project data
learns rally length distributions and shot patterns by player
"""

import os
import pickle
import numpy as np
from typing import Dict
from simulation.charting_loader import ChartingDataLoader


def train_rally_model(data_dir: str = 'data/charting',
                     tour: str = 'atp',
                     output_path: str = 'models/rally_patterns.pkl',
                     min_rallies: int = 30) -> Dict:
    """
    train rally model from charting data

    args:
        data_dir: directory containing charting csv files
        tour: 'atp' or 'wta'
        output_path: where to save trained rally patterns
        min_rallies: minimum rallies required per player

    returns:
        dict of rally patterns by player
    """
    print("=" * 70)
    print("RALLY MODEL TRAINING")
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

    # extract rally patterns
    print(f"extracting rally patterns (min {min_rallies} rallies per player)...")
    rally_patterns = loader.extract_rally_patterns(min_rallies=min_rallies)

    print(f"extracted rally patterns for {len(rally_patterns)} players")
    print()

    # display sample patterns
    print("sample rally patterns:")
    print("-" * 70)

    sample_players = list(rally_patterns.keys())[:10]
    for player in sample_players:
        pattern = rally_patterns[player]
        print(f"\n{player}:")
        print(f"  Average rally length: {pattern['avg_rally_length']:.2f} shots")
        print(f"  Total rallies: {pattern['total_rallies']}")

        # show distribution of rally lengths
        dist = pattern['rally_length_dist']
        if len(dist) > 10:
            # show first 10 rally lengths
            print(f"  Rally length distribution (shots 1-10):")
            for i in range(1, min(11, len(dist))):
                if dist[i] > 0:
                    print(f"    {i} shots: {dist[i]:.1%}")

    print()
    print("=" * 70)

    # compute and display default pattern
    default_pattern = loader.get_default_rally_pattern()
    print("\ndefault rally pattern (average across all players):")
    print(f"  Average rally length: {default_pattern['avg_rally_length']:.2f} shots")
    print(f"  Rally length distribution:")

    dist = default_pattern['rally_length_dist']
    for i in range(1, min(11, len(dist))):
        if dist[i] > 0:
            print(f"    {i} shots: {dist[i]:.1%}")

    # compute aggregate statistics
    all_patterns = list(rally_patterns.values())
    avg_lengths = [p['avg_rally_length'] for p in all_patterns]

    print(f"\naggregate statistics across {len(rally_patterns)} players:")
    print(f"  mean rally length: {np.mean(avg_lengths):.2f} shots")
    print(f"  std rally length:  {np.std(avg_lengths):.2f} shots")
    print(f"  min rally length:  {np.min(avg_lengths):.2f} shots")
    print(f"  max rally length:  {np.max(avg_lengths):.2f} shots")

    # save to pickle file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'rally_patterns': rally_patterns,
            'default_pattern': default_pattern,
            'tour': tour,
            'min_rallies': min_rallies,
            'num_players': len(rally_patterns),
            'avg_rally_length_mean': np.mean(avg_lengths),
            'avg_rally_length_std': np.std(avg_lengths)
        }, f)

    print()
    print(f"rally patterns saved to {output_path}")
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return rally_patterns


def load_rally_patterns(path: str = 'models/rally_patterns.pkl') -> Dict:
    """
    load trained rally patterns from file

    args:
        path: path to saved rally patterns

    returns:
        dict with rally patterns and metadata
    """
    if not os.path.exists(path):
        print(f"warning: rally patterns file not found at {path}")
        return None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train rally model from charting data')
    parser.add_argument('--data_dir', type=str, default='data/charting',
                       help='Directory containing charting CSV files')
    parser.add_argument('--tour', type=str, default='atp', choices=['atp', 'wta'],
                       help='Tour to train on (atp or wta)')
    parser.add_argument('--output', type=str, default='models/rally_patterns.pkl',
                       help='Output path for trained patterns')
    parser.add_argument('--min_rallies', type=int, default=30,
                       help='Minimum rallies required per player')

    args = parser.parse_args()

    # train rally model
    rally_patterns = train_rally_model(
        data_dir=args.data_dir,
        tour=args.tour,
        output_path=args.output,
        min_rallies=args.min_rallies
    )

    if rally_patterns:
        print(f"\nsuccess! trained rally model with {len(rally_patterns)} players")
    else:
        print("\ntraining failed - check that charting data is available")
