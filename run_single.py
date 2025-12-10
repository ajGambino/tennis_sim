"""
single match simulation runner

command-line interface for simulating a single tennis match
"""

import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator
from simulation.match_engine import MatchSimulator
from simulation.elo_system import EloSystem
from simulation.ranking_adjuster import EloAdjuster


def main():
    """run single match simulation from command line"""

    parser = argparse.ArgumentParser(
        description='simulate a single tennis match',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz"
  python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --surface hard --best_of 5
  python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --seed 12345
        """
    )

    parser.add_argument('--playerA', type=str, required=True,
                       help='name of player a')
    parser.add_argument('--playerB', type=str, required=True,
                       help='name of player b')
    parser.add_argument('--surface', type=str, default=None,
                       choices=['hard', 'clay', 'grass', 'carpet'],
                       help='surface type (default: all surfaces)')
    parser.add_argument('--best_of', type=int, default=3,
                       choices=[3, 5],
                       help='best of 3 or 5 sets (default: 3)')
    parser.add_argument('--seed', type=int, default=None,
                       help='random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='directory containing match data csvs (default: data/)')
    parser.add_argument('--tour', type=str, default='atp',
                       choices=['atp', 'wta'],
                       help='tour type (default: atp)')
    parser.add_argument('--years', type=int, nargs='+', default=None,
                       help='years to load data from (default: 2015-2024)')
    parser.add_argument('--use_elo', action='store_true',
                       help='use elo rating adjustments (default: false)')
    parser.add_argument('--elo_file', type=str, default=None,
                       help='path to elo ratings file (default: auto-generate)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"TENNIS MATCH SIMULATION")
    print(f"{'='*70}\n")

    # load match data
    print("loading match data...")
    loader = DataLoader(args.data_dir)
    match_data = loader.load_match_data(years=args.years, tour=args.tour)

    # calculate player statistics
    print("\ncalculating player statistics...")
    stats_calc = PlayerStatsCalculator(match_data)

    # get stats for both players
    player_a_stats = stats_calc.get_player_stats(args.playerA, args.surface)
    player_b_stats = stats_calc.get_player_stats(args.playerB, args.surface)

    # elo adjustments if enabled
    if args.use_elo:
        print("\nelo rating system enabled")

        # load or build elo ratings
        elo_file = args.elo_file or f"models/elo_ratings_{args.tour}.json"

        if os.path.exists(elo_file):
            print(f"loading elo ratings from {elo_file}...")
            elo_system = EloSystem()
            elo_system.load_ratings(elo_file)
        else:
            print("building elo ratings from match history...")
            elo_system = EloSystem(surface_specific=True)
            elo_system.build_from_matches(match_data, verbose=False)
            # save for future use
            os.makedirs('models', exist_ok=True)
            elo_system.save_ratings(elo_file)
            print(f"elo ratings saved to {elo_file}")

        # get elo ratings
        elo_a = elo_system.get_rating(args.playerA, args.surface)
        elo_b = elo_system.get_rating(args.playerB, args.surface)
        win_prob_elo = elo_system.get_win_probability(args.playerA, args.playerB, args.surface)

        print(f"\nelo ratings ({args.surface or 'all surfaces'}):")
        print(f"  {args.playerA}: {elo_a:.0f}")
        print(f"  {args.playerB}: {elo_b:.0f}")
        print(f"  expected win probability: {win_prob_elo:.1%} - {1-win_prob_elo:.1%}")

        # apply elo adjustments
        elo_adjuster = EloAdjuster(max_adjustment=0.15)
        player_a_stats = elo_adjuster.adjust_stats(player_a_stats, elo_a, elo_b)
        player_b_stats = elo_adjuster.adjust_stats(player_b_stats, elo_b, elo_a)

    # print player summaries
    stats_calc.print_player_summary(args.playerA, args.surface)
    stats_calc.print_player_summary(args.playerB, args.surface)

    # run simulation
    mode_str = "elo-adjusted" if args.use_elo else "baseline"
    print(f"simulating match (best of {args.best_of}, {mode_str})...\n")
    simulator = MatchSimulator(player_a_stats, player_b_stats,
                              best_of=args.best_of, seed=args.seed)
    match_stats = simulator.simulate_match()

    # print results
    simulator.print_match_summary()

    # print seed for reproducibility
    print(f"random seed used: {match_stats.random_seed}")
    elo_flag = " --use_elo" if args.use_elo else ""
    print(f"to reproduce this exact match, run:")
    print(f'  python run_single.py --playerA "{args.playerA}" '
          f'--playerB "{args.playerB}" --seed {match_stats.random_seed}{elo_flag}\n')


if __name__ == '__main__':
    main()
