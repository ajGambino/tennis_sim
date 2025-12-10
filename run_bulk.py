"""
bulk simulation runner

command-line interface for running monte carlo tennis match simulations
"""

import argparse
import sys
import os
from datetime import datetime
from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator
from simulation.simulator import run_simulations, analyze_results, print_analysis, save_results
from simulation.elo_system import EloSystem
from simulation.ranking_adjuster import EloAdjuster


def main():
    """run bulk monte carlo simulations from command line"""

    parser = argparse.ArgumentParser(
        description='run monte carlo tennis match simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 1000
  python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000 --surface clay
  python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 1000 --best_of 5 --output final_results.csv
        """
    )

    parser.add_argument('--playerA', type=str, required=True,
                       help='name of player a')
    parser.add_argument('--playerB', type=str, required=True,
                       help='name of player b')
    parser.add_argument('--n', type=int, default=1000,
                       help='number of simulations to run (default: 1000)')
    parser.add_argument('--surface', type=str, default=None,
                       choices=['hard', 'clay', 'grass', 'carpet'],
                       help='surface type (default: all surfaces)')
    parser.add_argument('--best_of', type=int, default=3,
                       choices=[3, 5],
                       help='best of 3 or 5 sets (default: 3)')
    parser.add_argument('--seed', type=int, default=None,
                       help='starting random seed (default: random)')
    parser.add_argument('--output', type=str, default=None,
                       help='output csv filename (default: auto-generated)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='directory containing match data csvs (default: data/)')
    parser.add_argument('--tour', type=str, default='atp',
                       choices=['atp', 'wta'],
                       help='tour type (default: atp)')
    parser.add_argument('--years', type=int, nargs='+', default=None,
                       help='years to load data from (default: 2015-2024)')
    parser.add_argument('--no_save', action='store_true',
                       help='do not save results to csv')
    parser.add_argument('--use_elo', action='store_true',
                       help='use elo rating adjustments (default: false)')
    parser.add_argument('--elo_file', type=str, default=None,
                       help='path to elo ratings file (default: auto-generate)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"MONTE CARLO TENNIS MATCH SIMULATION")
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

    # print brief summaries
    mode_str = " (elo-adjusted)" if args.use_elo else ""
    print(f"\n{args.playerA} stats{mode_str}:")
    print(f"  matches: {player_a_stats.matches_played}, "
          f"1st serve: {player_a_stats.first_serve_pct:.1%}, "
          f"1st serve win: {player_a_stats.first_serve_win_pct:.1%}")

    print(f"\n{args.playerB} stats{mode_str}:")
    print(f"  matches: {player_b_stats.matches_played}, "
          f"1st serve: {player_b_stats.first_serve_pct:.1%}, "
          f"1st serve win: {player_b_stats.first_serve_win_pct:.1%}")

    print()

    # run simulations
    results_df = run_simulations(
        player_a_stats,
        player_b_stats,
        n=args.n,
        best_of=args.best_of,
        start_seed=args.seed
    )

    # analyze results
    summary = analyze_results(results_df, args.playerA, args.playerB)

    # print analysis
    print_analysis(summary)

    # save results
    if not args.no_save:
        if args.output:
            filename = args.output
        else:
            # auto-generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            player_a_short = args.playerA.replace(' ', '_')
            player_b_short = args.playerB.replace(' ', '_')
            surface_str = f"_{args.surface}" if args.surface else ""
            elo_str = "_elo" if args.use_elo else ""
            filename = f"{player_a_short}_vs_{player_b_short}{surface_str}{elo_str}_{args.n}sims_{timestamp}.csv"

        save_results(results_df, filename)

    print("\ndone!")


if __name__ == '__main__':
    main()
