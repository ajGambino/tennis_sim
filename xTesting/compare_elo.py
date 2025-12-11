"""
comparison script: baseline vs elo-adjusted simulations

demonstrates the improvement from elo rating adjustments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.data_loader import load_match_data
from simulation.player_stats import PlayerStatsCalculator
from simulation.elo_system import EloSystem
from simulation.ranking_adjuster import EloAdjuster
from simulation.simulator import run_simulations, analyze_results


def compare_simulations(player_a: str, player_b: str, surface: str = 'hard',
                        n_sims: int = 1000, tour: str = 'atp'):
    """
    run simulations with and without elo adjustments

    args:
        player_a: first player name
        player_b: second player name
        surface: court surface
        n_sims: number of simulations
        tour: 'atp' or 'wta'
    """
    print(f"\n{'='*80}")
    print(f"BASELINE VS ELO COMPARISON")
    print(f"{player_a} vs {player_b} ({surface} court)")
    print(f"{'='*80}\n")

    # load data
    print("loading match data...")
    match_data = load_match_data(tour=tour)

    # get player stats
    stats_calc = PlayerStatsCalculator(match_data)
    stats_a = stats_calc.get_player_stats(player_a, surface)
    stats_b = stats_calc.get_player_stats(player_b, surface)

    print(f"\nbase statistics:")
    print(f"  {player_a}: {stats_a.matches_played} matches, "
          f"{stats_a.first_serve_win_pct:.1%} 1st serve win")
    print(f"  {player_b}: {stats_b.matches_played} matches, "
          f"{stats_b.first_serve_win_pct:.1%} 1st serve win")

    # build elo
    print("\nbuilding elo ratings...")
    import os
    elo_file = f"models/elo_ratings_{tour}.json"

    if os.path.exists(elo_file):
        elo_system = EloSystem()
        elo_system.load_ratings(elo_file)
    else:
        elo_system = EloSystem(surface_specific=True)
        elo_system.build_from_matches(match_data, verbose=False)
        os.makedirs('models', exist_ok=True)
        elo_system.save_ratings(elo_file)

    elo_a = elo_system.get_rating(player_a, surface)
    elo_b = elo_system.get_rating(player_b, surface)
    elo_diff = elo_a - elo_b
    win_prob_elo = elo_system.get_win_probability(player_a, player_b, surface)

    print(f"\nelo ratings:")
    print(f"  {player_a}: {elo_a:.0f}")
    print(f"  {player_b}: {elo_b:.0f}")
    print(f"  differential: {elo_diff:+.0f}")
    print(f"  expected win prob: {win_prob_elo:.1%} - {1-win_prob_elo:.1%}")

    # adjust stats
    elo_adjuster = EloAdjuster(max_adjustment=0.15)
    stats_a_elo = elo_adjuster.adjust_stats(stats_a, elo_a, elo_b)
    stats_b_elo = elo_adjuster.adjust_stats(stats_b, elo_b, elo_a)

    # run baseline simulations
    print(f"\n{'-'*80}")
    print("BASELINE SIMULATION (No Elo)")
    print(f"{'-'*80}")
    results_baseline = run_simulations(stats_a, stats_b, n=n_sims,
                                       best_of=3, start_seed=42)
    summary_baseline = analyze_results(results_baseline, player_a, player_b)

    # run elo simulations
    print(f"\n{'-'*80}")
    print("ELO-ADJUSTED SIMULATION")
    print(f"{'-'*80}")
    results_elo = run_simulations(stats_a_elo, stats_b_elo, n=n_sims,
                                  best_of=3, start_seed=42)
    summary_elo = analyze_results(results_elo, player_a, player_b)

    # comparison table
    print(f"\n{'='*80}")
    print(f"RESULTS COMPARISON ({n_sims} simulations)")
    print(f"{'='*80}\n")

    print(f"{'Metric':<30} {'Baseline':>15} {'Elo-Adjusted':>15} {'Difference':>15}")
    print(f"{'-'*80}")

    baseline_a_win = summary_baseline['a_win_pct']
    elo_a_win = summary_elo['a_win_pct']
    diff = elo_a_win - baseline_a_win

    print(f"{player_a + ' Win %':<30} {baseline_a_win:>14.1%} {elo_a_win:>14.1%} {diff:>+14.1%}")
    print(f"{player_b + ' Win %':<30} {summary_baseline['b_win_pct']:>14.1%} "
          f"{summary_elo['b_win_pct']:>14.1%} {-diff:>+14.1%}")

    print(f"\n{'Expected (Elo Formula)':<30} {'':>15} {win_prob_elo:>14.1%} {'':>15}")

    print(f"\n{'-'*80}")
    print("Average Match Stats:")
    print(f"{'-'*80}")

    print(f"{'Aces ' + player_a:<30} {summary_baseline['mean_aces_a']:>14.1f} "
          f"{summary_elo['mean_aces_a']:>14.1f} "
          f"{summary_elo['mean_aces_a'] - summary_baseline['mean_aces_a']:>+14.1f}")

    print(f"{'Double Faults ' + player_a:<30} {summary_baseline['mean_df_a']:>14.1f} "
          f"{summary_elo['mean_df_a']:>14.1f} "
          f"{summary_elo['mean_df_a'] - summary_baseline['mean_df_a']:>+14.1f}")

    print(f"{'Total Points ' + player_a:<30} {summary_baseline['mean_total_points_a']:>14.1f} "
          f"{summary_elo['mean_total_points_a']:>14.1f} "
          f"{summary_elo['mean_total_points_a'] - summary_baseline['mean_total_points_a']:>+14.1f}")

    print(f"{'='*80}\n")

    return {
        'baseline': summary_baseline,
        'elo': summary_elo,
        'improvement': diff
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='compare baseline vs elo simulations')
    parser.add_argument('--playerA', type=str, default='Jannik Sinner',
                       help='first player name')
    parser.add_argument('--playerB', type=str, default='Gabriel Diallo',
                       help='second player name')
    parser.add_argument('--surface', type=str, default='hard',
                       choices=['hard', 'clay', 'grass'],
                       help='court surface')
    parser.add_argument('--n', type=int, default=1000,
                       help='number of simulations')
    parser.add_argument('--tour', type=str, default='atp',
                       choices=['atp', 'wta'],
                       help='tour type')

    args = parser.parse_args()

    compare_simulations(
        args.playerA,
        args.playerB,
        args.surface,
        args.n,
        args.tour
    )
