"""
compare all prediction methods: baseline vs elo vs ml

demonstrates improvement from baseline → elo → ml
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime

from simulation.data_loader import load_match_data
from simulation.player_stats import PlayerStatsCalculator
from simulation.elo_system import EloSystem
from simulation.ranking_adjuster import EloAdjuster
from simulation.simulator import run_simulations, analyze_results
from simulation.ml_predictor import MLMatchPredictor


def compare_all_methods(player_a: str, player_b: str, surface: str = 'hard',
                       n_sims: int = 1000, tour: str = 'atp'):
    """
    compare all three prediction methods

    args:
        player_a: first player name
        player_b: second player name
        surface: court surface
        n_sims: number of simulations
        tour: 'atp' or 'wta'
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE COMPARISON: BASELINE vs ELO vs ML")
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
    print("\nloading elo ratings...")
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
    print(f"  elo win prob: {win_prob_elo:.1%} - {1-win_prob_elo:.1%}")

    # ml prediction
    ml_model_path = f'models/match_predictor_xgb.pkl'
    ml_available = os.path.exists(ml_model_path)

    if ml_available:
        print("\nloading ml model...")
        predictor = MLMatchPredictor(ml_model_path)
        ml_pred = predictor.predict_match(stats_a, stats_b, elo_a, elo_b, surface)
        ml_win_prob = ml_pred['player_a_win_prob']
        print(f"  ml win prob: {ml_win_prob:.1%} - {1-ml_win_prob:.1%}")
        print(f"  ml confidence: {ml_pred['confidence']:.1%}")
    else:
        print("\nml model not found - skipping ml predictions")
        print(f"  run 'python training/train_point_model.py' to train model")
        ml_win_prob = None

    # adjust stats with elo
    elo_adjuster = EloAdjuster(max_adjustment=0.15)
    stats_a_elo = elo_adjuster.adjust_stats(stats_a, elo_a, elo_b)
    stats_b_elo = elo_adjuster.adjust_stats(stats_b, elo_b, elo_a)

    # run baseline simulations
    print(f"\n{'-'*80}")
    print(f"METHOD 1: BASELINE (No Adjustments)")
    print(f"{'-'*80}")
    results_baseline = run_simulations(stats_a, stats_b, n=n_sims,
                                       best_of=3, start_seed=42)
    summary_baseline = analyze_results(results_baseline, player_a, player_b)

    # run elo simulations
    print(f"\n{'-'*80}")
    print(f"METHOD 2: ELO-ADJUSTED SIMULATION")
    print(f"{'-'*80}")
    results_elo = run_simulations(stats_a_elo, stats_b_elo, n=n_sims,
                                  best_of=3, start_seed=42)
    summary_elo = analyze_results(results_elo, player_a, player_b)

    # comparison table
    print(f"\n{'='*80}")
    print(f"RESULTS COMPARISON ({n_sims} simulations)")
    print(f"{'='*80}\n")

    # create comparison table
    print(f"{'Method':<25} {player_a + ' Win %':>20} {player_b + ' Win %':>20} {'vs Baseline':>12}")
    print(f"{'-'*80}")

    baseline_a = summary_baseline['a_win_pct']
    elo_a_sim = summary_elo['a_win_pct']

    print(f"{'1. Baseline':<25} {baseline_a:>19.1%} {summary_baseline['b_win_pct']:>19.1%} {'--':>12}")
    print(f"{'2. Elo (Simulation)':<25} {elo_a_sim:>19.1%} {summary_elo['b_win_pct']:>19.1%} {elo_a_sim - baseline_a:>+11.1%}")
    print(f"{'3. Elo (Formula)':<25} {win_prob_elo:>19.1%} {1-win_prob_elo:>19.1%} {win_prob_elo - baseline_a:>+11.1%}")

    if ml_available:
        print(f"{'4. ML Model':<25} {ml_win_prob:>19.1%} {1-ml_win_prob:>19.1%} {ml_win_prob - baseline_a:>+11.1%}")

    print(f"\n{'-'*80}")
    print("Improvements Summary:")
    print(f"{'-'*80}")
    print(f"Baseline → Elo Simulation:  {elo_a_sim - baseline_a:+.1%}")
    print(f"Baseline → Elo Formula:     {win_prob_elo - baseline_a:+.1%}")
    if ml_available:
        print(f"Baseline → ML Model:        {ml_win_prob - baseline_a:+.1%}")
        print(f"Elo Formula → ML Model:     {ml_win_prob - win_prob_elo:+.1%}")

    print(f"\n{'-'*80}")
    print("Average Match Stats (from simulations):")
    print(f"{'-'*80}")

    print(f"\n{'Statistic':<30} {'Baseline':>15} {'Elo-Adjusted':>15} {'Difference':>12}")
    print(f"{'-'*80}")

    print(f"{'Aces ' + player_a:<30} {summary_baseline['mean_aces_a']:>14.1f} "
          f"{summary_elo['mean_aces_a']:>14.1f} "
          f"{summary_elo['mean_aces_a'] - summary_baseline['mean_aces_a']:>+11.1f}")

    print(f"{'Double Faults ' + player_a:<30} {summary_baseline['mean_df_a']:>14.1f} "
          f"{summary_elo['mean_df_a']:>14.1f} "
          f"{summary_elo['mean_df_a'] - summary_baseline['mean_df_a']:>+11.1f}")

    print(f"{'Total Points ' + player_a:<30} {summary_baseline['mean_total_points_a']:>14.1f} "
          f"{summary_elo['mean_total_points_a']:>14.1f} "
          f"{summary_elo['mean_total_points_a'] - summary_baseline['mean_total_points_a']:>+11.1f}")

    print(f"{'='*80}\n")

    # return summary
    return {
        'baseline_win_pct': baseline_a,
        'elo_sim_win_pct': elo_a_sim,
        'elo_formula_win_pct': win_prob_elo,
        'ml_win_pct': ml_win_prob if ml_available else None,
        'baseline_to_elo_improvement': elo_a_sim - baseline_a,
        'baseline_to_ml_improvement': (ml_win_prob - baseline_a) if ml_available else None
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='compare all prediction methods')
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

    compare_all_methods(
        args.playerA,
        args.playerB,
        args.surface,
        args.n,
        args.tour
    )
