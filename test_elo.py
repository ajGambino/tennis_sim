"""
test elo system and compare results with/without elo adjustments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.data_loader import load_match_data
from simulation.player_stats import PlayerStatsCalculator
from simulation.elo_system import EloSystem
from simulation.ranking_adjuster import EloAdjuster
from simulation.simulator import run_simulations, analyze_results


def main():
    print("="*80)
    print("TESTING ELO SYSTEM")
    print("="*80)

    # load match data
    print("\nloading match data...")
    match_data = load_match_data(tour='atp')

    # build elo ratings
    print("\nbuilding elo ratings...")
    elo_system = EloSystem(
        k_factor=32.0,
        initial_rating=1500.0,
        surface_specific=True
    )
    elo_system.build_from_matches(match_data)

    # save elo ratings for future use
    elo_system.save_ratings('models/elo_ratings_atp.json')
    print("\nelo ratings saved to models/elo_ratings_atp.json")

    # test players
    player_a = 'Jannik Sinner'
    player_b = 'Gabriel Diallo'
    surface = 'hard'

    # get elo ratings
    elo_a = elo_system.get_rating(player_a, surface)
    elo_b = elo_system.get_rating(player_b, surface)
    elo_diff = elo_a - elo_b
    win_prob_elo = elo_system.get_win_probability(player_a, player_b, surface)

    print(f"\n{'-'*80}")
    print(f"ELO RATINGS ({surface} court)")
    print(f"{'-'*80}")
    print(f"{player_a:<30} {elo_a:>10.0f}")
    print(f"{player_b:<30} {elo_b:>10.0f}")
    print(f"{'Differential':<30} {elo_diff:>10.0f}")
    print(f"{'Expected Win % (Elo)':<30} {win_prob_elo:>9.1%}")
    print(f"{'-'*80}")

    # get player stats
    print("\ncalculating player statistics...")
    stats_calc = PlayerStatsCalculator(match_data)
    stats_a = stats_calc.get_player_stats(player_a, surface)
    stats_b = stats_calc.get_player_stats(player_b, surface)

    print(f"\n{player_a} (base stats):")
    print(f"  Matches: {stats_a.matches_played}")
    print(f"  1st Serve Win %: {stats_a.first_serve_win_pct:.1%}")
    print(f"  2nd Serve Win %: {stats_a.second_serve_win_pct:.1%}")
    print(f"  Return 1st Win %: {stats_a.return_first_win_pct:.1%}")

    print(f"\n{player_b} (base stats):")
    print(f"  Matches: {stats_b.matches_played}")
    print(f"  1st Serve Win %: {stats_b.first_serve_win_pct:.1%}")
    print(f"  2nd Serve Win %: {stats_b.second_serve_win_pct:.1%}")
    print(f"  Return 1st Win %: {stats_b.return_first_win_pct:.1%}")

    # adjust stats using elo
    print("\napplying elo adjustments...")
    elo_adjuster = EloAdjuster(max_adjustment=0.15)
    stats_a_adjusted = elo_adjuster.adjust_stats(stats_a, elo_a, elo_b)
    stats_b_adjusted = elo_adjuster.adjust_stats(stats_b, elo_b, elo_a)

    print(f"\n{player_a} (elo-adjusted stats):")
    print(f"  1st Serve Win %: {stats_a_adjusted.first_serve_win_pct:.1%} "
          f"(+{stats_a_adjusted.first_serve_win_pct - stats_a.first_serve_win_pct:+.1%})")
    print(f"  2nd Serve Win %: {stats_a_adjusted.second_serve_win_pct:.1%} "
          f"(+{stats_a_adjusted.second_serve_win_pct - stats_a.second_serve_win_pct:+.1%})")
    print(f"  Return 1st Win %: {stats_a_adjusted.return_first_win_pct:.1%} "
          f"(+{stats_a_adjusted.return_first_win_pct - stats_a.return_first_win_pct:+.1%})")

    print(f"\n{player_b} (elo-adjusted stats):")
    print(f"  1st Serve Win %: {stats_b_adjusted.first_serve_win_pct:.1%} "
          f"(+{stats_b_adjusted.first_serve_win_pct - stats_b.first_serve_win_pct:+.1%})")
    print(f"  2nd Serve Win %: {stats_b_adjusted.second_serve_win_pct:.1%} "
          f"(+{stats_b_adjusted.second_serve_win_pct - stats_b.second_serve_win_pct:+.1%})")
    print(f"  Return 1st Win %: {stats_b_adjusted.return_first_win_pct:.1%} "
          f"(+{stats_b_adjusted.return_first_win_pct - stats_b.return_first_win_pct:+.1%})")

    # run simulations WITHOUT elo adjustment
    print(f"\n{'='*80}")
    print("BASELINE SIMULATION (No Elo Adjustment)")
    print(f"{'='*80}")
    results_baseline = run_simulations(stats_a, stats_b, n=1000, best_of=3, start_seed=42)
    summary_baseline = analyze_results(results_baseline, player_a, player_b)

    print(f"\n{player_a} Win %: {summary_baseline['a_win_pct']:.1%}")
    print(f"{player_b} Win %: {summary_baseline['b_win_pct']:.1%}")

    # run simulations WITH elo adjustment
    print(f"\n{'='*80}")
    print("ELO-ADJUSTED SIMULATION")
    print(f"{'='*80}")
    results_elo = run_simulations(stats_a_adjusted, stats_b_adjusted, n=1000, best_of=3, start_seed=42)
    summary_elo = analyze_results(results_elo, player_a, player_b)

    print(f"\n{player_a} Win %: {summary_elo['a_win_pct']:.1%}")
    print(f"{player_b} Win %: {summary_elo['b_win_pct']:.1%}")

    # comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"Baseline (no Elo):  {summary_baseline['a_win_pct']:.1%}")
    print(f"Elo-Adjusted:       {summary_elo['a_win_pct']:.1%}")
    print(f"Elo Expected:       {win_prob_elo:.1%}")
    print(f"Improvement:        {summary_elo['a_win_pct'] - summary_baseline['a_win_pct']:+.1%}")
    print(f"{'='*80}\n")

    # test other matchups
    print("\nTesting other matchups...")
    test_matchups = [
        ('Jannik Sinner', 'Carlos Alcaraz'),
        ('Novak Djokovic', 'Carlos Alcaraz'),
        ('Novak Djokovic', 'Rafael Nadal'),
    ]

    for p_a, p_b in test_matchups:
        elo_a = elo_system.get_rating(p_a, surface)
        elo_b = elo_system.get_rating(p_b, surface)
        win_prob = elo_system.get_win_probability(p_a, p_b, surface)

        print(f"\n{p_a} vs {p_b}")
        print(f"  Elo: {elo_a:.0f} vs {elo_b:.0f}")
        print(f"  Expected: {win_prob:.1%} - {1-win_prob:.1%}")


if __name__ == '__main__':
    # ensure models directory exists
    import os
    os.makedirs('models', exist_ok=True)

    main()
