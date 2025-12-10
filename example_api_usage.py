"""
example programmatic api usage

demonstrates how to use the simulation framework in your own python code
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.data_loader import load_match_data
from simulation.player_stats import PlayerStatsCalculator, PlayerStats
from simulation.match_engine import MatchSimulator
from simulation.simulator import run_simulations, analyze_results, print_analysis


def example_1_basic_simulation():
    """example 1: basic single match simulation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Single Match Simulation")
    print("="*70 + "\n")

    # load data
    match_data = load_match_data(data_dir='data', tour='atp')

    # calculate player stats
    stats_calc = PlayerStatsCalculator(match_data)
    djokovic = stats_calc.get_player_stats('Novak Djokovic', surface='hard')
    alcaraz = stats_calc.get_player_stats('Carlos Alcaraz', surface='hard')

    # run single simulation
    sim = MatchSimulator(djokovic, alcaraz, best_of=3, seed=123)
    result = sim.simulate_match()

    # access results programmatically
    print(f"Winner: {result.winner}")
    print(f"Score: {result.score}")
    print(f"Aces: {result.aces_a} - {result.aces_b}")
    print(f"Double Faults: {result.double_faults_a} - {result.double_faults_b}")
    print(f"Total Points: {result.total_points_won_a} - {result.total_points_won_b}")


def example_2_monte_carlo():
    """example 2: monte carlo simulation with analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Monte Carlo Simulation")
    print("="*70 + "\n")

    # load data
    match_data = load_match_data(data_dir='data', tour='atp')

    # calculate player stats
    stats_calc = PlayerStatsCalculator(match_data)
    djokovic = stats_calc.get_player_stats('Novak Djokovic')
    federer = stats_calc.get_player_stats('Roger Federer')

    # run 1000 simulations
    results_df = run_simulations(djokovic, federer, n=500, best_of=3)

    # analyze results
    summary = analyze_results(results_df, 'Novak Djokovic', 'Roger Federer')

    # access summary statistics
    print(f"Win Probability: {summary['a_win_pct']:.2%}")
    print(f"Average Aces: {summary['mean_aces_a']:.1f}")
    print(f"Most Common Score: {list(summary['score_distribution'].keys())[0]}")


def example_3_custom_stats():
    """example 3: using custom player statistics"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Player Statistics")
    print("="*70 + "\n")

    # create custom player with elite serve
    elite_server = PlayerStats(
        player_name='Elite Server',
        surface='hard',
        matches_played=0,
        first_serve_pct=0.70,      # 70% first serves in
        ace_pct=0.15,              # 15% aces
        df_pct=0.02,               # only 2% double faults
        first_serve_win_pct=0.80,  # 80% first serve points won
        second_serve_win_pct=0.60, # 60% second serve points won
        return_first_win_pct=0.25, # 25% return first serve won
        return_second_win_pct=0.45 # 45% return second serve won
    )

    # create custom player with elite return
    elite_returner = PlayerStats(
        player_name='Elite Returner',
        surface='hard',
        matches_played=0,
        first_serve_pct=0.60,      # 60% first serves in
        ace_pct=0.05,              # 5% aces
        df_pct=0.03,               # 3% double faults
        first_serve_win_pct=0.68,  # 68% first serve points won
        second_serve_win_pct=0.50, # 50% second serve points won
        return_first_win_pct=0.35, # 35% return first serve won (elite!)
        return_second_win_pct=0.55 # 55% return second serve won (elite!)
    )

    # simulate matchup
    results_df = run_simulations(elite_server, elite_returner, n=1000)
    summary = analyze_results(results_df, 'Elite Server', 'Elite Returner')

    print(f"Elite Server Win %: {summary['a_win_pct']:.2%}")
    print(f"Elite Returner Win %: {summary['b_win_pct']:.2%}")


def example_4_surface_comparison():
    """example 4: compare player performance across surfaces"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Surface Comparison")
    print("="*70 + "\n")

    # load data
    match_data = load_match_data(data_dir='data', tour='atp')
    stats_calc = PlayerStatsCalculator(match_data)

    # get stats for same player on different surfaces
    player = 'Novak Djokovic'

    for surface in ['hard', 'clay', 'grass']:
        stats = stats_calc.get_player_stats(player, surface=surface)
        print(f"\n{player} on {surface.upper()}:")
        print(f"  Matches: {stats.matches_played}")
        print(f"  First Serve %: {stats.first_serve_pct:.1%}")
        print(f"  Ace %: {stats.ace_pct:.1%}")
        print(f"  1st Serve Win %: {stats.first_serve_win_pct:.1%}")


def example_5_export_results():
    """example 5: export and analyze results dataframe"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Export and Analyze Results")
    print("="*70 + "\n")

    # load data
    match_data = load_match_data(data_dir='data', tour='atp')
    stats_calc = PlayerStatsCalculator(match_data)

    djokovic = stats_calc.get_player_stats('Novak Djokovic')
    nadal = stats_calc.get_player_stats('Rafael Nadal')

    # run simulations
    results_df = run_simulations(djokovic, nadal, n=100)

    # custom analysis using pandas
    print("\nCustom Analysis:")
    print(f"Total simulations: {len(results_df)}")
    print(f"\nScore distribution:")
    print(results_df['score'].value_counts().head())

    print(f"\nAverage total points by winner:")
    print(results_df.groupby('winner')[['total_points_a', 'total_points_b']].mean())

    # save to csv
    results_df.to_csv('results/custom_analysis.csv', index=False)
    print("\nResults saved to results/custom_analysis.csv")


if __name__ == '__main__':
    # run all examples
    example_1_basic_simulation()
    example_2_monte_carlo()
    example_3_custom_stats()
    example_4_surface_comparison()
    example_5_export_results()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")
