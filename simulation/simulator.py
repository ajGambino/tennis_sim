"""
bulk simulation runner

runs monte carlo simulations and aggregates results
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from simulation.player_stats import PlayerStats
from simulation.match_engine import MatchSimulator, MatchStats


def run_simulations(player_a_stats: PlayerStats,
                   player_b_stats: PlayerStats,
                   n: int = 1000,
                   best_of: int = 3,
                   start_seed: int = None) -> pd.DataFrame:
    """
    run monte carlo match simulations

    args:
        player_a_stats: statistics for player a
        player_b_stats: statistics for player b
        n: number of simulations to run
        best_of: 3 or 5 sets
        start_seed: starting seed (will increment for each simulation)

    returns:
        dataframe with one row per simulation
    """
    results = []

    if start_seed is None:
        start_seed = np.random.randint(0, 2**31)

    print(f"running {n} simulations...")
    print(f"{player_a_stats.player_name} vs {player_b_stats.player_name}")
    print(f"best of {best_of}, starting seed: {start_seed}\n")

    # run simulations
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"  completed {i + 1}/{n} simulations...")

        seed = start_seed + i

        # create and run simulation
        sim = MatchSimulator(player_a_stats, player_b_stats, best_of, seed)
        match_stats = sim.simulate_match()

        # store result
        results.append(match_stats.to_dict())

    print(f"completed all {n} simulations\n")

    # convert to dataframe
    df = pd.DataFrame(results)
    return df


def analyze_results(df: pd.DataFrame, player_a_name: str,
                   player_b_name: str) -> Dict:
    """
    analyze simulation results and compute summary statistics

    args:
        df: dataframe from run_simulations
        player_a_name: name of player a
        player_b_name: name of player b

    returns:
        dictionary with summary statistics
    """
    total_sims = len(df)

    # win probability
    a_wins = (df['winner'] == player_a_name).sum()
    b_wins = (df['winner'] == player_b_name).sum()

    a_win_pct = a_wins / total_sims
    b_win_pct = b_wins / total_sims

    # score distributions
    score_dist = df['score'].value_counts(normalize=True).to_dict()

    # set score distributions
    sets_a_dist = df['sets_a'].value_counts(normalize=True).sort_index().to_dict()
    sets_b_dist = df['sets_b'].value_counts(normalize=True).sort_index().to_dict()

    # statistical summaries
    summary = {
        'total_simulations': total_sims,
        'player_a': player_a_name,
        'player_b': player_b_name,
        'a_wins': a_wins,
        'b_wins': b_wins,
        'a_win_pct': a_win_pct,
        'b_win_pct': b_win_pct,

        # mean statistics for player a
        'mean_aces_a': df['aces_a'].mean(),
        'median_aces_a': df['aces_a'].median(),
        'mean_df_a': df['df_a'].mean(),
        'median_df_a': df['df_a'].median(),
        'mean_first_serve_pct_a': df['first_serve_pct_a'].mean(),
        'mean_total_points_a': df['total_points_a'].mean(),

        # mean statistics for player b
        'mean_aces_b': df['aces_b'].mean(),
        'median_aces_b': df['aces_b'].median(),
        'mean_df_b': df['df_b'].mean(),
        'median_df_b': df['df_b'].median(),
        'mean_first_serve_pct_b': df['first_serve_pct_b'].mean(),
        'mean_total_points_b': df['total_points_b'].mean(),

        # distributions
        'score_distribution': score_dist,
        'sets_a_distribution': sets_a_dist,
        'sets_b_distribution': sets_b_dist,
    }

    return summary


def print_analysis(summary: Dict):
    """print formatted analysis summary"""

    print(f"{'='*70}")
    print(f"MONTE CARLO SIMULATION RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal Simulations: {summary['total_simulations']:,}")
    print(f"\n{summary['player_a']} vs {summary['player_b']}")
    print(f"{'-'*70}")

    # win probabilities
    print(f"\nWin Probability:")
    print(f"  {summary['player_a']:<30} {summary['a_win_pct']:>6.2%} ({summary['a_wins']:,} wins)")
    print(f"  {summary['player_b']:<30} {summary['b_win_pct']:>6.2%} ({summary['b_wins']:,} wins)")

    # score distribution
    print(f"\nMost Common Scores:")
    sorted_scores = sorted(summary['score_distribution'].items(),
                          key=lambda x: x[1], reverse=True)[:5]
    for score, pct in sorted_scores:
        print(f"  {score:<30} {pct:>6.2%}")

    # sets won distribution
    print(f"\nSets Won Distribution ({summary['player_a']}):")
    for sets, pct in sorted(summary['sets_a_distribution'].items()):
        print(f"  {sets} sets: {pct:>6.2%}")

    # average statistics
    print(f"\nAverage Match Statistics:")
    print(f"{'Statistic':<30} {summary['player_a'][:15]:>15} {summary['player_b'][:15]:>15}")
    print(f"{'-'*70}")
    print(f"{'Aces (mean)':<30} {summary['mean_aces_a']:>15.1f} {summary['mean_aces_b']:>15.1f}")
    print(f"{'Aces (median)':<30} {summary['median_aces_a']:>15.1f} {summary['median_aces_b']:>15.1f}")
    print(f"{'Double Faults (mean)':<30} {summary['mean_df_a']:>15.1f} {summary['mean_df_b']:>15.1f}")
    print(f"{'Double Faults (median)':<30} {summary['median_df_a']:>15.1f} {summary['median_df_b']:>15.1f}")
    print(f"{'First Serve % (mean)':<30} {summary['mean_first_serve_pct_a']:>14.1%} "
          f"{summary['mean_first_serve_pct_b']:>14.1%}")
    print(f"{'Total Points (mean)':<30} {summary['mean_total_points_a']:>15.1f} "
          f"{summary['mean_total_points_b']:>15.1f}")

    print(f"{'='*70}\n")


def save_results(df: pd.DataFrame, filename: str):
    """
    save simulation results to csv

    args:
        df: dataframe from run_simulations
        filename: output filename (will be saved to results/ directory)
    """
    import os

    # ensure results directory exists
    os.makedirs('results', exist_ok=True)

    filepath = os.path.join('results', filename)
    df.to_csv(filepath, index=False)

    print(f"results saved to {filepath}")
