"""
player statistics calculator

computes per-player probability parameters from historical match data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PlayerStats:
    """container for player serve and return statistics"""

    # serve probabilities
    first_serve_pct: float  # probability of first serve going in
    ace_pct: float  # probability of ace on any serve
    df_pct: float  # probability of double fault on second serve
    first_serve_win_pct: float  # probability of winning point when 1st serve in
    second_serve_win_pct: float  # probability of winning point on 2nd serve

    # return probabilities (derived from opponent stats)
    return_first_win_pct: float  # prob of winning point when returning 1st serve
    return_second_win_pct: float  # prob of winning point when returning 2nd serve

    # metadata
    player_name: str
    surface: str
    matches_played: int

    def to_dict(self) -> Dict:
        """convert to dictionary"""
        return {
            'player_name': self.player_name,
            'surface': self.surface,
            'matches_played': self.matches_played,
            'first_serve_pct': self.first_serve_pct,
            'ace_pct': self.ace_pct,
            'df_pct': self.df_pct,
            'first_serve_win_pct': self.first_serve_win_pct,
            'second_serve_win_pct': self.second_serve_win_pct,
            'return_first_win_pct': self.return_first_win_pct,
            'return_second_win_pct': self.return_second_win_pct,
        }


class PlayerStatsCalculator:
    """calculates player statistics from historical match data"""

    # default fallback values (atp tour averages)
    DEFAULT_STATS = {
        'first_serve_pct': 0.62,
        'ace_pct': 0.06,
        'df_pct': 0.04,
        'first_serve_win_pct': 0.72,
        'second_serve_win_pct': 0.53,
        'return_first_win_pct': 0.28,
        'return_second_win_pct': 0.47,
    }

    def __init__(self, match_data: pd.DataFrame, min_matches: int = 10):
        """
        initialize calculator

        args:
            match_data: dataframe of match results from data_loader
            min_matches: minimum matches required for reliable stats
        """
        self.match_data = match_data
        self.min_matches = min_matches
        self._stats_cache = {}

    def get_player_stats(self, player_name: str,
                        surface: Optional[str] = None) -> PlayerStats:
        """
        compute comprehensive stats for a player

        args:
            player_name: player name as it appears in data
            surface: optional surface filter (hard/clay/grass)
                    if none, uses aggregate across all surfaces

        returns:
            playerstats object with all probabilities
        """
        # check cache
        cache_key = (player_name, surface)
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        # get player matches
        player_matches = self._get_player_matches(player_name, surface)

        if len(player_matches) < self.min_matches:
            print(f"warning: {player_name} has only {len(player_matches)} matches "
                  f"on {surface or 'all surfaces'}, using fallback stats")
            stats = self._create_fallback_stats(player_name, surface)
        else:
            stats = self._calculate_stats(player_name, player_matches, surface)

        # cache and return
        self._stats_cache[cache_key] = stats
        return stats

    def _get_player_matches(self, player_name: str,
                           surface: Optional[str]) -> pd.DataFrame:
        """get all matches for a player, optionally filtered by surface"""
        matches = self.match_data[
            (self.match_data['winner_name'] == player_name) |
            (self.match_data['loser_name'] == player_name)
        ].copy()

        if surface:
            matches = matches[matches['surface'] == surface.lower()]

        return matches

    def _calculate_stats(self, player_name: str, matches: pd.DataFrame,
                        surface: Optional[str]) -> PlayerStats:
        """calculate actual stats from match data"""

        # separate wins and losses
        wins = matches[matches['winner_name'] == player_name].copy()
        losses = matches[matches['loser_name'] == player_name].copy()

        # collect serve stats (when player was serving)
        serve_stats = self._aggregate_serve_stats(wins, losses)

        # collect return stats (when player was returning)
        return_stats = self._aggregate_return_stats(wins, losses)

        return PlayerStats(
            player_name=player_name,
            surface=surface or 'all',
            matches_played=len(matches),
            **serve_stats,
            **return_stats
        )

    def _aggregate_serve_stats(self, wins: pd.DataFrame,
                               losses: pd.DataFrame) -> Dict[str, float]:
        """aggregate serve statistics from wins and losses"""

        # combine stats from wins (w_ prefix) and losses (l_ prefix)
        total_svpt = wins['w_svpt'].sum() + losses['l_svpt'].sum()
        total_1st_in = wins['w_1stIn'].sum() + losses['l_1stIn'].sum()
        total_1st_won = wins['w_1stWon'].sum() + losses['l_1stWon'].sum()
        total_2nd_won = wins['w_2ndWon'].sum() + losses['l_2ndWon'].sum()
        total_aces = wins['w_ace'].sum() + losses['l_ace'].sum()
        total_dfs = wins['w_df'].sum() + losses['l_df'].sum()

        # calculate probabilities with smoothing
        first_serve_pct = self._safe_divide(total_1st_in, total_svpt,
                                            self.DEFAULT_STATS['first_serve_pct'])

        ace_pct = self._safe_divide(total_aces, total_svpt,
                                    self.DEFAULT_STATS['ace_pct'])

        # df rate is per second serve attempt
        second_serves = total_svpt - total_1st_in
        df_pct = self._safe_divide(total_dfs, second_serves,
                                   self.DEFAULT_STATS['df_pct'])

        first_serve_win_pct = self._safe_divide(total_1st_won, total_1st_in,
                                                self.DEFAULT_STATS['first_serve_win_pct'])

        # second serve points is total minus first serve points
        second_serve_win_pct = self._safe_divide(total_2nd_won, second_serves,
                                                 self.DEFAULT_STATS['second_serve_win_pct'])

        return {
            'first_serve_pct': first_serve_pct,
            'ace_pct': ace_pct,
            'df_pct': df_pct,
            'first_serve_win_pct': first_serve_win_pct,
            'second_serve_win_pct': second_serve_win_pct,
        }

    def _aggregate_return_stats(self, wins: pd.DataFrame,
                                losses: pd.DataFrame) -> Dict[str, float]:
        """
        aggregate return statistics

        return win % is the complement of opponent's serve win %
        """

        # when player won, opponent was loser (l_ prefix for opponent)
        # when player lost, opponent was winner (w_ prefix for opponent)

        # opponent first serve stats
        opp_1st_in = wins['l_1stIn'].sum() + losses['w_1stIn'].sum()
        opp_1st_won = wins['l_1stWon'].sum() + losses['w_1stWon'].sum()

        # opponent second serve stats
        opp_svpt = wins['l_svpt'].sum() + losses['w_svpt'].sum()
        opp_2nd_pts = opp_svpt - opp_1st_in
        opp_2nd_won = wins['l_2ndWon'].sum() + losses['w_2ndWon'].sum()

        # return win % = 1 - opponent serve win %
        opp_1st_win_pct = self._safe_divide(opp_1st_won, opp_1st_in, 0.72)
        opp_2nd_win_pct = self._safe_divide(opp_2nd_won, opp_2nd_pts, 0.53)

        return {
            'return_first_win_pct': 1.0 - opp_1st_win_pct,
            'return_second_win_pct': 1.0 - opp_2nd_win_pct,
        }

    def _safe_divide(self, numerator: float, denominator: float,
                    default: float) -> float:
        """
        safe division with fallback

        uses additive smoothing (laplace) to blend actual stats with defaults
        """
        if denominator == 0 or pd.isna(numerator) or pd.isna(denominator):
            return default

        # laplace smoothing: add pseudo-counts based on defaults
        alpha = 5  # smoothing parameter (higher = more weight to default)
        smoothed = (numerator + alpha * default) / (denominator + alpha)

        # clamp to reasonable bounds
        return np.clip(smoothed, 0.01, 0.99)

    def _create_fallback_stats(self, player_name: str,
                               surface: Optional[str]) -> PlayerStats:
        """create stats using default values when insufficient data"""

        return PlayerStats(
            player_name=player_name,
            surface=surface or 'all',
            matches_played=0,
            **self.DEFAULT_STATS
        )

    def get_matchup_stats(self, player_a: str, player_b: str,
                         surface: Optional[str] = None) -> Tuple[PlayerStats, PlayerStats]:
        """
        get stats for both players in a matchup

        args:
            player_a: first player name
            player_b: second player name
            surface: optional surface filter

        returns:
            tuple of (player_a_stats, player_b_stats)
        """
        stats_a = self.get_player_stats(player_a, surface)
        stats_b = self.get_player_stats(player_b, surface)

        return stats_a, stats_b

    def print_player_summary(self, player_name: str,
                            surface: Optional[str] = None):
        """print formatted summary of player stats"""

        stats = self.get_player_stats(player_name, surface)

        print(f"\n{'='*60}")
        print(f"Player: {stats.player_name}")
        print(f"Surface: {stats.surface}")
        print(f"Matches: {stats.matches_played}")
        print(f"{'='*60}")
        print("\nServe Statistics:")
        print(f"  First Serve %:        {stats.first_serve_pct:.1%}")
        print(f"  Ace %:                {stats.ace_pct:.1%}")
        print(f"  Double Fault %:       {stats.df_pct:.1%}")
        print(f"  1st Serve Win %:      {stats.first_serve_win_pct:.1%}")
        print(f"  2nd Serve Win %:      {stats.second_serve_win_pct:.1%}")
        print("\nReturn Statistics:")
        print(f"  Return 1st Serve Win %: {stats.return_first_win_pct:.1%}")
        print(f"  Return 2nd Serve Win %: {stats.return_second_win_pct:.1%}")
        print(f"{'='*60}\n")
