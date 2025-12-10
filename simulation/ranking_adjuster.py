"""
ranking-based probability adjuster

adjusts serve/return probabilities based on player ranking or elo differential
"""

import numpy as np
from typing import Optional
from simulation.player_stats import PlayerStats
from simulation.elo_system import calculate_skill_adjustment


class RankingAdjuster:
    """adjusts match probabilities based on ranking differential"""

    def __init__(self, base_advantage: float = 0.05):
        """
        initialize adjuster

        args:
            base_advantage: probability boost per 100 ranking points difference
        """
        self.base_advantage = base_advantage

    def adjust_stats(self, player_stats: PlayerStats, player_rank: int,
                    opponent_rank: int) -> PlayerStats:
        """
        create adjusted stats based on ranking differential

        args:
            player_stats: original player statistics
            player_rank: player's atp/wta ranking (lower is better)
            opponent_rank: opponent's ranking

        returns:
            new playerstats with adjusted probabilities
        """
        # calculate ranking differential (positive = player is higher ranked)
        rank_diff = opponent_rank - player_rank

        # convert to probability adjustment
        # higher ranked player (lower number) gets boost
        adjustment = self._calculate_adjustment(rank_diff)

        # apply adjustment to serve stats
        adjusted_stats = PlayerStats(
            player_name=player_stats.player_name,
            surface=player_stats.surface,
            matches_played=player_stats.matches_played,

            # boost serve stats for higher-ranked player
            first_serve_pct=self._adjust_prob(player_stats.first_serve_pct, adjustment * 0.3),
            ace_pct=self._adjust_prob(player_stats.ace_pct, adjustment * 0.5),
            df_pct=self._adjust_prob(player_stats.df_pct, -adjustment * 0.3),
            first_serve_win_pct=self._adjust_prob(player_stats.first_serve_win_pct, adjustment),
            second_serve_win_pct=self._adjust_prob(player_stats.second_serve_win_pct, adjustment),

            # boost return stats for higher-ranked player
            return_first_win_pct=self._adjust_prob(player_stats.return_first_win_pct, adjustment),
            return_second_win_pct=self._adjust_prob(player_stats.return_second_win_pct, adjustment),
        )

        return adjusted_stats

    def _calculate_adjustment(self, rank_diff: int) -> float:
        """
        calculate probability adjustment from ranking differential

        args:
            rank_diff: opponent_rank - player_rank (positive = player is better)

        returns:
            probability adjustment (-0.15 to +0.15)
        """
        # scale adjustment based on rank difference
        # every 50 ranking points = ~2.5% advantage
        adjustment = (rank_diff / 50.0) * self.base_advantage

        # cap adjustment to prevent extreme values
        return np.clip(adjustment, -0.15, 0.15)

    def _adjust_prob(self, original: float, adjustment: float) -> float:
        """
        apply adjustment to probability, keeping within valid bounds

        args:
            original: original probability
            adjustment: adjustment to apply

        returns:
            adjusted probability clamped to [0.01, 0.99]
        """
        adjusted = original + adjustment
        return np.clip(adjusted, 0.01, 0.99)


class EloAdjuster:
    """adjusts match probabilities based on elo differential"""

    def __init__(self, max_adjustment: float = 0.15):
        """
        initialize elo adjuster

        args:
            max_adjustment: maximum probability adjustment to apply
        """
        self.max_adjustment = max_adjustment

    def adjust_stats(self, player_stats: PlayerStats, player_elo: float,
                    opponent_elo: float) -> PlayerStats:
        """
        create adjusted stats based on elo differential

        args:
            player_stats: original player statistics
            player_elo: player's elo rating
            opponent_elo: opponent's elo rating

        returns:
            new playerstats with adjusted probabilities
        """
        # calculate elo differential
        elo_diff = player_elo - opponent_elo

        # convert to probability adjustment using sigmoid-like scaling
        adjustment = calculate_skill_adjustment(elo_diff, self.max_adjustment)

        # apply adjustment to all probabilities
        adjusted_stats = PlayerStats(
            player_name=player_stats.player_name,
            surface=player_stats.surface,
            matches_played=player_stats.matches_played,

            # boost serve stats for higher-rated player
            first_serve_pct=self._adjust_prob(player_stats.first_serve_pct, adjustment * 0.3),
            ace_pct=self._adjust_prob(player_stats.ace_pct, adjustment * 0.5),
            df_pct=self._adjust_prob(player_stats.df_pct, -adjustment * 0.4),
            first_serve_win_pct=self._adjust_prob(player_stats.first_serve_win_pct, adjustment),
            second_serve_win_pct=self._adjust_prob(player_stats.second_serve_win_pct, adjustment * 1.2),

            # boost return stats for higher-rated player
            return_first_win_pct=self._adjust_prob(player_stats.return_first_win_pct, adjustment),
            return_second_win_pct=self._adjust_prob(player_stats.return_second_win_pct, adjustment * 1.2),
        )

        return adjusted_stats

    def _adjust_prob(self, original: float, adjustment: float) -> float:
        """
        apply adjustment to probability, keeping within valid bounds

        args:
            original: original probability
            adjustment: adjustment to apply

        returns:
            adjusted probability clamped to [0.01, 0.99]
        """
        adjusted = original + adjustment
        return np.clip(adjusted, 0.01, 0.99)
