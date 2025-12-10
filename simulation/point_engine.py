"""
point simulation engine

simulates individual tennis points using serve and return probabilities
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from simulation.player_stats import PlayerStats


@dataclass
class PointResult:
    """result of a single point simulation"""

    server_won: bool  # true if server won the point
    was_ace: bool  # true if point ended in ace
    was_double_fault: bool  # true if point ended in double fault
    was_first_serve: bool  # true if first serve went in


class PointSimulator:
    """simulates individual tennis points"""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        initialize point simulator

        args:
            rng: numpy random generator for reproducibility
                 if none, creates new generator
        """
        self.rng = rng if rng is not None else np.random.default_rng()

    def simulate_point(self, server_stats: PlayerStats,
                      returner_stats: PlayerStats) -> PointResult:
        """
        simulate a single point

        logic flow:
        1. check for ace (based on server ace %)
        2. determine if first serve goes in (based on server 1st serve %)
        3. if first serve in:
           - use server 1st serve win % vs returner return 1st win %
        4. if first serve out (second serve):
           - check for double fault (based on server df %)
           - if not df, use server 2nd serve win % vs returner return 2nd win %

        args:
            server_stats: stats for the serving player
            returner_stats: stats for the returning player

        returns:
            pointresult with outcome and details
        """

        # check for ace first
        if self.rng.random() < server_stats.ace_pct:
            return PointResult(
                server_won=True,
                was_ace=True,
                was_double_fault=False,
                was_first_serve=True
            )

        # determine if first serve goes in
        first_serve_in = self.rng.random() < server_stats.first_serve_pct

        if first_serve_in:
            # first serve in play - use blended probability
            # combine server's 1st serve win % and returner's return 1st win %
            server_win_prob = self._blend_probabilities(
                server_stats.first_serve_win_pct,
                1.0 - returner_stats.return_first_win_pct
            )

            server_won = self.rng.random() < server_win_prob

            return PointResult(
                server_won=server_won,
                was_ace=False,
                was_double_fault=False,
                was_first_serve=True
            )

        else:
            # first serve missed - second serve
            # check for double fault
            if self.rng.random() < server_stats.df_pct:
                return PointResult(
                    server_won=False,
                    was_ace=False,
                    was_double_fault=True,
                    was_first_serve=False
                )

            # second serve in play
            server_win_prob = self._blend_probabilities(
                server_stats.second_serve_win_pct,
                1.0 - returner_stats.return_second_win_pct
            )

            server_won = self.rng.random() < server_win_prob

            return PointResult(
                server_won=server_won,
                was_ace=False,
                was_double_fault=False,
                was_first_serve=False
            )

    def _blend_probabilities(self, server_prob: float,
                            returner_complement_prob: float,
                            server_weight: float = 0.6) -> float:
        """
        blend server and returner probabilities

        gives more weight to server stats since they control the point more

        args:
            server_prob: probability server wins (from server's perspective)
            returner_complement_prob: 1 - returner win prob (server win from returner stats)
            server_weight: weight given to server's stats (default 0.6)

        returns:
            blended probability that server wins the point
        """
        blended = (server_weight * server_prob +
                  (1 - server_weight) * returner_complement_prob)

        # clamp to valid probability range
        return np.clip(blended, 0.01, 0.99)
