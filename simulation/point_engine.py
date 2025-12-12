"""
point simulation engine

simulates individual tennis points using serve and return probabilities
supports both point-level (legacy) and shot-level (new) simulation
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

    # shot-level extensions (optional, only populated if use_shot_simulation=true)
    rally: Optional['Rally'] = None  # full rally object with shot sequence
    rally_length: int = 0  # number of shots in rally
    serve_placement: Optional['ServePlacement'] = None  # where serve was placed
    point_ending_shot_type: Optional['ShotType'] = None  # type of shot that ended point

    # score context (optional, only populated if track_point_history=true)
    server_name: Optional[str] = None  # name of server for this point
    returner_name: Optional[str] = None  # name of returner for this point
    game_score: Optional[str] = None  # game score before this point (e.g., "30-15", "Deuce")
    games_score: Optional[str] = None  # games score in current set (e.g., "3-2")
    set_number: Optional[int] = None  # which set this point is in (1, 2, 3, ...)
    sets_score: Optional[str] = None  # sets won by each player (e.g., "1-0")


class PointSimulator:
    """simulates individual tennis points"""

    def __init__(self, rng: Optional[np.random.Generator] = None,
                 use_shot_simulation: bool = False,
                 serve_model: Optional['ServeModel'] = None,
                 return_model: Optional['ReturnModel'] = None,
                 rally_model: Optional['RallyModel'] = None):
        """
        initialize point simulator

        args:
            rng: numpy random generator for reproducibility
                 if none, creates new generator
            use_shot_simulation: whether to use shot-by-shot simulation
            serve_model: serve placement model (required if use_shot_simulation=true)
            return_model: return quality model (required if use_shot_simulation=true)
            rally_model: rally simulation model (required if use_shot_simulation=true)
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.use_shot_simulation = use_shot_simulation
        self.serve_model = serve_model
        self.return_model = return_model
        self.rally_model = rally_model

        # if shot simulation enabled, ensure models are provided
        if self.use_shot_simulation:
            if not all([self.serve_model, self.return_model, self.rally_model]):
                # lazy import to avoid circular dependencies
                from simulation.serve_model import ServeModel
                from simulation.return_model import ReturnModel
                from simulation.rally_model import RallyModel

                if self.serve_model is None:
                    self.serve_model = ServeModel(rng=self.rng)
                if self.return_model is None:
                    self.return_model = ReturnModel(rng=self.rng)
                if self.rally_model is None:
                    self.rally_model = RallyModel(rng=self.rng)

    def simulate_point(self, server_stats: PlayerStats,
                      returner_stats: PlayerStats,
                      server_name: Optional[str] = None,
                      returner_name: Optional[str] = None,
                      surface: str = 'hard') -> PointResult:
        """
        simulate a single point

        supports both point-level (legacy) and shot-level simulation

        logic flow (point-level):
        1. check for ace (based on server ace %)
        2. determine if first serve goes in (based on server 1st serve %)
        3. if first serve in:
           - use server 1st serve win % vs returner return 1st win %
        4. if first serve out (second serve):
           - check for double fault (based on server df %)
           - if not df, use server 2nd serve win % vs returner return 2nd win %

        logic flow (shot-level):
        1. sample serve placement (wide/t/body)
        2. determine serve outcome (ace/fault/in-play)
        3. if in-play, simulate return shot
        4. if return in-play, simulate rally exchange shot-by-shot

        args:
            server_stats: stats for the serving player
            returner_stats: stats for the returning player
            server_name: server's name (for player-specific shot patterns)
            returner_name: returner's name (for player-specific shot patterns)
            surface: court surface ('hard', 'clay', 'grass')

        returns:
            pointresult with outcome and details
        """
        # route to appropriate simulation method
        if self.use_shot_simulation:
            return self._simulate_point_shot_level(
                server_stats, returner_stats, server_name, returner_name, surface
            )
        else:
            return self._simulate_point_legacy(server_stats, returner_stats)

    def _simulate_point_legacy(self, server_stats: PlayerStats,
                               returner_stats: PlayerStats) -> PointResult:
        """
        legacy point-level simulation (original implementation)

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

    def _simulate_point_shot_level(self, server_stats: PlayerStats,
                                   returner_stats: PlayerStats,
                                   server_name: Optional[str],
                                   returner_name: Optional[str],
                                   surface: str) -> PointResult:
        """
        shot-by-shot simulation using serve/return/rally models

        args:
            server_stats: server's stats
            returner_stats: returner's stats
            server_name: server's name
            returner_name: returner's name
            surface: court surface

        returns:
            pointresult with shot-level details
        """
        # simulate first serve
        first_serve = self.serve_model.simulate_first_serve(
            player_stats=server_stats,
            player_name=server_name,
            surface=surface
        )

        # check if first serve was a fault
        if first_serve.is_fault:
            # simulate second serve
            second_serve = self.serve_model.simulate_second_serve(
                player_stats=server_stats,
                player_name=server_name,
                surface=surface
            )

            # check for double fault
            if second_serve.is_fault:
                return PointResult(
                    server_won=False,
                    was_ace=False,
                    was_double_fault=True,
                    was_first_serve=False,
                    rally=None,
                    rally_length=0,
                    serve_placement=None,
                    point_ending_shot_type=None
                )

            # second serve in play
            serve_outcome = second_serve
            is_second_serve = True
        else:
            # first serve in play
            serve_outcome = first_serve
            is_second_serve = False

        # check for ace
        if serve_outcome.is_ace:
            # create minimal rally for ace
            from simulation.shot import Rally
            rally = Rally(
                shots=[],
                serve_outcome=serve_outcome,
                is_second_serve=is_second_serve,
                winner='server'
            )

            return PointResult(
                server_won=True,
                was_ace=True,
                was_double_fault=False,
                was_first_serve=(not is_second_serve),
                rally=rally,
                rally_length=1,
                serve_placement=serve_outcome.placement,
                point_ending_shot_type=None
            )

        # serve is in play - simulate return
        return_shot = self.return_model.simulate_return(
            returner_stats=returner_stats,
            serve_placement=serve_outcome.placement,
            is_second_serve=is_second_serve,
            surface=surface
        )

        # check if return ended the point
        if return_shot.is_point_ending():
            # determine winner
            if return_shot.outcome.value == 'winner':
                winner = 'returner'
            else:  # error
                winner = 'server'

            # create rally with just serve and return
            from simulation.shot import Rally, Shot, ShotType, ShotDirection, ShotOutcome, CourtPosition
            serve_shot = Shot(
                shot_number=1,
                player='server',
                shot_type=ShotType.FOREHAND,
                direction=ShotDirection.DOWN_THE_MIDDLE,
                outcome=ShotOutcome.IN_PLAY,
                position=CourtPosition.BASELINE,
                is_approach=False
            )

            rally = Rally(
                shots=[serve_shot, return_shot],
                serve_outcome=serve_outcome,
                is_second_serve=is_second_serve,
                winner=winner
            )

            return PointResult(
                server_won=(winner == 'server'),
                was_ace=False,
                was_double_fault=False,
                was_first_serve=(not is_second_serve),
                rally=rally,
                rally_length=2,
                serve_placement=serve_outcome.placement,
                point_ending_shot_type=return_shot.shot_type
            )

        # return is in play - simulate rally exchange
        rally_shots = self.rally_model.simulate_rally_exchange(
            server_stats=server_stats,
            returner_stats=returner_stats,
            return_shot=return_shot,
            serve_placement=serve_outcome.placement,
            is_second_serve=is_second_serve,
            surface=surface,
            server_name=server_name,
            returner_name=returner_name
        )

        # create rally object
        rally = self.rally_model.create_rally_from_shots(
            shots=rally_shots,
            serve_outcome=serve_outcome,
            is_second_serve=is_second_serve
        )

        # extract point ending shot
        point_ending_shot = rally.point_ending_shot

        return PointResult(
            server_won=(rally.winner == 'server'),
            was_ace=False,
            was_double_fault=False,
            was_first_serve=(not is_second_serve),
            rally=rally,
            rally_length=rally.rally_length,
            serve_placement=serve_outcome.placement,
            point_ending_shot_type=point_ending_shot.shot_type if point_ending_shot else None
        )
