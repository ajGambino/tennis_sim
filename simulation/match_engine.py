"""
match simulation engine

implements tennis game, set, and match logic with full rules including tiebreaks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from simulation.player_stats import PlayerStats
from simulation.point_engine import PointSimulator, PointResult


@dataclass
class GameResult:
    """result of a single game"""

    server_won: bool
    server_points: int  # points won by server in this game
    returner_points: int  # points won by returner in this game
    aces: int
    double_faults: int
    first_serves_in: int
    first_serves_total: int


@dataclass
class SetResult:
    """result of a single set"""

    player_a_games: int
    player_b_games: int
    was_tiebreak: bool


@dataclass
class MatchStats:
    """comprehensive match statistics"""

    # player names
    player_a_name: str
    player_b_name: str

    # match outcome
    winner: str  # player name
    sets_won_a: int
    sets_won_b: int
    score: str  # formatted score string like "6-4, 3-6, 7-6"

    # set-by-set scores
    set_scores: List[Tuple[int, int]]  # [(6,4), (3,6), (7,6)]

    # player a stats
    aces_a: int = 0
    double_faults_a: int = 0
    first_serves_in_a: int = 0
    first_serves_total_a: int = 0
    first_serve_points_won_a: int = 0
    first_serve_points_total_a: int = 0
    second_serve_points_won_a: int = 0
    second_serve_points_total_a: int = 0
    break_points_won_a: int = 0
    break_points_total_a: int = 0
    service_games_won_a: int = 0
    service_games_total_a: int = 0
    total_points_won_a: int = 0

    # player b stats
    aces_b: int = 0
    double_faults_b: int = 0
    first_serves_in_b: int = 0
    first_serves_total_b: int = 0
    first_serve_points_won_b: int = 0
    first_serve_points_total_b: int = 0
    second_serve_points_won_b: int = 0
    second_serve_points_total_b: int = 0
    break_points_won_b: int = 0
    break_points_total_b: int = 0
    service_games_won_b: int = 0
    service_games_total_b: int = 0
    total_points_won_b: int = 0

    # total points
    total_points: int = 0

    # random seed
    random_seed: int = 0

    # point-by-point history (optional, for detailed analysis)
    point_history: List['PointResult'] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """convert to dictionary for easy export"""
        return {
            'player_a': self.player_a_name,
            'player_b': self.player_b_name,
            'winner': self.winner,
            'sets_a': self.sets_won_a,
            'sets_b': self.sets_won_b,
            'score': self.score,
            'aces_a': self.aces_a,
            'aces_b': self.aces_b,
            'df_a': self.double_faults_a,
            'df_b': self.double_faults_b,
            'first_serve_pct_a': self.first_serves_in_a / max(1, self.first_serves_total_a),
            'first_serve_pct_b': self.first_serves_in_b / max(1, self.first_serves_total_b),
            'first_serve_won_a': self.first_serve_points_won_a,
            'first_serve_won_b': self.first_serve_points_won_b,
            'second_serve_won_a': self.second_serve_points_won_a,
            'second_serve_won_b': self.second_serve_points_won_b,
            'bp_won_a': self.break_points_won_a,
            'bp_won_b': self.break_points_won_b,
            'bp_total_a': self.break_points_total_a,
            'bp_total_b': self.break_points_total_b,
            'total_points_a': self.total_points_won_a,
            'total_points_b': self.total_points_won_b,
            'seed': self.random_seed
        }


class MatchSimulator:
    """simulates complete tennis matches"""

    # tennis scoring
    SCORE_NAMES = ['0', '15', '30', '40', 'AD']

    def __init__(self, player_a_stats: PlayerStats, player_b_stats: PlayerStats,
                 best_of: int = 3, seed: int = None,
                 point_simulator: Optional[PointSimulator] = None,
                 track_point_history: bool = False,
                 surface: str = 'hard'):
        """
        initialize match simulator

        args:
            player_a_stats: statistics for player a
            player_b_stats: statistics for player b
            best_of: 3 or 5 sets
            seed: random seed for reproducibility
            point_simulator: optional custom point simulator (for shot-level simulation)
                           if none, creates default point-level simulator
            track_point_history: whether to store point-by-point details (default false)
            surface: court surface for shot-level simulation (default 'hard')
        """
        self.player_a_stats = player_a_stats
        self.player_b_stats = player_b_stats
        self.best_of = best_of
        self.sets_to_win = (best_of + 1) // 2
        self.surface = surface

        # initialize random generator
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # initialize point simulator (use custom if provided, otherwise create default)
        if point_simulator is not None:
            self.point_sim = point_simulator
        else:
            self.point_sim = PointSimulator(self.rng)

        # match tracking
        self.match_stats = None
        self.track_point_history = track_point_history

    def simulate_match(self) -> MatchStats:
        """
        simulate complete match

        returns:
            matchstats object with full details
        """
        # initialize stats
        self.match_stats = MatchStats(
            player_a_name=self.player_a_stats.player_name,
            player_b_name=self.player_b_stats.player_name,
            winner='',
            sets_won_a=0,
            sets_won_b=0,
            score='',
            set_scores=[],
            random_seed=self.seed
        )

        # track sets
        sets_a = 0
        sets_b = 0
        set_scores = []

        # determine who serves first (random)
        a_serves_first = self.rng.random() < 0.5

        # play sets until someone wins
        while sets_a < self.sets_to_win and sets_b < self.sets_to_win:
            # alternate who serves first each set
            server_is_a = a_serves_first if len(set_scores) % 2 == 0 else not a_serves_first

            # current set number (1-based)
            current_set = len(set_scores) + 1

            set_result = self._simulate_set(server_is_a, sets_a, sets_b, current_set)

            # update set counts
            if set_result.player_a_games > set_result.player_b_games:
                sets_a += 1
            else:
                sets_b += 1

            set_scores.append((set_result.player_a_games, set_result.player_b_games))

        # finalize match stats
        self.match_stats.sets_won_a = sets_a
        self.match_stats.sets_won_b = sets_b
        self.match_stats.set_scores = set_scores
        self.match_stats.winner = (self.player_a_stats.player_name if sets_a > sets_b
                                   else self.player_b_stats.player_name)
        self.match_stats.score = self._format_score(set_scores)

        return self.match_stats

    def _simulate_set(self, a_serves_first: bool, sets_a: int = 0,
                      sets_b: int = 0, set_number: int = 1) -> SetResult:
        """
        simulate a single set

        args:
            a_serves_first: true if player a serves first in this set
            sets_a: current sets won by player a (for score tracking)
            sets_b: current sets won by player b (for score tracking)
            set_number: current set number (for score tracking)

        returns:
            setresult with games won by each player
        """
        games_a = 0
        games_b = 0
        game_num = 0

        # play games until set is decided
        while True:
            # determine server for this game
            server_is_a = (game_num % 2 == 0) == a_serves_first

            # simulate game
            game_result = self._simulate_game(server_is_a, games_a, games_b,
                                             sets_a, sets_b, set_number)

            # update game count
            if game_result.server_won:
                if server_is_a:
                    games_a += 1
                else:
                    games_b += 1
            else:
                if server_is_a:
                    games_b += 1
                else:
                    games_a += 1

            game_num += 1

            # check for set winner
            # win by 2 games with at least 6 games
            if games_a >= 6 or games_b >= 6:
                if abs(games_a - games_b) >= 2:
                    return SetResult(games_a, games_b, was_tiebreak=False)

            # tiebreak at 6-6
            if games_a == 6 and games_b == 6:
                tiebreak_winner_is_a = self._simulate_tiebreak(a_serves_first, game_num,
                                                               sets_a, sets_b, set_number)
                if tiebreak_winner_is_a:
                    games_a = 7
                else:
                    games_b = 7
                return SetResult(games_a, games_b, was_tiebreak=True)

    def _simulate_game(self, server_is_a: bool, games_a: int,
                      games_b: int, sets_a: int = 0, sets_b: int = 0,
                      set_number: int = 1) -> GameResult:
        """
        simulate a single game with advantage scoring

        args:
            server_is_a: true if player a is serving
            games_a: current games won by a (for break point tracking)
            games_b: current games won by b (for break point tracking)
            sets_a: current sets won by a (for score tracking)
            sets_b: current sets won by b (for score tracking)
            set_number: current set number (for score tracking)

        returns:
            gameresult with winner and stats
        """
        server_stats = self.player_a_stats if server_is_a else self.player_b_stats
        returner_stats = self.player_b_stats if server_is_a else self.player_a_stats

        server_points = 0
        returner_points = 0

        # track stats for this game
        aces = 0
        dfs = 0
        first_in = 0
        first_total = 0
        first_won = 0
        second_won = 0

        while True:
            # simulate point
            server_name = self.player_a_stats.player_name if server_is_a else self.player_b_stats.player_name
            returner_name = self.player_b_stats.player_name if server_is_a else self.player_a_stats.player_name

            point = self.point_sim.simulate_point(
                server_stats, returner_stats,
                server_name=server_name,
                returner_name=returner_name,
                surface=self.surface
            )

            # update game score first
            if point.server_won:
                server_points += 1
            else:
                returner_points += 1

            # add score context if tracking enabled (AFTER updating points)
            if self.track_point_history and self.match_stats:
                # populate score context fields with score AFTER this point
                point.server_name = server_name
                point.returner_name = returner_name
                point.game_score = self._format_game_score(server_points, returner_points)
                point.games_score = f"{games_a}-{games_b}"
                point.set_number = set_number
                point.sets_score = f"{sets_a}-{sets_b}"

                self.match_stats.point_history.append(point)

            # update stats
            if point.was_ace:
                aces += 1
            if point.was_double_fault:
                dfs += 1

            first_total += 1
            if point.was_first_serve:
                first_in += 1
                if point.server_won:
                    first_won += 1
            else:
                if point.server_won and not point.was_double_fault:
                    second_won += 1

            # check for game over (standard scoring)
            if server_points >= 4 or returner_points >= 4:
                if server_points >= returner_points + 2:
                    # server won game
                    self._update_match_stats(server_is_a, True, aces, dfs,
                                            first_in, first_total, first_won, second_won,
                                            games_a, games_b)
                    return GameResult(True, server_points, returner_points,
                                    aces, dfs, first_in, first_total)

                elif returner_points >= server_points + 2:
                    # returner broke serve
                    self._update_match_stats(server_is_a, False, aces, dfs,
                                            first_in, first_total, first_won, second_won,
                                            games_a, games_b)
                    return GameResult(False, server_points, returner_points,
                                    aces, dfs, first_in, first_total)

            # check for break point (40-30 or deuce advantage to returner)
            if returner_points >= 3 and returner_points > server_points:
                self._track_break_point(server_is_a, point.server_won)

    def _simulate_tiebreak(self, a_served_first_in_set: bool,
                          game_num: int, sets_a: int = 0, sets_b: int = 0,
                          set_number: int = 1) -> bool:
        """
        simulate tiebreak (first to 7, win by 2)

        args:
            a_served_first_in_set: who served first in the set
            game_num: current game number (for server rotation)
            sets_a: current sets won by a (for score tracking)
            sets_b: current sets won by b (for score tracking)
            set_number: current set number (for score tracking)

        returns:
            true if player a won tiebreak
        """
        points_a = 0
        points_b = 0
        total_points = 0

        # determine who serves first point of tiebreak
        # player who would serve next regular game serves first
        a_serves_first = (game_num % 2 == 0) == a_served_first_in_set

        while True:
            # server alternates every 2 points (first point is special)
            if total_points == 0:
                server_is_a = a_serves_first
            elif total_points % 2 == 1:
                server_is_a = a_serves_first
            else:
                server_is_a = not a_serves_first

            # simulate point
            server_stats = self.player_a_stats if server_is_a else self.player_b_stats
            returner_stats = self.player_b_stats if server_is_a else self.player_a_stats
            server_name = self.player_a_stats.player_name if server_is_a else self.player_b_stats.player_name
            returner_name = self.player_b_stats.player_name if server_is_a else self.player_a_stats.player_name

            point = self.point_sim.simulate_point(
                server_stats, returner_stats,
                server_name=server_name,
                returner_name=returner_name,
                surface=self.surface
            )

            # update score first
            if point.server_won:
                if server_is_a:
                    points_a += 1
                else:
                    points_b += 1
            else:
                if server_is_a:
                    points_b += 1
                else:
                    points_a += 1

            total_points += 1

            # add score context if tracking enabled (AFTER updating points)
            if self.track_point_history and self.match_stats:
                # populate score context fields for tiebreak with score AFTER this point
                point.server_name = server_name
                point.returner_name = returner_name
                point.game_score = f"TB {points_a}-{points_b}"  # tiebreak score after point
                point.games_score = "6-6"  # always 6-6 in tiebreak
                point.set_number = set_number
                point.sets_score = f"{sets_a}-{sets_b}"

                self.match_stats.point_history.append(point)

            # update match stats
            aces = 1 if point.was_ace else 0
            dfs = 1 if point.was_double_fault else 0
            first_in = 1 if point.was_first_serve else 0
            first_won = 1 if point.was_first_serve and point.server_won else 0
            second_won = 1 if not point.was_first_serve and point.server_won and not point.was_double_fault else 0

            self._update_match_stats(server_is_a, point.server_won, aces, dfs,
                                    first_in, 1, first_won, second_won, 6, 6)

            # check for tiebreak winner
            if points_a >= 7 or points_b >= 7:
                if abs(points_a - points_b) >= 2:
                    return points_a > points_b

    def _update_match_stats(self, server_is_a: bool, server_won_point: bool,
                           aces: int, dfs: int, first_in: int, first_total: int,
                           first_won: int, second_won: int,
                           games_a: int, games_b: int):
        """update cumulative match statistics after a point or game"""

        if server_is_a:
            self.match_stats.aces_a += aces
            self.match_stats.double_faults_a += dfs
            self.match_stats.first_serves_in_a += first_in
            self.match_stats.first_serves_total_a += first_total
            self.match_stats.first_serve_points_won_a += first_won
            self.match_stats.first_serve_points_total_a += first_in
            self.match_stats.second_serve_points_won_a += second_won
            self.match_stats.second_serve_points_total_a += (first_total - first_in)

            if server_won_point:
                self.match_stats.total_points_won_a += 1
            else:
                self.match_stats.total_points_won_b += 1
        else:
            self.match_stats.aces_b += aces
            self.match_stats.double_faults_b += dfs
            self.match_stats.first_serves_in_b += first_in
            self.match_stats.first_serves_total_b += first_total
            self.match_stats.first_serve_points_won_b += first_won
            self.match_stats.first_serve_points_total_b += first_in
            self.match_stats.second_serve_points_won_b += second_won
            self.match_stats.second_serve_points_total_b += (first_total - first_in)

            if server_won_point:
                self.match_stats.total_points_won_b += 1
            else:
                self.match_stats.total_points_won_a += 1

        self.match_stats.total_points += 1

    def _track_break_point(self, server_is_a: bool, server_saved: bool):
        """track break point statistics"""

        if server_is_a:
            self.match_stats.break_points_total_a += 1
            if not server_saved:
                self.match_stats.break_points_won_b += 1
        else:
            self.match_stats.break_points_total_b += 1
            if not server_saved:
                self.match_stats.break_points_won_a += 1

    def _format_score(self, set_scores: List[Tuple[int, int]]) -> str:
        """format set scores as readable string"""
        return ', '.join(f"{a}-{b}" for a, b in set_scores)

    def _format_game_score(self, server_points: int, returner_points: int) -> str:
        """
        format game score using tennis scoring system

        args:
            server_points: points won by server in current game
            returner_points: points won by returner in current game

        returns:
            formatted score string (e.g., "30-15", "Deuce", "Ad In")
        """
        score_names = ['0', '15', '30', '40']

        # both at 40 or higher = deuce/advantage
        if server_points >= 3 and returner_points >= 3:
            if server_points == returner_points:
                return "Deuce"
            elif server_points > returner_points:
                return "Ad In"  # server advantage
            else:
                return "Ad Out"  # returner advantage

        # normal scoring
        server_score = score_names[min(server_points, 3)]
        returner_score = score_names[min(returner_points, 3)]

        return f"{server_score}-{returner_score}"

    def print_match_summary(self):
        """print formatted match summary"""

        if self.match_stats is None:
            print("no match simulated yet")
            return

        s = self.match_stats

        print(f"\n{'='*70}")
        print(f"{s.player_a_name} vs {s.player_b_name}")
        print(f"Winner: {s.winner}")
        print(f"Score: {s.score}")
        print(f"{'='*70}")
        print(f"\n{'Statistic':<30} {s.player_a_name[:15]:>15} {s.player_b_name[:15]:>15}")
        print(f"{'-'*70}")
        print(f"{'Aces':<30} {s.aces_a:>15} {s.aces_b:>15}")
        print(f"{'Double Faults':<30} {s.double_faults_a:>15} {s.double_faults_b:>15}")

        first_pct_a = s.first_serves_in_a / max(1, s.first_serves_total_a) * 100
        first_pct_b = s.first_serves_in_b / max(1, s.first_serves_total_b) * 100
        print(f"{'First Serve %':<30} {first_pct_a:>14.1f}% {first_pct_b:>14.1f}%")

        print(f"{'1st Serve Points Won':<30} {s.first_serve_points_won_a}/{s.first_serve_points_total_a:>6} "
              f"{s.first_serve_points_won_b}/{s.first_serve_points_total_b:>6}")

        print(f"{'2nd Serve Points Won':<30} {s.second_serve_points_won_a}/{s.second_serve_points_total_a:>6} "
              f"{s.second_serve_points_won_b}/{s.second_serve_points_total_b:>6}")

        print(f"{'Break Points Won':<30} {s.break_points_won_a}/{s.break_points_total_b:>6} "
              f"{s.break_points_won_b}/{s.break_points_total_a:>6}")

        print(f"{'Total Points Won':<30} {s.total_points_won_a:>15} {s.total_points_won_b:>15}")
        print(f"{'='*70}\n")
