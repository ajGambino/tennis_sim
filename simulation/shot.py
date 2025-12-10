"""
shot-level data structures for tennis simulation

defines shots, rallies, and serve outcomes for granular point modeling
"""

from dataclasses import dataclass
from typing import List, Optional, Literal
from enum import Enum


class ShotType(Enum):
    """type of tennis shot"""
    FOREHAND = 'forehand'
    BACKHAND = 'backhand'
    SLICE = 'slice'
    VOLLEY = 'volley'
    SMASH = 'smash'
    DROP_SHOT = 'drop_shot'
    LOB = 'lob'


class ShotDirection(Enum):
    """direction of shot"""
    CROSS_COURT = 'cross'
    DOWN_THE_LINE = 'dtl'
    DOWN_THE_MIDDLE = 'middle'
    WIDE = 'wide'


class ShotOutcome(Enum):
    """result of a shot"""
    IN_PLAY = 'in_play'  # rally continues
    WINNER = 'winner'  # clean winner
    FORCED_ERROR = 'forced_error'  # opponent forced into error
    UNFORCED_ERROR = 'unforced_error'  # player error
    NET_ERROR = 'net_error'  # ball hits net
    OUT_ERROR = 'out_error'  # ball goes out


class CourtPosition(Enum):
    """player position on court"""
    BASELINE = 'baseline'
    INSIDE_BASELINE = 'inside_baseline'
    NET = 'net'
    MID_COURT = 'mid_court'


class ServePlacement(Enum):
    """serve placement zones"""
    WIDE = 'W'  # wide serve (pulls opponent off court)
    T = 'T'  # serve down the T (center line)
    BODY = 'B'  # serve at the body


@dataclass
class Shot:
    """represents a single shot in a rally"""

    shot_number: int  # position in rally (1 = serve, 2 = return, etc)
    player: str  # 'server' or 'returner'
    shot_type: ShotType
    direction: ShotDirection
    outcome: ShotOutcome
    position: CourtPosition
    is_approach: bool = False  # whether player approaches net after this shot

    def is_point_ending(self) -> bool:
        """check if this shot ends the point"""
        return self.outcome in [
            ShotOutcome.WINNER,
            ShotOutcome.FORCED_ERROR,
            ShotOutcome.UNFORCED_ERROR,
            ShotOutcome.NET_ERROR,
            ShotOutcome.OUT_ERROR
        ]

    def is_error(self) -> bool:
        """check if this shot is an error"""
        return self.outcome in [
            ShotOutcome.FORCED_ERROR,
            ShotOutcome.UNFORCED_ERROR,
            ShotOutcome.NET_ERROR,
            ShotOutcome.OUT_ERROR
        ]


@dataclass
class ServeOutcome:
    """result of a serve attempt"""

    is_fault: bool  # whether serve was a fault
    is_ace: bool  # whether serve was an ace
    placement: Optional[ServePlacement]  # where serve was placed (if in)
    speed_mph: Optional[float] = None  # serve speed (if available)

    @property
    def is_in_play(self) -> bool:
        """check if serve is in play (not ace, not fault)"""
        return not self.is_fault and not self.is_ace


@dataclass
class Rally:
    """represents a complete point rally"""

    shots: List[Shot]  # sequence of shots in the rally
    serve_outcome: ServeOutcome  # how the serve went
    is_second_serve: bool  # whether this was a second serve
    winner: Literal['server', 'returner']  # who won the point

    @property
    def rally_length(self) -> int:
        """number of shots in the rally (including serve)"""
        return len(self.shots)

    @property
    def point_ending_shot(self) -> Optional[Shot]:
        """get the shot that ended the point"""
        if not self.shots:
            return None
        for shot in reversed(self.shots):
            if shot.is_point_ending():
                return shot
        return None

    def was_break_point(self) -> bool:
        """check if point was won by returner (break point won)"""
        return self.winner == 'returner'

    def get_shot_count_by_player(self, player: str) -> int:
        """count how many shots a player hit in the rally"""
        return sum(1 for shot in self.shots if shot.player == player)

    def get_winners_by_player(self, player: str) -> int:
        """count winners hit by a player"""
        return sum(
            1 for shot in self.shots
            if shot.player == player and shot.outcome == ShotOutcome.WINNER
        )

    def get_errors_by_player(self, player: str) -> int:
        """count errors made by a player"""
        return sum(1 for shot in self.shots if shot.player == player and shot.is_error())


@dataclass
class PointResultDetailed:
    """extended point result with shot-level details"""

    # basic point result (backward compatible)
    server_won: bool
    was_ace: bool
    was_double_fault: bool
    was_first_serve: bool

    # shot-level extensions
    rally: Optional[Rally] = None
    rally_length: int = 0
    serve_placement: Optional[ServePlacement] = None
    point_ending_shot_type: Optional[ShotType] = None

    @classmethod
    def from_rally(cls, rally: Rally, was_double_fault: bool = False):
        """create detailed point result from a rally simulation"""
        return cls(
            server_won=(rally.winner == 'server'),
            was_ace=rally.serve_outcome.is_ace,
            was_double_fault=was_double_fault,
            was_first_serve=(not rally.is_second_serve),
            rally=rally,
            rally_length=rally.rally_length,
            serve_placement=rally.serve_outcome.placement,
            point_ending_shot_type=rally.point_ending_shot.shot_type if rally.point_ending_shot else None
        )


# charting notation constants for parsing match charting project data
class ChartingNotation:
    """
    constants for parsing match charting project shot notation

    example: '4fxb2@n*'
    - 4: rally length (4 shots)
    - f: forehand
    - x: cross-court
    - b: backhand
    - 2: shot number
    - @: error
    - n: net error
    - *: unforced
    """

    # shot types
    SHOT_TYPES = {
        'f': ShotType.FOREHAND,
        'b': ShotType.BACKHAND,
        's': ShotType.SLICE,
        'v': ShotType.VOLLEY,
        'z': ShotType.SMASH,
        'd': ShotType.DROP_SHOT,
        'l': ShotType.LOB,
    }

    # shot directions
    DIRECTIONS = {
        'x': ShotDirection.CROSS_COURT,
        '1': ShotDirection.DOWN_THE_LINE,
        'i': ShotDirection.DOWN_THE_MIDDLE,
        'w': ShotDirection.WIDE,
    }

    # shot outcomes
    OUTCOMES = {
        '#': ShotOutcome.WINNER,
        '@': ShotOutcome.UNFORCED_ERROR,
        'n': ShotOutcome.NET_ERROR,
        'w': ShotOutcome.OUT_ERROR,
    }

    # serve placements
    SERVE_PLACEMENTS = {
        '4': ServePlacement.WIDE,  # deuce court wide
        '5': ServePlacement.T,     # deuce court T
        '6': ServePlacement.BODY,  # deuce court body
        '1': ServePlacement.WIDE,  # ad court wide
        '2': ServePlacement.T,     # ad court T
        '3': ServePlacement.BODY,  # ad court body
    }

    @staticmethod
    def parse_serve_placement(code: str) -> Optional[ServePlacement]:
        """
        parse serve placement from charting notation

        args:
            code: first character of shot sequence (e.g., '4' for wide in deuce court)

        returns:
            serveplacement or none if not a serve code
        """
        return ChartingNotation.SERVE_PLACEMENTS.get(code)
