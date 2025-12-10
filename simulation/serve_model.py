"""
serve placement and outcome model

models serve placement (wide/T/body), ace probabilities, and fault rates
based on match charting project data and player-specific patterns
"""

import numpy as np
from typing import Dict, Optional
from simulation.shot import ServePlacement, ServeOutcome
from simulation.player_stats import PlayerStats


class ServeModel:
    """
    probabilistic serve placement and outcome model

    learns serve patterns from match charting project data
    samples serve placement and determines outcome (ace/fault/in-play)
    """

    def __init__(self, serve_patterns: Optional[Dict[str, Dict]] = None,
                 default_pattern: Optional[Dict] = None,
                 rng: Optional[np.random.Generator] = None):
        """
        initialize serve model

        args:
            serve_patterns: dict mapping player names to serve pattern stats
                           from charting_loader.extract_serve_patterns()
            default_pattern: default serve pattern for unknown players
            rng: numpy random generator for reproducibility
        """
        self.serve_patterns = serve_patterns or {}
        self.default_pattern = default_pattern or self._get_hardcoded_default()
        self.rng = rng if rng is not None else np.random.default_rng()

    def _get_hardcoded_default(self) -> Dict:
        """get hardcoded default serve pattern"""
        return {
            'wide_pct': 0.35,
            'T_pct': 0.40,
            'body_pct': 0.25,
            'ace_pct_by_placement': {
                'W': 0.06,
                'T': 0.09,  # T serves have higher ace rate
                'B': 0.03,
            },
            'overall_ace_pct': 0.06
        }

    def sample_serve_placement(self, player_name: Optional[str] = None,
                               surface: str = 'hard',
                               is_second_serve: bool = False) -> ServePlacement:
        """
        sample serve placement from player's learned pattern

        args:
            player_name: name of server (for player-specific patterns)
            surface: court surface ('hard', 'clay', 'grass')
            is_second_serve: whether this is a second serve

        returns:
            serveplacement (W/T/B)
        """
        # get player's serve pattern or use default
        pattern = self.serve_patterns.get(player_name, self.default_pattern)

        # base probabilities
        wide_pct = pattern['wide_pct']
        t_pct = pattern['T_pct']
        body_pct = pattern['body_pct']

        # adjust for surface
        if surface == 'grass':
            # more T serves on grass (faster surface, harder to return)
            t_pct *= 1.15
            wide_pct *= 0.92
            body_pct *= 0.93
        elif surface == 'clay':
            # more wide serves on clay (pull opponent off court)
            wide_pct *= 1.10
            t_pct *= 0.95
            body_pct *= 0.95

        # adjust for second serve (more conservative)
        if is_second_serve:
            # second serves typically more to body/T (safer)
            t_pct *= 1.15
            body_pct *= 1.10
            wide_pct *= 0.80

        # renormalize to ensure probabilities sum to 1
        total = wide_pct + t_pct + body_pct
        wide_pct /= total
        t_pct /= total
        body_pct /= total

        # sample placement
        placement_choice = self.rng.choice(
            [ServePlacement.WIDE, ServePlacement.T, ServePlacement.BODY],
            p=[wide_pct, t_pct, body_pct]
        )

        return placement_choice

    def sample_serve_outcome(self, player_stats: PlayerStats,
                             placement: ServePlacement,
                             is_second_serve: bool = False,
                             surface: str = 'hard') -> ServeOutcome:
        """
        sample serve outcome given placement and player stats

        args:
            player_stats: server's stats (for ace % and fault %)
            placement: where the serve was placed
            is_second_serve: whether this is a second serve
            surface: court surface

        returns:
            serveoutcome with fault/ace/in_play status
        """
        # check for fault first
        if is_second_serve:
            # second serve - use double fault rate
            fault_prob = player_stats.df_pct
        else:
            # first serve - compute fault rate from first serve in %
            fault_prob = 1.0 - player_stats.first_serve_pct

        is_fault = self.rng.random() < fault_prob

        if is_fault:
            return ServeOutcome(
                is_fault=True,
                is_ace=False,
                placement=None
            )

        # serve is in - check for ace
        ace_prob = self._get_ace_probability(
            player_stats=player_stats,
            placement=placement,
            is_second_serve=is_second_serve,
            surface=surface
        )

        is_ace = self.rng.random() < ace_prob

        return ServeOutcome(
            is_fault=False,
            is_ace=is_ace,
            placement=placement
        )

    def _get_ace_probability(self, player_stats: PlayerStats,
                            placement: ServePlacement,
                            is_second_serve: bool,
                            surface: str) -> float:
        """
        compute ace probability based on placement and context

        args:
            player_stats: server's stats
            placement: serve placement
            is_second_serve: whether second serve
            surface: court surface

        returns:
            probability of ace
        """
        # base ace rate from player stats
        base_ace_rate = player_stats.ace_pct

        # second serves rarely aces
        if is_second_serve:
            return base_ace_rate * 0.05  # ~5% of first serve ace rate

        # adjust by placement
        # T serves have highest ace rate, body serves lowest
        placement_multipliers = {
            ServePlacement.WIDE: 0.85,
            ServePlacement.T: 1.30,
            ServePlacement.BODY: 0.60,
        }

        placement_mult = placement_multipliers.get(placement, 1.0)

        # adjust by surface
        surface_multipliers = {
            'grass': 1.25,  # faster surface, more aces
            'hard': 1.0,
            'clay': 0.70,   # slower surface, fewer aces
            'carpet': 1.15
        }

        surface_mult = surface_multipliers.get(surface, 1.0)

        # combined probability
        ace_prob = base_ace_rate * placement_mult * surface_mult

        # clamp to reasonable range
        return np.clip(ace_prob, 0.0, 0.35)

    def simulate_first_serve(self, player_stats: PlayerStats,
                            player_name: Optional[str] = None,
                            surface: str = 'hard') -> ServeOutcome:
        """
        simulate a complete first serve (placement + outcome)

        args:
            player_stats: server's stats
            player_name: server's name (for player-specific patterns)
            surface: court surface

        returns:
            serveoutcome
        """
        placement = self.sample_serve_placement(
            player_name=player_name,
            surface=surface,
            is_second_serve=False
        )

        return self.sample_serve_outcome(
            player_stats=player_stats,
            placement=placement,
            is_second_serve=False,
            surface=surface
        )

    def simulate_second_serve(self, player_stats: PlayerStats,
                             player_name: Optional[str] = None,
                             surface: str = 'hard') -> ServeOutcome:
        """
        simulate a complete second serve (placement + outcome)

        args:
            player_stats: server's stats
            player_name: server's name
            surface: court surface

        returns:
            serveoutcome
        """
        placement = self.sample_serve_placement(
            player_name=player_name,
            surface=surface,
            is_second_serve=True
        )

        return self.sample_serve_outcome(
            player_stats=player_stats,
            placement=placement,
            is_second_serve=True,
            surface=surface
        )

    def get_placement_distribution(self, player_name: Optional[str] = None,
                                   surface: str = 'hard',
                                   is_second_serve: bool = False) -> Dict[str, float]:
        """
        get serve placement distribution for analysis

        args:
            player_name: server name
            surface: court surface
            is_second_serve: whether second serve

        returns:
            dict with placement probabilities
        """
        pattern = self.serve_patterns.get(player_name, self.default_pattern)

        wide_pct = pattern['wide_pct']
        t_pct = pattern['T_pct']
        body_pct = pattern['body_pct']

        # apply same adjustments as sample_serve_placement
        if surface == 'grass':
            t_pct *= 1.15
            wide_pct *= 0.92
            body_pct *= 0.93
        elif surface == 'clay':
            wide_pct *= 1.10
            t_pct *= 0.95
            body_pct *= 0.95

        if is_second_serve:
            t_pct *= 1.15
            body_pct *= 1.10
            wide_pct *= 0.80

        # normalize
        total = wide_pct + t_pct + body_pct
        wide_pct /= total
        t_pct /= total
        body_pct /= total

        return {
            'wide': wide_pct,
            'T': t_pct,
            'body': body_pct
        }

    def add_player_pattern(self, player_name: str, pattern: Dict):
        """
        add or update serve pattern for a player

        args:
            player_name: player name
            pattern: serve pattern dict from charting_loader
        """
        self.serve_patterns[player_name] = pattern

    def set_default_pattern(self, pattern: Dict):
        """
        set default serve pattern for unknown players

        args:
            pattern: default serve pattern dict
        """
        self.default_pattern = pattern
