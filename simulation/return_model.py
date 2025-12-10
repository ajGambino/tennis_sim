"""
return quality model

models return shot outcomes based on serve placement and returner skill
determines whether return is a winner, deep, short, or error
"""

import numpy as np
from typing import Literal, Optional
from simulation.shot import ServePlacement, Shot, ShotType, ShotDirection, ShotOutcome, CourtPosition
from simulation.player_stats import PlayerStats


ReturnQuality = Literal['winner', 'deep', 'short', 'error']


class ReturnModel:
    """
    probabilistic return outcome model

    models return quality based on:
    - serve placement (harder to return wide/T serves)
    - returner skill (return win %)
    - serve speed (first vs second serve)
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        initialize return model

        args:
            rng: numpy random generator for reproducibility
        """
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_return_quality(self, returner_stats: PlayerStats,
                             serve_placement: ServePlacement,
                             is_second_serve: bool,
                             surface: str = 'hard') -> ReturnQuality:
        """
        sample return quality based on serve and returner

        args:
            returner_stats: returner's stats
            serve_placement: where serve was placed
            is_second_serve: whether this is a second serve
            surface: court surface

        returns:
            return quality: 'winner', 'deep', 'short', or 'error'
        """
        # get base return win rate
        if is_second_serve:
            base_return_win = returner_stats.return_second_win_pct
        else:
            base_return_win = returner_stats.return_first_win_pct

        # adjust for serve placement
        # wide and T serves are harder to return
        placement_multipliers = {
            ServePlacement.WIDE: 0.85,  # harder to return
            ServePlacement.T: 0.80,     # hardest to return (jams returner)
            ServePlacement.BODY: 0.95,  # easier to return
        }

        placement_mult = placement_multipliers.get(serve_placement, 1.0)
        adjusted_return_win = base_return_win * placement_mult

        # compute probabilities for each outcome
        # winner: rare but possible especially on second serves
        winner_prob = self._get_return_winner_probability(
            returner_stats=returner_stats,
            serve_placement=serve_placement,
            is_second_serve=is_second_serve,
            surface=surface
        )

        # error: return errors are relatively rare in professional tennis
        # most returns go in, but may be weak/strong
        # typical return error rates: 10-20% depending on serve quality
        base_error_rate = 0.15  # 15% of returns are errors

        # adjust for serve placement
        if serve_placement == ServePlacement.T:
            # T serves are hardest to return (jams returner)
            error_prob = base_error_rate * 1.40
        elif serve_placement == ServePlacement.WIDE:
            # wide serves force returner off court
            error_prob = base_error_rate * 1.25
        else:  # body
            # body serves are easier to return
            error_prob = base_error_rate * 0.90

        # adjust for first vs second serve
        if is_second_serve:
            # second serves are slower/weaker, easier to return
            error_prob *= 0.60
        else:
            # first serves are harder
            error_prob *= 1.10

        # adjust for returner skill
        # better returners make fewer errors
        if is_second_serve:
            returner_quality = returner_stats.return_second_win_pct
        else:
            returner_quality = returner_stats.return_first_win_pct

        skill_mult = 1.0 - (returner_quality - 0.35) * 0.8  # scale around 35% baseline
        error_prob *= np.clip(skill_mult, 0.5, 1.5)

        # clamp error probability
        error_prob = np.clip(error_prob, 0.05, 0.35)

        # remaining probability split between deep and short returns
        remaining_prob = 1.0 - winner_prob - error_prob
        remaining_prob = max(0.0, remaining_prob)

        # deep returns are generally better (60-70% of successful returns)
        # short returns give opponent opportunity to attack
        deep_prob = remaining_prob * 0.65
        short_prob = remaining_prob * 0.35

        # sample outcome
        outcomes = ['winner', 'error', 'deep', 'short']
        probs = [winner_prob, error_prob, deep_prob, short_prob]

        # normalize probabilities
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            # fallback
            probs = [0.0, 0.3, 0.5, 0.2]

        return self.rng.choice(outcomes, p=probs)

    def _get_return_winner_probability(self, returner_stats: PlayerStats,
                                       serve_placement: ServePlacement,
                                       is_second_serve: bool,
                                       surface: str) -> float:
        """
        compute probability of return winner

        args:
            returner_stats: returner's stats
            serve_placement: serve placement
            is_second_serve: whether second serve
            surface: court surface

        returns:
            probability of return winner
        """
        # base return winner rate (very low)
        base_winner_rate = 0.02  # ~2% of returns are outright winners

        # much higher on second serves (weaker, slower)
        if is_second_serve:
            base_winner_rate *= 3.0

        # adjust by returner quality
        # better returners hit more return winners
        # use return win % as proxy for returner quality
        if is_second_serve:
            returner_quality = returner_stats.return_second_win_pct
        else:
            returner_quality = returner_stats.return_first_win_pct

        # scale by returner quality (better returners = more winners)
        quality_mult = 0.5 + (returner_quality * 1.5)
        base_winner_rate *= quality_mult

        # adjust by serve placement
        # body serves easier to attack
        placement_multipliers = {
            ServePlacement.WIDE: 0.70,
            ServePlacement.T: 0.60,
            ServePlacement.BODY: 1.40,
        }

        placement_mult = placement_multipliers.get(serve_placement, 1.0)

        # adjust by surface
        # slower surfaces allow more time for aggressive returns
        surface_multipliers = {
            'clay': 1.15,
            'hard': 1.0,
            'grass': 0.80,
            'carpet': 0.85
        }

        surface_mult = surface_multipliers.get(surface, 1.0)

        winner_prob = base_winner_rate * placement_mult * surface_mult

        # clamp to reasonable range
        return np.clip(winner_prob, 0.0, 0.15)

    def create_return_shot(self, return_quality: ReturnQuality,
                          serve_placement: ServePlacement,
                          returner_forehand_pct: float = 0.55) -> Shot:
        """
        create a shot object for the return

        args:
            return_quality: quality of return ('winner', 'deep', 'short', 'error')
            serve_placement: where serve was placed
            returner_forehand_pct: probability returner uses forehand

        returns:
            shot object representing the return
        """
        # determine shot type (forehand or backhand)
        # depends on serve placement and returner's handedness
        # for simplicity, assume right-handed returner
        if serve_placement == ServePlacement.WIDE:
            # wide serve typically goes to backhand (ad court) or forehand (deuce court)
            # assume 50/50
            use_forehand = self.rng.random() < 0.5
        elif serve_placement == ServePlacement.T:
            # T serve can be hit with either wing
            use_forehand = self.rng.random() < returner_forehand_pct
        else:  # BODY
            # body serves often lead to forehand returns
            use_forehand = self.rng.random() < 0.65

        shot_type = ShotType.FOREHAND if use_forehand else ShotType.BACKHAND

        # determine direction
        # returns typically go cross-court (safer, higher margin)
        direction_probs = {
            ShotDirection.CROSS_COURT: 0.60,
            ShotDirection.DOWN_THE_LINE: 0.25,
            ShotDirection.DOWN_THE_MIDDLE: 0.15
        }

        direction = self.rng.choice(
            list(direction_probs.keys()),
            p=list(direction_probs.values())
        )

        # determine outcome
        if return_quality == 'winner':
            outcome = ShotOutcome.WINNER
        elif return_quality == 'error':
            # randomly choose error type
            outcome = self.rng.choice([
                ShotOutcome.UNFORCED_ERROR,
                ShotOutcome.NET_ERROR,
                ShotOutcome.OUT_ERROR
            ], p=[0.4, 0.35, 0.25])
        else:  # deep or short
            outcome = ShotOutcome.IN_PLAY

        # returner is always at baseline
        position = CourtPosition.BASELINE

        return Shot(
            shot_number=2,  # return is always shot #2
            player='returner',
            shot_type=shot_type,
            direction=direction,
            outcome=outcome,
            position=position,
            is_approach=False
        )

    def simulate_return(self, returner_stats: PlayerStats,
                       serve_placement: ServePlacement,
                       is_second_serve: bool,
                       surface: str = 'hard') -> Shot:
        """
        simulate complete return shot

        args:
            returner_stats: returner's stats
            serve_placement: where serve was placed
            is_second_serve: whether second serve
            surface: court surface

        returns:
            shot object for the return
        """
        return_quality = self.sample_return_quality(
            returner_stats=returner_stats,
            serve_placement=serve_placement,
            is_second_serve=is_second_serve,
            surface=surface
        )

        return self.create_return_shot(
            return_quality=return_quality,
            serve_placement=serve_placement
        )

    def get_return_quality_distribution(self, returner_stats: PlayerStats,
                                       serve_placement: ServePlacement,
                                       is_second_serve: bool,
                                       surface: str = 'hard') -> dict:
        """
        get return quality probability distribution for analysis

        args:
            returner_stats: returner's stats
            serve_placement: serve placement
            is_second_serve: whether second serve
            surface: court surface

        returns:
            dict with outcome probabilities
        """
        # get base return win rate
        if is_second_serve:
            base_return_win = returner_stats.return_second_win_pct
        else:
            base_return_win = returner_stats.return_first_win_pct

        # apply placement adjustment
        placement_multipliers = {
            ServePlacement.WIDE: 0.85,
            ServePlacement.T: 0.80,
            ServePlacement.BODY: 0.95,
        }
        placement_mult = placement_multipliers.get(serve_placement, 1.0)
        adjusted_return_win = base_return_win * placement_mult

        # compute probabilities
        winner_prob = self._get_return_winner_probability(
            returner_stats, serve_placement, is_second_serve, surface
        )
        error_prob = 1.0 - adjusted_return_win
        remaining_prob = max(0.0, 1.0 - winner_prob - error_prob)
        deep_prob = remaining_prob * 0.65
        short_prob = remaining_prob * 0.35

        return {
            'winner': winner_prob,
            'error': error_prob,
            'deep': deep_prob,
            'short': short_prob
        }
