"""
rally simulation model

simulates shot-by-shot exchanges after the serve and return
models shot selection, positioning, errors, and winners
"""

import numpy as np
from typing import List, Optional, Tuple
from simulation.shot import (
    Shot, Rally, ServeOutcome, ShotType, ShotDirection,
    ShotOutcome, CourtPosition, ServePlacement
)
from simulation.player_stats import PlayerStats


class RallyModel:
    """
    shot-by-shot rally simulation model

    simulates the exchange after serve and return
    models shot selection based on:
    - player position
    - rally length (fatigue/pressure)
    - player skill
    - shot quality of previous shot
    """

    def __init__(self, rally_patterns: Optional[dict] = None,
                 rng: Optional[np.random.Generator] = None):
        """
        initialize rally model

        args:
            rally_patterns: player-specific rally patterns from charting_loader
            rng: numpy random generator for reproducibility
        """
        self.rally_patterns = rally_patterns or {}
        self.rng = rng if rng is not None else np.random.default_rng()

    def _get_player_aggression(self, player_name: Optional[str] = None) -> float:
        """
        get player's aggression multiplier based on avg rally length

        aggressive players (shorter rallies) have higher error/winner rates
        defensive players (longer rallies) have lower error/winner rates

        args:
            player_name: player's name (for player-specific patterns)

        returns:
            aggression multiplier (0.5-1.5, default 1.0)
        """
        if player_name and player_name in self.rally_patterns:
            pattern = self.rally_patterns[player_name]
            avg_length = pattern.get('avg_rally_length')

            if avg_length:
                # baseline: 3.2 shots (typical avg rally length)
                # federer (2.85) = aggressive → 1.12x multiplier
                # nadal (3.59) = defensive → 0.86x multiplier
                # range in data: 1.64 to 6.87 shots
                baseline = 3.2
                aggression = 1.0 + (baseline - avg_length) * 0.35

                return np.clip(aggression, 0.5, 1.5)

        return 1.0  # default neutral aggression

    # NOTE: This method is currently unused. Rally lengths emerge naturally
    # from aggression-calibrated error/winner rates. Kept for future extensions.
    def _sample_rally_length(self, server_name: Optional[str] = None,
                             returner_name: Optional[str] = None) -> int:
        """
        sample rally length from player-specific learned patterns

        args:
            server_name: server's name (for player-specific patterns)
            returner_name: returner's name (for player-specific patterns)

        returns:
            target rally length (number of shots)
        """
        # try to get player patterns (prefer server, fallback to returner)
        pattern = None
        if server_name and server_name in self.rally_patterns:
            pattern = self.rally_patterns[server_name]
        elif returner_name and returner_name in self.rally_patterns:
            pattern = self.rally_patterns[returner_name]

        if pattern and 'rally_length_dist' in pattern:
            # sample from learned distribution
            dist = pattern['rally_length_dist']
            # distribution is indexed by rally length (0, 1, 2, 3, ...)
            rally_length = self.rng.choice(len(dist), p=dist)
            return rally_length
        else:
            # fallback: sample from default distribution
            # realistic rally lengths: most rallies 2-6 shots, fewer long rallies
            default_dist = [
                0.0,    # 0 shots (not possible in rally exchange)
                0.0,    # 1 shot (serve only, handled elsewhere)
                0.30,   # 2 shots (return ends point)
                0.20,   # 3 shots
                0.15,   # 4 shots
                0.10,   # 5 shots
                0.08,   # 6 shots
                0.06,   # 7 shots
                0.04,   # 8 shots
                0.03,   # 9 shots
                0.02,   # 10 shots
                0.01,   # 11 shots
                0.01,   # 12+ shots
            ]
            rally_length = self.rng.choice(len(default_dist), p=default_dist)
            return rally_length

    def simulate_rally_exchange(self, server_stats: PlayerStats,
                               returner_stats: PlayerStats,
                               return_shot: Shot,
                               serve_placement: ServePlacement,
                               is_second_serve: bool,
                               surface: str = 'hard',
                               server_name: Optional[str] = None,
                               returner_name: Optional[str] = None) -> List[Shot]:
        """
        simulate rally exchange after the return

        args:
            server_stats: server's stats
            returner_stats: returner's stats
            return_shot: the return shot (shot #2)
            serve_placement: where serve was placed
            is_second_serve: whether this was a second serve
            surface: court surface
            server_name: server's name (for player-specific rally patterns)
            returner_name: returner's name (for player-specific rally patterns)

        returns:
            list of shots in the rally (including serve and return)
        """
        shots = []

        # shot 1: serve (not simulated here, but counted)
        serve_shot = Shot(
            shot_number=1,
            player='server',
            shot_type=ShotType.FOREHAND,  # serves are technically forehands
            direction=self._placement_to_direction(serve_placement),
            outcome=ShotOutcome.IN_PLAY,
            position=CourtPosition.BASELINE,
            is_approach=False
        )
        shots.append(serve_shot)

        # shot 2: return (already simulated)
        shots.append(return_shot)

        # if return ended point, we're done
        if return_shot.is_point_ending():
            return shots

        # get player aggression levels from rally patterns
        server_aggression = self._get_player_aggression(server_name)
        returner_aggression = self._get_player_aggression(returner_name)

        # continue rally shot by shot
        current_player = 'server'  # server hits shot 3
        shot_number = 3

        # track positions
        server_position = self._get_position_after_serve(serve_placement)
        returner_position = CourtPosition.BASELINE

        # track previous shot quality
        previous_shot_quality = self._assess_shot_quality(return_shot)

        # rally continues until someone hits a point-ending shot
        max_rally_length = 30  # safety limit

        while shot_number <= max_rally_length:
            # determine who's hitting and their stats
            if current_player == 'server':
                player_stats = server_stats
                opponent_stats = returner_stats
                player_position = server_position
                opponent_position = returner_position
                player_aggression = server_aggression
            else:
                player_stats = returner_stats
                opponent_stats = server_stats
                player_position = returner_position
                opponent_position = server_position
                player_aggression = returner_aggression

            # simulate next shot
            shot = self._simulate_rally_shot(
                player_stats=player_stats,
                opponent_stats=opponent_stats,
                shot_number=shot_number,
                player=current_player,
                player_position=player_position,
                previous_shot_quality=previous_shot_quality,
                surface=surface,
                player_aggression=player_aggression,
                opponent_position=opponent_position
            )

            shots.append(shot)

            # if shot ended point, we're done
            if shot.is_point_ending():
                break

            # update positions
            if current_player == 'server':
                server_position = self._update_position(shot, server_position)
            else:
                returner_position = self._update_position(shot, returner_position)

            # update shot quality for next iteration
            previous_shot_quality = self._assess_shot_quality(shot)

            # switch players
            current_player = 'returner' if current_player == 'server' else 'server'
            shot_number += 1

        return shots

    def _simulate_rally_shot(self, player_stats: PlayerStats,
                             opponent_stats: PlayerStats,
                             shot_number: int,
                             player: str,
                             player_position: CourtPosition,
                             previous_shot_quality: str,
                             surface: str,
                             player_aggression: float = 1.0,
                             opponent_position: CourtPosition = CourtPosition.BASELINE) -> Shot:
        """
        simulate a single rally shot

        args:
            player_stats: current player's stats
            opponent_stats: opponent's stats
            shot_number: shot number in rally
            player: 'server' or 'returner'
            player_position: current position
            previous_shot_quality: 'offensive', 'neutral', 'defensive'
            surface: court surface
            player_aggression: aggression multiplier from player's avg rally length
            opponent_position: opponent's court position (for tactical shot selection)

        returns:
            shot object
        """
        # determine shot type based on position and situation
        if player_position == CourtPosition.NET:
            # at net: volley, smash, or groundstroke
            rand = self.rng.random()
            if rand < 0.65:
                shot_type = ShotType.VOLLEY
            elif rand < 0.75:  # 10% smash when at net
                shot_type = ShotType.SMASH
            else:
                shot_type = ShotType.FOREHAND if self.rng.random() < 0.5 else ShotType.BACKHAND
        elif opponent_position == CourtPosition.NET:
            # opponent at net: lob or passing shot
            if self.rng.random() < 0.25:  # 25% lob when opponent at net
                shot_type = ShotType.LOB
            else:
                # passing shot (forehand/backhand)
                shot_type = ShotType.FOREHAND if self.rng.random() < 0.55 else ShotType.BACKHAND
        elif player_position == CourtPosition.INSIDE_BASELINE:
            # attacking position: more aggressive shot selection
            rand = self.rng.random()
            if rand < 0.05:  # 5% drop shot when attacking
                shot_type = ShotType.DROP_SHOT
            elif rand < 0.20:  # 15% slice (approach or change of pace)
                shot_type = ShotType.SLICE
            else:
                shot_type = ShotType.FOREHAND if self.rng.random() < 0.60 else ShotType.BACKHAND
        elif previous_shot_quality == 'defensive':
            # defensive situation: more slice and safe shots
            rand = self.rng.random()
            if rand < 0.25:  # 25% slice when defensive
                shot_type = ShotType.SLICE
            elif rand < 0.30:  # 5% lob to reset
                shot_type = ShotType.LOB
            else:
                shot_type = ShotType.FOREHAND if self.rng.random() < 0.50 else ShotType.BACKHAND
        else:
            # neutral baseline rally
            rand = self.rng.random()
            if rand < 0.10:  # 10% slice (variety)
                shot_type = ShotType.SLICE
            elif rand < 0.12:  # 2% drop shot
                shot_type = ShotType.DROP_SHOT
            else:
                # 55% forehand, 33% backhand
                shot_type = ShotType.FOREHAND if self.rng.random() < 0.625 else ShotType.BACKHAND

        # determine direction
        direction = self._choose_shot_direction(previous_shot_quality, player_position)

        # determine whether to approach net
        is_approach = self._should_approach_net(
            player_position, previous_shot_quality, shot_number
        )

        # determine shot outcome (winner/error/in_play)
        outcome = self._determine_shot_outcome(
            player_stats=player_stats,
            opponent_stats=opponent_stats,
            shot_type=shot_type,
            player_position=player_position,
            previous_shot_quality=previous_shot_quality,
            shot_number=shot_number,
            surface=surface,
            player_aggression=player_aggression
        )

        return Shot(
            shot_number=shot_number,
            player=player,
            shot_type=shot_type,
            direction=direction,
            outcome=outcome,
            position=player_position,
            is_approach=is_approach
        )

    def _determine_shot_outcome(self, player_stats: PlayerStats,
                                opponent_stats: PlayerStats,
                                shot_type: ShotType,
                                player_position: CourtPosition,
                                previous_shot_quality: str,
                                shot_number: int,
                                surface: str,
                                player_aggression: float = 1.0) -> ShotOutcome:
        """
        determine whether shot is winner/error/in_play

        args:
            player_stats: player's stats
            opponent_stats: opponent's stats
            shot_type: type of shot
            player_position: player's position
            previous_shot_quality: quality of previous shot
            shot_number: shot number in rally
            surface: court surface
            player_aggression: aggression multiplier (higher = more winners/errors)

        returns:
            shotoutcome
        """
        # compute base error rate and winner rate
        base_error_rate = self._get_base_error_rate(player_stats, opponent_stats)
        base_winner_rate = self._get_base_winner_rate(player_stats, opponent_stats)

        # apply player aggression multiplier
        # aggressive players (shorter avg rallies): more winners AND more errors
        # defensive players (longer avg rallies): fewer winners AND fewer errors
        base_error_rate *= player_aggression
        base_winner_rate *= player_aggression

        # adjust for position
        if player_position == CourtPosition.NET:
            # volleys have lower error rate but higher winner rate
            base_error_rate *= 0.70
            base_winner_rate *= 2.50
        elif player_position == CourtPosition.INSIDE_BASELINE:
            # attacking position
            base_error_rate *= 1.20
            base_winner_rate *= 1.80

        # adjust for shot quality
        if previous_shot_quality == 'defensive':
            # under pressure, more errors
            base_error_rate *= 1.60
            base_winner_rate *= 0.30
        elif previous_shot_quality == 'offensive':
            # good position, more winners
            base_error_rate *= 0.80
            base_winner_rate *= 1.50

        # adjust for rally length (fatigue/pressure)
        if shot_number > 10:
            # long rallies increase error rate
            fatigue_factor = 1.0 + (shot_number - 10) * 0.03
            base_error_rate *= fatigue_factor

        # adjust for shot type
        if shot_type == ShotType.VOLLEY:
            base_winner_rate *= 1.30
        elif shot_type == ShotType.SMASH:
            base_winner_rate *= 3.00
            base_error_rate *= 0.50

        # clamp rates
        error_rate = np.clip(base_error_rate, 0.0, 0.50)
        winner_rate = np.clip(base_winner_rate, 0.0, 0.25)

        # sample outcome
        rand = self.rng.random()

        if rand < error_rate:
            # error occurred - choose type
            return self.rng.choice([
                ShotOutcome.UNFORCED_ERROR,
                ShotOutcome.NET_ERROR,
                ShotOutcome.OUT_ERROR
            ], p=[0.40, 0.35, 0.25])

        elif rand < error_rate + winner_rate:
            # winner
            return ShotOutcome.WINNER

        else:
            # rally continues
            return ShotOutcome.IN_PLAY

    def _get_base_error_rate(self, player_stats: PlayerStats,
                            opponent_stats: PlayerStats) -> float:
        """
        compute base error rate from player stats

        better players make fewer errors
        """
        # use serve win % and return win % as proxies for overall skill
        player_skill = (player_stats.first_serve_win_pct + player_stats.return_first_win_pct) / 2
        opponent_skill = (opponent_stats.first_serve_win_pct + opponent_stats.return_first_win_pct) / 2

        # base error rate around 12-16% per shot
        base_error = 0.14

        # better players have lower error rates
        skill_adjustment = 1.0 - (player_skill - 0.50)  # assuming 0.50 is average

        error_rate = base_error * skill_adjustment

        # add opponent pressure
        opponent_pressure = opponent_skill - 0.50
        error_rate += opponent_pressure * 0.05

        return np.clip(error_rate, 0.05, 0.20)

    def _get_base_winner_rate(self, player_stats: PlayerStats,
                             opponent_stats: PlayerStats) -> float:
        """
        compute base winner rate from player stats

        better players hit more winners
        """
        # use serve win % and return win % as proxies for overall skill
        player_skill = (player_stats.first_serve_win_pct + player_stats.return_first_win_pct) / 2

        # base winner rate around 8-12% per shot
        base_winner = 0.10

        # better players hit more winners
        skill_adjustment = 1.0 + (player_skill - 0.50) * 1.0

        winner_rate = base_winner * skill_adjustment

        return np.clip(winner_rate, 0.02, 0.12)

    def _assess_shot_quality(self, shot: Shot) -> str:
        """
        assess quality of a shot (offensive/neutral/defensive)

        args:
            shot: shot to assess

        returns:
            'offensive', 'neutral', or 'defensive'
        """
        # errors are defensive (opponent was in control)
        if shot.is_error():
            return 'defensive'

        # winners are offensive
        if shot.outcome == ShotOutcome.WINNER:
            return 'offensive'

        # approach shots are offensive
        if shot.is_approach:
            return 'offensive'

        # position-based assessment
        if shot.position == CourtPosition.NET:
            return 'offensive'
        elif shot.position == CourtPosition.INSIDE_BASELINE:
            return 'offensive'
        else:
            return 'neutral'

    def _choose_shot_direction(self, previous_shot_quality: str,
                              player_position: CourtPosition) -> ShotDirection:
        """
        choose shot direction based on position and tactics

        args:
            previous_shot_quality: quality of previous shot
            player_position: player's position

        returns:
            shotdirection
        """
        if previous_shot_quality == 'defensive':
            # under pressure, play safe (cross-court)
            return self.rng.choice([
                ShotDirection.CROSS_COURT,
                ShotDirection.DOWN_THE_MIDDLE,
                ShotDirection.DOWN_THE_LINE
            ], p=[0.70, 0.20, 0.10])

        elif previous_shot_quality == 'offensive':
            # attacking, be more aggressive
            return self.rng.choice([
                ShotDirection.CROSS_COURT,
                ShotDirection.DOWN_THE_LINE,
                ShotDirection.DOWN_THE_MIDDLE
            ], p=[0.45, 0.40, 0.15])

        else:  # neutral
            return self.rng.choice([
                ShotDirection.CROSS_COURT,
                ShotDirection.DOWN_THE_LINE,
                ShotDirection.DOWN_THE_MIDDLE
            ], p=[0.60, 0.25, 0.15])

    def _should_approach_net(self, position: CourtPosition,
                            previous_shot_quality: str,
                            shot_number: int) -> bool:
        """
        determine whether to approach net on this shot

        args:
            position: current position
            previous_shot_quality: quality of previous shot
            shot_number: shot number

        returns:
            whether to approach net
        """
        # already at net
        if position == CourtPosition.NET:
            return False

        # only approach on offensive shots
        if previous_shot_quality != 'offensive':
            return False

        # more likely to approach on short rallies
        if shot_number > 8:
            approach_prob = 0.05
        else:
            approach_prob = 0.15

        return self.rng.random() < approach_prob

    def _update_position(self, shot: Shot, current_position: CourtPosition) -> CourtPosition:
        """
        update player position after hitting a shot

        args:
            shot: shot that was hit
            current_position: current position

        returns:
            new position
        """
        # approaching net: move to mid-court then net
        if shot.is_approach:
            if current_position == CourtPosition.BASELINE:
                return CourtPosition.MID_COURT
            elif current_position == CourtPosition.INSIDE_BASELINE or current_position == CourtPosition.MID_COURT:
                return CourtPosition.NET
            else:
                return CourtPosition.NET

        # if at net, stay at net unless forced back
        if current_position == CourtPosition.NET:
            if shot.outcome == ShotOutcome.IN_PLAY:
                return CourtPosition.NET
            else:
                return current_position

        # if in mid-court and not approaching, complete approach to net
        if current_position == CourtPosition.MID_COURT:
            return CourtPosition.NET

        # drop shots bring player inside baseline
        if shot.shot_type == ShotType.DROP_SHOT:
            return CourtPosition.INSIDE_BASELINE

        # move inside baseline on offensive shots (20% chance)
        if current_position == CourtPosition.BASELINE:
            # check if shot is offensive (aggressive direction or winner)
            if shot.outcome == ShotOutcome.WINNER or shot.direction == ShotDirection.DOWN_THE_LINE:
                if self.rng.random() < 0.20:
                    return CourtPosition.INSIDE_BASELINE

        # after being inside baseline, drift back to baseline (60%) or stay (40%)
        if current_position == CourtPosition.INSIDE_BASELINE:
            if shot.shot_type in [ShotType.FOREHAND, ShotType.BACKHAND]:
                if self.rng.random() < 0.60:
                    return CourtPosition.BASELINE

        # default: stay at current position
        return current_position

    def _get_position_after_serve(self, serve_placement: ServePlacement) -> CourtPosition:
        """
        determine server's position after serve

        args:
            serve_placement: where serve was placed

        returns:
            server's position
        """
        # most servers stay at baseline in modern game
        # occasionally approach net on wide serves
        if serve_placement == ServePlacement.WIDE:
            if self.rng.random() < 0.10:  # 10% of time approach on wide serve
                return CourtPosition.NET

        return CourtPosition.BASELINE

    def _placement_to_direction(self, placement: ServePlacement) -> ShotDirection:
        """convert serve placement to shot direction"""
        mapping = {
            ServePlacement.WIDE: ShotDirection.WIDE,
            ServePlacement.T: ShotDirection.DOWN_THE_MIDDLE,
            ServePlacement.BODY: ShotDirection.DOWN_THE_MIDDLE,
        }
        return mapping.get(placement, ShotDirection.DOWN_THE_MIDDLE)

    def create_rally_from_shots(self, shots: List[Shot],
                               serve_outcome: ServeOutcome,
                               is_second_serve: bool) -> Rally:
        """
        create rally object from list of shots

        args:
            shots: list of shots in rally
            serve_outcome: outcome of the serve
            is_second_serve: whether this was a second serve

        returns:
            rally object
        """
        # determine winner
        last_shot = shots[-1]

        if last_shot.is_error():
            # player who made error loses
            winner = 'returner' if last_shot.player == 'server' else 'server'
        elif last_shot.outcome == ShotOutcome.WINNER:
            # player who hit winner wins
            winner = last_shot.player
        else:
            # shouldn't happen, but default to server
            winner = 'server'

        return Rally(
            shots=shots,
            serve_outcome=serve_outcome,
            is_second_serve=is_second_serve,
            winner=winner
        )
