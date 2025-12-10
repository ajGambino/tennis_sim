"""
elo rating system for tennis players

computes and maintains elo ratings from historical match results
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class EloSystem:
    """manages elo ratings for tennis players"""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0,
                 surface_specific: bool = True):
        """
        initialize elo system

        args:
            k_factor: sensitivity of rating changes (higher = more volatile)
            initial_rating: starting elo for new players
            surface_specific: maintain separate elo per surface
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.surface_specific = surface_specific

        # storage: {player_name: {surface: rating}} or {player_name: rating}
        self.ratings = {}

        # track rating history for each player
        self.rating_history = {}

    def get_rating(self, player_name: str, surface: Optional[str] = None) -> float:
        """
        get current elo rating for a player

        args:
            player_name: player name
            surface: surface type (hard/clay/grass) if surface_specific=true

        returns:
            elo rating (defaults to initial_rating if player not found)
        """
        if player_name not in self.ratings:
            return self.initial_rating

        if self.surface_specific and surface:
            if isinstance(self.ratings[player_name], dict):
                return self.ratings[player_name].get(surface, self.initial_rating)
            else:
                # backwards compatibility
                return self.ratings[player_name]
        else:
            if isinstance(self.ratings[player_name], dict):
                # if surface-specific but no surface specified, use average
                return np.mean(list(self.ratings[player_name].values()))
            return self.ratings[player_name]

    def _set_rating(self, player_name: str, rating: float,
                   surface: Optional[str] = None):
        """internal method to set rating"""
        if self.surface_specific and surface:
            if player_name not in self.ratings:
                self.ratings[player_name] = {}
            self.ratings[player_name][surface] = rating
        else:
            self.ratings[player_name] = rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        calculate expected score for player a vs player b

        uses standard elo formula: expected = 1 / (1 + 10^((rb - ra) / 400))

        args:
            rating_a: elo rating of player a
            rating_b: elo rating of player b

        returns:
            expected score for player a (0 to 1)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, winner: str, loser: str,
                      surface: Optional[str] = None,
                      winner_rating: Optional[float] = None,
                      loser_rating: Optional[float] = None) -> Tuple[float, float]:
        """
        update elo ratings after a match

        args:
            winner: name of winning player
            loser: name of losing player
            surface: surface type (if surface_specific=true)
            winner_rating: override current winner rating (for initialization)
            loser_rating: override current loser rating (for initialization)

        returns:
            tuple of (new_winner_rating, new_loser_rating)
        """
        # get current ratings
        if winner_rating is None:
            winner_rating = self.get_rating(winner, surface)
        if loser_rating is None:
            loser_rating = self.get_rating(loser, surface)

        # calculate expected scores
        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = self.expected_score(loser_rating, winner_rating)

        # actual scores (winner gets 1, loser gets 0)
        actual_winner = 1.0
        actual_loser = 0.0

        # update ratings using elo formula
        new_winner_rating = winner_rating + self.k_factor * (actual_winner - expected_winner)
        new_loser_rating = loser_rating + self.k_factor * (actual_loser - expected_loser)

        # store updated ratings
        self._set_rating(winner, new_winner_rating, surface)
        self._set_rating(loser, new_loser_rating, surface)

        return new_winner_rating, new_loser_rating

    def build_from_matches(self, match_data: pd.DataFrame,
                          date_column: str = 'tourney_date',
                          verbose: bool = True) -> Dict:
        """
        build elo ratings from historical match data

        processes matches chronologically to compute evolving ratings

        args:
            match_data: dataframe with match results
                       must have: winner_name, loser_name, surface (optional)
            date_column: column name for match date (for chronological ordering)
            verbose: print progress

        returns:
            dictionary with final ratings
        """
        # ensure data is sorted chronologically
        if date_column in match_data.columns:
            match_data = match_data.sort_values(date_column).copy()

        total_matches = len(match_data)

        if verbose:
            print(f"building elo ratings from {total_matches} matches...")

        # process each match
        for idx, match in match_data.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']

            # get surface if available and needed
            surface = None
            if self.surface_specific and 'surface' in match:
                surface = match['surface']

            # update ratings
            self.update_ratings(winner, loser, surface)

            # periodic progress
            if verbose and (idx + 1) % 5000 == 0:
                print(f"  processed {idx + 1}/{total_matches} matches...")

        if verbose:
            print(f"elo ratings built for {len(self.ratings)} players")

            # show top 10 players
            if self.surface_specific:
                # average across surfaces
                avg_ratings = {
                    player: np.mean(list(surfaces.values())) if isinstance(surfaces, dict) else surfaces
                    for player, surfaces in self.ratings.items()
                }
            else:
                avg_ratings = self.ratings

            top_players = sorted(avg_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\ntop 10 players by elo rating:")
            for rank, (player, rating) in enumerate(top_players, 1):
                print(f"  {rank}. {player}: {rating:.0f}")

        return self.ratings

    def get_win_probability(self, player_a: str, player_b: str,
                           surface: Optional[str] = None) -> float:
        """
        calculate win probability for player a vs player b

        args:
            player_a: first player name
            player_b: second player name
            surface: surface type (optional)

        returns:
            probability that player a wins (0 to 1)
        """
        rating_a = self.get_rating(player_a, surface)
        rating_b = self.get_rating(player_b, surface)

        return self.expected_score(rating_a, rating_b)

    def get_rating_differential(self, player_a: str, player_b: str,
                               surface: Optional[str] = None) -> float:
        """
        get elo rating difference between two players

        args:
            player_a: first player name
            player_b: second player name
            surface: surface type (optional)

        returns:
            rating_a - rating_b (positive means a is higher rated)
        """
        rating_a = self.get_rating(player_a, surface)
        rating_b = self.get_rating(player_b, surface)

        return rating_a - rating_b

    def save_ratings(self, filepath: str):
        """save elo ratings to json file"""
        import json

        with open(filepath, 'w') as f:
            json.dump({
                'ratings': self.ratings,
                'k_factor': self.k_factor,
                'initial_rating': self.initial_rating,
                'surface_specific': self.surface_specific
            }, f, indent=2)

    def load_ratings(self, filepath: str):
        """load elo ratings from json file"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)
            self.ratings = data['ratings']
            self.k_factor = data['k_factor']
            self.initial_rating = data['initial_rating']
            self.surface_specific = data['surface_specific']


def calculate_skill_adjustment(elo_diff: float, max_adjustment: float = 0.15) -> float:
    """
    convert elo difference to probability adjustment

    uses sigmoid-like scaling to map elo diff to probability boost

    args:
        elo_diff: rating_a - rating_b (positive = player a is stronger)
        max_adjustment: maximum probability adjustment (cap)

    returns:
        probability adjustment to apply (-max_adjustment to +max_adjustment)
    """
    # scale elo difference
    # 200 elo points ≈ 75% expected win rate
    # we want smooth scaling from -max to +max

    # use tanh for smooth s-curve
    # scale factor: 400 elo points → ±max_adjustment
    scale_factor = 400.0 / max_adjustment

    adjustment = max_adjustment * np.tanh(elo_diff / scale_factor)

    return adjustment
