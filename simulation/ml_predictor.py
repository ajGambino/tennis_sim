"""
ml-powered match predictor

uses trained xgboost model to predict match outcomes
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime

from simulation.player_stats import PlayerStats
from simulation.elo_system import EloSystem
from simulation.feature_engineering import FeatureEngineer


class MLMatchPredictor:
    """predicts match outcomes using ml model"""

    def __init__(self, model_path: str = 'models/match_predictor_xgb.pkl'):
        """
        initialize ml predictor

        args:
            model_path: path to trained model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.metadata = None

        self._load_model()

    def _load_model(self):
        """load trained model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"model not found at {self.model_path}. "
                "run training/train_point_model.py first"
            )

        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['features']
            self.metadata = {
                'train_metrics': data.get('train_metrics'),
                'val_metrics': data.get('val_metrics'),
                'test_metrics': data.get('test_metrics'),
                'trained_date': data.get('trained_date')
            }

        print(f"loaded ml model from {self.model_path}")
        if self.metadata.get('test_metrics'):
            print(f"  test accuracy: {self.metadata['test_metrics']['accuracy']:.3f}")
            print(f"  test log loss: {self.metadata['test_metrics']['log_loss']:.3f}")

    def predict_match(self, player_a_stats: PlayerStats,
                     player_b_stats: PlayerStats,
                     elo_a: float, elo_b: float,
                     surface: str = 'hard') -> Dict[str, float]:
        """
        predict match outcome using ml model

        args:
            player_a_stats: stats for player a
            player_b_stats: stats for player b
            elo_a: elo rating for player a
            elo_b: elo rating for player b
            surface: court surface

        returns:
            dict with win probabilities and confidence
        """
        # create feature vector
        features = self._create_features(
            player_a_stats, player_b_stats, elo_a, elo_b, surface
        )

        # get prediction
        win_prob_a = self.model.predict_proba(features)[0, 1]

        return {
            'player_a_win_prob': win_prob_a,
            'player_b_win_prob': 1 - win_prob_a,
            'confidence': abs(win_prob_a - 0.5) * 2,  # 0 = toss-up, 1 = very confident
            'model': 'xgboost'
        }

    def _create_features(self, player_a_stats: PlayerStats,
                        player_b_stats: PlayerStats,
                        elo_a: float, elo_b: float,
                        surface: str) -> pd.DataFrame:
        """
        create feature vector from player stats

        args:
            player_a_stats: player a statistics
            player_b_stats: player b statistics
            elo_a: player a elo rating
            elo_b: player b elo rating
            surface: court surface

        returns:
            dataframe with features (single row)
        """
        features = {
            # player a serve stats
            'a_first_serve_pct': player_a_stats.first_serve_pct,
            'a_ace_pct': player_a_stats.ace_pct,
            'a_df_pct': player_a_stats.df_pct,
            'a_first_serve_win_pct': player_a_stats.first_serve_win_pct,
            'a_second_serve_win_pct': player_a_stats.second_serve_win_pct,
            'a_return_first_win_pct': player_a_stats.return_first_win_pct,
            'a_return_second_win_pct': player_a_stats.return_second_win_pct,

            # player b serve stats
            'b_first_serve_pct': player_b_stats.first_serve_pct,
            'b_ace_pct': player_b_stats.ace_pct,
            'b_df_pct': player_b_stats.df_pct,
            'b_first_serve_win_pct': player_b_stats.first_serve_win_pct,
            'b_second_serve_win_pct': player_b_stats.second_serve_win_pct,
            'b_return_first_win_pct': player_b_stats.return_first_win_pct,
            'b_return_second_win_pct': player_b_stats.return_second_win_pct,

            # win rates (use 0.5 as default since we don't have recent data)
            'a_win_rate': 0.5,
            'b_win_rate': 0.5,

            # recent form (use defaults)
            'a_recent_win_pct': 0.5,
            'b_recent_win_pct': 0.5,
            'a_days_since_last': 7,  # assume recent match
            'b_days_since_last': 7,

            # head to head (use default)
            'h2h_win_pct_a': 0.5,
            'h2h_matches': 0,

            # elo
            'elo_a': elo_a,
            'elo_b': elo_b,
            'elo_diff': elo_a - elo_b,

            # surface encoding
            'surface_hard': 1 if surface == 'hard' else 0,
            'surface_clay': 1 if surface == 'clay' else 0,
            'surface_grass': 1 if surface == 'grass' else 0,
        }

        # ensure features are in correct order
        feature_df = pd.DataFrame([features])
        return feature_df[self.feature_names]

    def compare_with_elo(self, player_a_stats: PlayerStats,
                        player_b_stats: PlayerStats,
                        elo_a: float, elo_b: float,
                        surface: str = 'hard') -> Dict[str, float]:
        """
        compare ml prediction with elo-based prediction

        args:
            player_a_stats: stats for player a
            player_b_stats: stats for player b
            elo_a: elo rating for player a
            elo_b: elo rating for player b
            surface: court surface

        returns:
            dict with both predictions
        """
        # ml prediction
        ml_pred = self.predict_match(
            player_a_stats, player_b_stats, elo_a, elo_b, surface
        )

        # elo prediction (standard elo formula)
        elo_win_prob_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

        return {
            'ml_win_prob_a': ml_pred['player_a_win_prob'],
            'elo_win_prob_a': elo_win_prob_a,
            'difference': ml_pred['player_a_win_prob'] - elo_win_prob_a,
            'ml_confidence': ml_pred['confidence']
        }


def predict_match_from_names(player_a: str, player_b: str,
                            surface: str = 'hard',
                            tour: str = 'atp',
                            model_path: str = 'models/match_predictor_xgb.pkl',
                            elo_path: Optional[str] = None) -> Dict:
    """
    convenience function to predict from player names

    args:
        player_a: first player name
        player_b: second player name
        surface: court surface
        tour: 'atp' or 'wta'
        model_path: path to ml model
        elo_path: path to elo ratings (optional)

    returns:
        dict with prediction results
    """
    from simulation.data_loader import load_match_data
    from simulation.player_stats import PlayerStatsCalculator

    # load data
    match_data = load_match_data(tour=tour)

    # get player stats
    stats_calc = PlayerStatsCalculator(match_data)
    stats_a = stats_calc.get_player_stats(player_a, surface)
    stats_b = stats_calc.get_player_stats(player_b, surface)

    # get elo ratings
    if elo_path is None:
        elo_path = f'models/elo_ratings_{tour}.json'

    if os.path.exists(elo_path):
        elo_system = EloSystem()
        elo_system.load_ratings(elo_path)
        elo_a = elo_system.get_rating(player_a, surface)
        elo_b = elo_system.get_rating(player_b, surface)
    else:
        elo_a, elo_b = 1500, 1500

    # predict
    predictor = MLMatchPredictor(model_path)
    prediction = predictor.predict_match(stats_a, stats_b, elo_a, elo_b, surface)

    # add comparison with elo
    comparison = predictor.compare_with_elo(stats_a, stats_b, elo_a, elo_b, surface)

    return {
        'player_a': player_a,
        'player_b': player_b,
        'surface': surface,
        'elo_a': elo_a,
        'elo_b': elo_b,
        **prediction,
        **comparison
    }
