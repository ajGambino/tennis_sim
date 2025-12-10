"""
feature engineering for ml-based tennis prediction

extracts and computes features from historical match data for training ml models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class FeatureEngineer:
    """extracts ml features from match data"""

    def __init__(self, match_data: pd.DataFrame):
        """
        initialize feature engineer

        args:
            match_data: dataframe of historical matches
        """
        self.match_data = match_data.copy()
        self._prepare_data()

    def _prepare_data(self):
        """prepare match data with derived fields"""
        # ensure date column is datetime
        if 'tourney_date' in self.match_data.columns:
            # only convert if not already datetime
            if not pd.api.types.is_datetime64_any_dtype(self.match_data['tourney_date']):
                self.match_data['tourney_date'] = pd.to_datetime(
                    self.match_data['tourney_date'].astype(str),
                    format='%Y%m%d',
                    errors='coerce'
                )

        # calculate serve statistics
        self._calculate_serve_stats()

    def _calculate_serve_stats(self):
        """calculate derived serve statistics"""
        # winner stats
        self.match_data['w_first_serve_pct'] = (
            self.match_data['w_1stIn'] / self.match_data['w_svpt']
        )
        self.match_data['w_first_serve_win_pct'] = (
            self.match_data['w_1stWon'] / self.match_data['w_1stIn']
        )
        self.match_data['w_second_serve_win_pct'] = (
            self.match_data['w_2ndWon'] /
            (self.match_data['w_svpt'] - self.match_data['w_1stIn'])
        )
        self.match_data['w_ace_pct'] = (
            self.match_data['w_ace'] / self.match_data['w_svpt']
        )
        self.match_data['w_df_pct'] = (
            self.match_data['w_df'] /
            (self.match_data['w_svpt'] - self.match_data['w_1stIn'])
        )

        # loser stats
        self.match_data['l_first_serve_pct'] = (
            self.match_data['l_1stIn'] / self.match_data['l_svpt']
        )
        self.match_data['l_first_serve_win_pct'] = (
            self.match_data['l_1stWon'] / self.match_data['l_1stIn']
        )
        self.match_data['l_second_serve_win_pct'] = (
            self.match_data['l_2ndWon'] /
            (self.match_data['l_svpt'] - self.match_data['l_1stIn'])
        )
        self.match_data['l_ace_pct'] = (
            self.match_data['l_ace'] / self.match_data['l_svpt']
        )
        self.match_data['l_df_pct'] = (
            self.match_data['l_df'] /
            (self.match_data['l_svpt'] - self.match_data['l_1stIn'])
        )

        # replace inf/nan with 0
        self.match_data = self.match_data.replace([np.inf, -np.inf], np.nan)
        self.match_data = self.match_data.fillna(0).infer_objects(copy=False)

    def compute_rolling_stats(self, player_name: str,
                             date: datetime,
                             window: int = 10,
                             surface: Optional[str] = None) -> Dict[str, float]:
        """
        compute rolling average stats for a player

        args:
            player_name: player name
            date: reference date (compute stats before this date)
            window: number of recent matches to average
            surface: optional surface filter

        returns:
            dict with rolling average statistics
        """
        # get matches before reference date
        mask = (
            ((self.match_data['winner_name'] == player_name) |
             (self.match_data['loser_name'] == player_name)) &
            (self.match_data['tourney_date'] < date)
        )

        if surface:
            mask &= (self.match_data['surface'] == surface)

        recent_matches = self.match_data[mask].sort_values(
            'tourney_date', ascending=False
        ).head(window)

        if len(recent_matches) == 0:
            return self._default_stats()

        # separate wins and losses
        wins = recent_matches[recent_matches['winner_name'] == player_name]
        losses = recent_matches[recent_matches['loser_name'] == player_name]

        # aggregate stats
        stats = {}

        # serve stats (combine from w_ and l_ prefixes)
        total_svpt = wins['w_svpt'].sum() + losses['l_svpt'].sum()
        total_1st_in = wins['w_1stIn'].sum() + losses['l_1stIn'].sum()
        total_aces = wins['w_ace'].sum() + losses['l_ace'].sum()
        total_dfs = wins['w_df'].sum() + losses['l_df'].sum()
        total_1st_won = wins['w_1stWon'].sum() + losses['l_1stWon'].sum()
        total_2nd_won = wins['w_2ndWon'].sum() + losses['l_2ndWon'].sum()

        stats['first_serve_pct'] = total_1st_in / max(1, total_svpt)
        stats['ace_pct'] = total_aces / max(1, total_svpt)
        stats['df_pct'] = total_dfs / max(1, total_svpt - total_1st_in)
        stats['first_serve_win_pct'] = total_1st_won / max(1, total_1st_in)
        stats['second_serve_win_pct'] = total_2nd_won / max(1, total_svpt - total_1st_in)

        # return stats from opponent perspective
        opp_1st_in = wins['l_1stIn'].sum() + losses['w_1stIn'].sum()
        opp_svpt = wins['l_svpt'].sum() + losses['w_svpt'].sum()
        opp_1st_won = wins['l_1stWon'].sum() + losses['w_1stWon'].sum()
        opp_2nd_won = wins['l_2ndWon'].sum() + losses['w_2ndWon'].sum()

        stats['return_first_win_pct'] = 1.0 - (opp_1st_won / max(1, opp_1st_in))
        stats['return_second_win_pct'] = 1.0 - (opp_2nd_won / max(1, opp_svpt - opp_1st_in))

        # win rate
        stats['win_rate'] = len(wins) / max(1, len(recent_matches))

        # matches played
        stats['matches_played'] = len(recent_matches)

        return stats

    def compute_recent_form(self, player_name: str,
                          date: datetime,
                          days: int = 30) -> Dict[str, float]:
        """
        compute recent form metrics

        args:
            player_name: player name
            date: reference date
            days: number of days to look back

        returns:
            dict with form metrics
        """
        cutoff_date = date - timedelta(days=days)

        mask = (
            ((self.match_data['winner_name'] == player_name) |
             (self.match_data['loser_name'] == player_name)) &
            (self.match_data['tourney_date'] >= cutoff_date) &
            (self.match_data['tourney_date'] < date)
        )

        recent = self.match_data[mask]

        if len(recent) == 0:
            return {
                'recent_wins': 0,
                'recent_losses': 0,
                'recent_win_pct': 0.5,
                'matches_last_30d': 0,
                'days_since_last_match': 999
            }

        wins = (recent['winner_name'] == player_name).sum()
        losses = len(recent) - wins

        # days since last match
        if len(recent) > 0:
            last_match_date = recent['tourney_date'].max()
            days_since = (date - last_match_date).days
        else:
            days_since = 999

        return {
            'recent_wins': wins,
            'recent_losses': losses,
            'recent_win_pct': wins / max(1, len(recent)),
            'matches_last_30d': len(recent),
            'days_since_last_match': min(days_since, 999)
        }

    def compute_head_to_head(self, player_a: str, player_b: str,
                            date: datetime) -> Dict[str, float]:
        """
        compute head-to-head statistics

        args:
            player_a: first player
            player_b: second player
            date: reference date

        returns:
            dict with h2h stats
        """
        mask = (
            (((self.match_data['winner_name'] == player_a) &
              (self.match_data['loser_name'] == player_b)) |
             ((self.match_data['winner_name'] == player_b) &
              (self.match_data['loser_name'] == player_a))) &
            (self.match_data['tourney_date'] < date)
        )

        h2h_matches = self.match_data[mask]

        if len(h2h_matches) == 0:
            return {
                'h2h_wins_a': 0,
                'h2h_wins_b': 0,
                'h2h_matches': 0,
                'h2h_win_pct_a': 0.5
            }

        wins_a = (h2h_matches['winner_name'] == player_a).sum()
        wins_b = len(h2h_matches) - wins_a

        return {
            'h2h_wins_a': wins_a,
            'h2h_wins_b': wins_b,
            'h2h_matches': len(h2h_matches),
            'h2h_win_pct_a': wins_a / len(h2h_matches)
        }

    def _default_stats(self) -> Dict[str, float]:
        """return default stats when no data available"""
        return {
            'first_serve_pct': 0.62,
            'ace_pct': 0.06,
            'df_pct': 0.04,
            'first_serve_win_pct': 0.72,
            'second_serve_win_pct': 0.53,
            'return_first_win_pct': 0.28,
            'return_second_win_pct': 0.47,
            'win_rate': 0.5,
            'matches_played': 0
        }

    def create_match_features(self, player_a: str, player_b: str,
                             surface: str, date: datetime,
                             elo_a: float = 1500, elo_b: float = 1500) -> pd.DataFrame:
        """
        create feature vector for a match

        args:
            player_a: first player name
            player_b: second player name
            surface: court surface
            date: match date
            elo_a: elo rating for player a
            elo_b: elo rating for player b

        returns:
            dataframe with features (single row)
        """
        # rolling stats
        stats_a = self.compute_rolling_stats(player_a, date, window=20, surface=surface)
        stats_b = self.compute_rolling_stats(player_b, date, window=20, surface=surface)

        # recent form
        form_a = self.compute_recent_form(player_a, date, days=30)
        form_b = self.compute_recent_form(player_b, date, days=30)

        # head to head
        h2h = self.compute_head_to_head(player_a, player_b, date)

        # combine features
        features = {
            # player a serve stats
            'a_first_serve_pct': stats_a['first_serve_pct'],
            'a_ace_pct': stats_a['ace_pct'],
            'a_df_pct': stats_a['df_pct'],
            'a_first_serve_win_pct': stats_a['first_serve_win_pct'],
            'a_second_serve_win_pct': stats_a['second_serve_win_pct'],
            'a_return_first_win_pct': stats_a['return_first_win_pct'],
            'a_return_second_win_pct': stats_a['return_second_win_pct'],

            # player b serve stats
            'b_first_serve_pct': stats_b['first_serve_pct'],
            'b_ace_pct': stats_b['ace_pct'],
            'b_df_pct': stats_b['df_pct'],
            'b_first_serve_win_pct': stats_b['first_serve_win_pct'],
            'b_second_serve_win_pct': stats_b['second_serve_win_pct'],
            'b_return_first_win_pct': stats_b['return_first_win_pct'],
            'b_return_second_win_pct': stats_b['return_second_win_pct'],

            # win rates
            'a_win_rate': stats_a['win_rate'],
            'b_win_rate': stats_b['win_rate'],

            # recent form
            'a_recent_win_pct': form_a['recent_win_pct'],
            'b_recent_win_pct': form_b['recent_win_pct'],
            'a_days_since_last': form_a['days_since_last_match'],
            'b_days_since_last': form_b['days_since_last_match'],

            # head to head
            'h2h_win_pct_a': h2h['h2h_win_pct_a'],
            'h2h_matches': h2h['h2h_matches'],

            # elo
            'elo_a': elo_a,
            'elo_b': elo_b,
            'elo_diff': elo_a - elo_b,

            # surface encoding (one-hot)
            'surface_hard': 1 if surface == 'hard' else 0,
            'surface_clay': 1 if surface == 'clay' else 0,
            'surface_grass': 1 if surface == 'grass' else 0,
        }

        return pd.DataFrame([features])


def create_training_dataset(match_data: pd.DataFrame,
                           elo_ratings: Optional[Dict] = None,
                           min_date: Optional[datetime] = None,
                           max_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    create training dataset from historical matches

    args:
        match_data: historical match data
        elo_ratings: optional pre-computed elo ratings
        min_date: minimum date for training data
        max_date: maximum date for training data

    returns:
        dataframe with features and target (winner)
    """
    engineer = FeatureEngineer(match_data)

    # filter by date if specified
    data = engineer.match_data.copy()  # use preprocessed data with datetime
    if min_date:
        data = data[data['tourney_date'] >= min_date]
    if max_date:
        data = data[data['tourney_date'] <= max_date]

    # only keep matches with complete stats
    required_cols = ['w_svpt', 'l_svpt', 'w_1stIn', 'l_1stIn',
                    'w_1stWon', 'l_1stWon', 'w_2ndWon', 'l_2ndWon']
    for col in required_cols:
        data = data[data[col].notna()]
        data = data[data[col] > 0]

    print(f"creating training dataset from {len(data)} matches...")

    all_features = []
    all_targets = []

    for idx, match in data.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  processed {idx + 1}/{len(data)} matches...")

        winner = match['winner_name']
        loser = match['loser_name']
        surface = match.get('surface', 'hard')
        date = match['tourney_date']

        # get elo ratings if available
        if elo_ratings:
            elo_w = elo_ratings.get(winner, {}).get(surface, 1500) if isinstance(elo_ratings.get(winner), dict) else 1500
            elo_l = elo_ratings.get(loser, {}).get(surface, 1500) if isinstance(elo_ratings.get(loser), dict) else 1500
        else:
            elo_w = 1500
            elo_l = 1500

        # randomly assign winner/loser to player a/b to avoid ordering bias
        # this prevents feature duplication and label leakage
        if np.random.random() < 0.5:
            # winner is player a
            features = engineer.create_match_features(
                winner, loser, surface, date, elo_w, elo_l
            )
            all_features.append(features)
            all_targets.append(1)  # player a won
        else:
            # loser is player a
            features = engineer.create_match_features(
                loser, winner, surface, date, elo_l, elo_w
            )
            all_features.append(features)
            all_targets.append(0)  # player a lost

    # combine
    X = pd.concat(all_features, ignore_index=True)
    y = pd.Series(all_targets)

    print(f"training dataset created: {len(X)} samples")

    return X, y
