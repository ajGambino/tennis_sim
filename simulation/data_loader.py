"""
data loader module

loads and preprocesses jeff sackmann's tennis_atp or tennis_wta match data
"""

import os
import pandas as pd
from typing import List, Optional


class DataLoader:
    """loads match data from sackmann csv files"""

    def __init__(self, data_dir: str = 'data'):
        """
        initialize data loader

        args:
            data_dir: directory containing csv files from tennis_atp or tennis_wta repos
        """
        self.data_dir = data_dir
        self.matches = None

    def load_match_data(self, years: Optional[List[int]] = None,
                       tour: str = 'atp') -> pd.DataFrame:
        """
        load match data from csv files

        args:
            years: list of years to load (e.g., [2020, 2021, 2022])
                  if none, attempts to load all available years
            tour: 'atp' or 'wta'

        returns:
            dataframe with all match data
        """
        all_matches = []

        # if no years specified, try common recent years
        if years is None:
            years = range(2015, 2025)

        for year in years:
            filename = f"{tour}_matches_{year}.csv"
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                    all_matches.append(df)
                    print(f"loaded {len(df)} matches from {filename}")
                except Exception as e:
                    print(f"warning: failed to load {filename}: {e}")

        if not all_matches:
            print(f"warning: no match data found in {self.data_dir}")
            print("creating synthetic demo data for testing")
            return self._create_synthetic_data()

        # combine all years
        self.matches = pd.concat(all_matches, ignore_index=True)

        # clean and standardize
        self._clean_data()

        print(f"total matches loaded: {len(self.matches)}")
        return self.matches

    def _clean_data(self):
        """clean and standardize match data"""
        if self.matches is None:
            return

        # standardize surface names
        if 'surface' in self.matches.columns:
            surface_map = {
                'Hard': 'hard',
                'Clay': 'clay',
                'Grass': 'grass',
                'Carpet': 'carpet'
            }
            self.matches['surface'] = self.matches['surface'].map(
                lambda x: surface_map.get(x, str(x).lower()) if pd.notna(x) else 'hard'
            )

        # ensure numeric columns are proper type
        stat_cols = [
            'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
            'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon'
        ]

        for col in stat_cols:
            if col in self.matches.columns:
                self.matches[col] = pd.to_numeric(self.matches[col], errors='coerce')

        # drop matches with no stats
        required_cols = ['w_svpt', 'l_svpt']
        for col in required_cols:
            if col in self.matches.columns:
                self.matches = self.matches[self.matches[col].notna()]

    def _create_synthetic_data(self) -> pd.DataFrame:
        """
        create synthetic demo data when real data is unavailable

        uses realistic tennis statistics for demonstration purposes
        """
        import numpy as np

        # sample player names
        players = [
            'Novak Djokovic', 'Carlos Alcaraz', 'Daniil Medvedev',
            'Jannik Sinner', 'Andrey Rublev', 'Stefanos Tsitsipas',
            'Holger Rune', 'Taylor Fritz', 'Casper Ruud', 'Alexander Zverev'
        ]

        surfaces = ['hard', 'clay', 'grass']

        # generate 500 synthetic matches
        n_matches = 500
        np.random.seed(42)

        data = {
            'tourney_date': [20230101 + i for i in range(n_matches)],
            'tourney_name': ['Demo Tournament'] * n_matches,
            'surface': np.random.choice(surfaces, n_matches),
            'winner_name': np.random.choice(players, n_matches),
            'loser_name': np.random.choice(players, n_matches),
        }

        # generate realistic serve statistics
        for prefix in ['w_', 'l_']:
            # serve points typically 60-120 per match
            svpt = np.random.randint(60, 120, n_matches)
            data[f'{prefix}svpt'] = svpt

            # first serve % typically 55-70%
            first_serve_pct = np.random.uniform(0.55, 0.70, n_matches)
            first_in = (svpt * first_serve_pct).astype(int)
            data[f'{prefix}1stIn'] = first_in

            # aces typically 3-15 per match
            data[f'{prefix}ace'] = np.random.randint(3, 15, n_matches)

            # double faults typically 1-5 per match
            data[f'{prefix}df'] = np.random.randint(1, 6, n_matches)

            # first serve win % typically 65-80%
            first_won_pct = np.random.uniform(0.65, 0.80, n_matches)
            data[f'{prefix}1stWon'] = (first_in * first_won_pct).astype(int)

            # second serve win % typically 45-60%
            second_pts = svpt - first_in
            second_won_pct = np.random.uniform(0.45, 0.60, n_matches)
            data[f'{prefix}2ndWon'] = (second_pts * second_won_pct).astype(int)

        df = pd.DataFrame(data)

        # ensure winner and loser are different
        mask = df['winner_name'] == df['loser_name']
        while mask.any():
            df.loc[mask, 'loser_name'] = np.random.choice(players, mask.sum())
            mask = df['winner_name'] == df['loser_name']

        print("synthetic data created with 500 demo matches")
        return df

    def get_player_matches(self, player_name: str,
                          surface: Optional[str] = None) -> pd.DataFrame:
        """
        get all matches for a specific player

        args:
            player_name: player name as it appears in data
            surface: optional surface filter ('hard', 'clay', 'grass')

        returns:
            dataframe with player's matches
        """
        if self.matches is None:
            raise ValueError("no data loaded - call load_match_data() first")

        # find matches where player won or lost
        player_matches = self.matches[
            (self.matches['winner_name'] == player_name) |
            (self.matches['loser_name'] == player_name)
        ].copy()

        if surface:
            player_matches = player_matches[
                player_matches['surface'] == surface.lower()
            ]

        return player_matches


def load_match_data(data_dir: str = 'data', years: Optional[List[int]] = None,
                   tour: str = 'atp') -> pd.DataFrame:
    """
    convenience function to load match data

    args:
        data_dir: directory containing csv files
        years: list of years to load
        tour: 'atp' or 'wta'

    returns:
        dataframe with match data
    """
    loader = DataLoader(data_dir)
    return loader.load_match_data(years, tour)
