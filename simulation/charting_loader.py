"""
match charting project data loader

loads and parses shot-by-shot data from jeff sackmann's match charting project
provides player-specific shot patterns and rally characteristics
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from simulation.shot import (
    ServePlacement, ShotType, ShotDirection, ShotOutcome,
    ChartingNotation
)


class ChartingDataLoader:
    """
    loads and parses match charting project data

    the match charting project contains shot-by-shot data for ~17000 atp matches
    and ~5000 wta matches with detailed rally sequences
    """

    def __init__(self, data_dir: str = 'data/charting'):
        """
        initialize charting data loader

        args:
            data_dir: directory containing charting csv files
        """
        self.data_dir = data_dir
        self.charting_data = None
        self.serve_patterns = None
        self.rally_patterns = None

    def load_charting_data(self, tour: str = 'atp') -> pd.DataFrame:
        """
        load match charting project csv files

        mcp data is split into decade files:
        - charting-m-points-to-2009.csv
        - charting-m-points-2010s.csv
        - charting-m-points-2020s.csv

        args:
            tour: 'atp' or 'wta'

        returns:
            dataframe with point-by-point charting data
        """
        # match charting project file naming (by decade)
        if tour.lower() == 'atp':
            prefix = 'charting-m-points'
        else:
            prefix = 'charting-w-points'

        # try to load decade files
        decade_files = [
            f'{prefix}-to-2009.csv',
            f'{prefix}-2010s.csv',
            f'{prefix}-2020s.csv'
        ]

        all_data = []
        files_loaded = []

        for filename in decade_files:
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                    all_data.append(df)
                    files_loaded.append(filename)
                    print(f"loaded {len(df):,} points from {filename}")
                except Exception as e:
                    print(f"warning: failed to load {filename}: {e}")

        # if no decade files found, try legacy single file
        if not all_data:
            legacy_file = f'{prefix}.csv'
            filepath = os.path.join(self.data_dir, legacy_file)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                    all_data.append(df)
                    files_loaded.append(legacy_file)
                    print(f"loaded {len(df):,} points from {legacy_file}")
                except Exception as e:
                    print(f"error loading {legacy_file}: {e}")

        # if still no data, use demo
        if not all_data:
            print(f"warning: no match charting data found in {self.data_dir}")
            print("download from: https://github.com/JeffSackmann/tennis_MatchChartingProject")
            return self._create_demo_charting_data()

        # combine all loaded data
        df = pd.concat(all_data, ignore_index=True)
        print(f"total: {len(df):,} points loaded from {len(files_loaded)} files")

        # load matches file to get player names
        print("loading matches data to get player names...")
        matches_file = f'charting-m-matches.csv' if tour.lower() == 'atp' else 'charting-w-matches.csv'
        matches_path = os.path.join(self.data_dir, matches_file)

        if os.path.exists(matches_path):
            try:
                matches_df = pd.read_csv(matches_path, encoding='utf-8')
                print(f"loaded {len(matches_df):,} matches")

                # merge points with matches to get player names
                df = df.merge(
                    matches_df[['match_id', 'Player 1', 'Player 2']],
                    on='match_id',
                    how='left'
                )

                # add server name column based on Svr indicator
                df['server'] = df.apply(
                    lambda row: row['Player 1'] if row['Svr'] == 1 else row['Player 2']
                    if pd.notna(row.get('Svr')) else None,
                    axis=1
                )

                # add returner name column
                df['returner'] = df.apply(
                    lambda row: row['Player 2'] if row['Svr'] == 1 else row['Player 1']
                    if pd.notna(row.get('Svr')) else None,
                    axis=1
                )

                print(f"merged player names for {df['server'].notna().sum():,} points")

            except Exception as e:
                print(f"warning: failed to load matches file: {e}")
        else:
            print(f"warning: matches file not found at {matches_path}")

        # clean and standardize
        self._clean_charting_data(df)

        self.charting_data = df
        return df

    def _clean_charting_data(self, df: pd.DataFrame):
        """clean and standardize charting data"""
        # validation removed - MCP data format uses 'server', 'returner', '1st', '2nd' columns
        pass

    def _create_demo_charting_data(self) -> pd.DataFrame:
        """
        create synthetic demo charting data for testing

        uses realistic shot patterns when actual mcp data is unavailable
        """
        print("creating synthetic charting demo data")

        # sample players
        players = [
            'Novak Djokovic', 'Rafael Nadal', 'Roger Federer',
            'Carlos Alcaraz', 'Jannik Sinner', 'Daniil Medvedev'
        ]

        # generate 1000 demo points
        n_points = 1000
        np.random.seed(42)

        data = []

        for i in range(n_points):
            server = np.random.choice(players)
            returner = np.random.choice([p for p in players if p != server])

            # generate realistic rally
            rally_length = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                           p=[0.1, 0.15, 0.15, 0.15, 0.12, 0.1, 0.08, 0.07, 0.05, 0.03])

            # generate serve placement
            serve_placement = np.random.choice(['W', 'T', 'B'], p=[0.35, 0.40, 0.25])

            # simple point outcome
            is_ace = rally_length == 1 and np.random.random() < 0.5
            server_won = np.random.random() < 0.65  # server advantage

            data.append({
                'match_id': f'demo_{i // 100}',
                'player': server,
                'opponent': returner,
                'point': i,
                'serve_placement': serve_placement,
                'rally_length': rally_length,
                'is_ace': is_ace,
                'server_won': server_won,
            })

        df = pd.DataFrame(data)
        print(f"created {len(df)} demo charting points")
        return df

    def _parse_serve_placement(self, notation: str) -> Optional[str]:
        """
        parse serve placement from mcp notation

        mcp serve codes (first character):
        - 4 or 1 = wide
        - 5 or 2 = body
        - 6 or 3 = T (down the middle)

        args:
            notation: point notation string (e.g., "4f2f1*")

        returns:
            'W', 'T', or 'B', or none if invalid
        """
        if pd.isna(notation) or len(str(notation)) == 0:
            return None

        first_char = str(notation)[0]

        # map mcp codes to placement
        if first_char in ['4', '1']:  # wide (deuce or ad court)
            return 'W'
        elif first_char in ['5', '2']:  # body
            return 'B'
        elif first_char in ['6', '3']:  # T
            return 'T'

        return None

    def _is_ace(self, notation: str) -> bool:
        """
        check if serve was an ace from notation

        ace notation: serve code followed immediately by #
        example: "4#" = ace wide

        args:
            notation: point notation string

        returns:
            true if ace, false otherwise
        """
        if pd.isna(notation) or len(str(notation)) < 2:
            return False

        notation = str(notation)

        # ace is serve code + # with nothing in between
        return len(notation) == 2 and notation[1] == '#'

    def _is_fault(self, notation: str) -> bool:
        """
        check if serve was a fault from notation

        fault indicators (second character):
        - x = serve fault (generic)
        - w = wide fault
        - n = net fault
        - d = deep fault (long)

        args:
            notation: point notation string

        returns:
            true if fault, false otherwise
        """
        if pd.isna(notation) or len(str(notation)) < 2:
            return False

        notation = str(notation)

        # fault is indicated by x, w, n, or d as second character
        return notation[1] in ['x', 'w', 'n', 'd']

    def _count_rally_length(self, notation: str) -> int:
        """
        count rally length from shot notation

        shot characters: f, b, s, r, v, l, o, z
        example: "4f2f1*" = 3 shots (serve, forehand, forehand winner)

        args:
            notation: point notation string

        returns:
            number of shots in rally
        """
        if pd.isna(notation) or len(str(notation)) == 0:
            return 0

        notation = str(notation)

        # count shot type characters
        shot_chars = ['f', 'b', 's', 'r', 'v', 'l', 'o', 'z']
        rally_length = 1  # start with 1 for serve

        for char in notation[1:]:  # skip first char (serve code)
            if char.lower() in shot_chars:
                rally_length += 1

        return rally_length

    def extract_serve_patterns(self, min_points: int = 50) -> Dict[str, Dict]:
        """
        extract serve placement patterns for each player from mcp data

        parses serve notation from '1st' and '2nd' columns to extract:
        - serve placement (wide/T/body)
        - ace rates by placement
        - total serves

        args:
            min_points: minimum serve points required for player to be included

        returns:
            dict mapping player name to serve pattern statistics
            {
                'player_name': {
                    'wide_pct': 0.35,
                    'T_pct': 0.40,
                    'body_pct': 0.25,
                    'ace_pct_by_placement': {'W': 0.08, 'T': 0.12, 'B': 0.05},
                    'total_serves': 500
                }
            }
        """
        if self.charting_data is None:
            print("no charting data loaded - call load_charting_data() first")
            return {}

        # check for server column (should be added in load_charting_data)
        if 'server' not in self.charting_data.columns:
            print("warning: 'server' column not found - ensure matches data was merged")
            return {}

        serve_patterns = {}
        print(f"analyzing serve patterns from {len(self.charting_data):,} points...")

        # process each row to extract serve info
        serve_data = []

        for _, row in self.charting_data.iterrows():
            server = row.get('server')
            if pd.isna(server):
                continue

            # check first serve
            first_serve = row.get('1st')
            if not pd.isna(first_serve):
                placement = self._parse_serve_placement(first_serve)
                is_fault = self._is_fault(first_serve)
                is_ace = self._is_ace(first_serve)

                if placement and not is_fault:  # valid serve in play
                    serve_data.append({
                        'player': server,
                        'placement': placement,
                        'is_ace': is_ace,
                        'is_second': False
                    })
                elif is_fault:
                    # first serve fault - check second serve
                    second_serve = row.get('2nd')
                    if not pd.isna(second_serve):
                        placement_2nd = self._parse_serve_placement(second_serve)
                        is_fault_2nd = self._is_fault(second_serve)
                        is_ace_2nd = self._is_ace(second_serve)

                        if placement_2nd and not is_fault_2nd:
                            serve_data.append({
                                'player': server,
                                'placement': placement_2nd,
                                'is_ace': is_ace_2nd,
                                'is_second': True
                            })

        # convert to dataframe for easier analysis
        serves_df = pd.DataFrame(serve_data)

        if len(serves_df) == 0:
            print("warning: no serves extracted from charting data")
            return {}

        print(f"extracted {len(serves_df):,} serves from point notation")

        # group by player and calculate patterns
        for player in serves_df['player'].unique():
            player_serves = serves_df[serves_df['player'] == player]

            if len(player_serves) < min_points:
                continue

            # count placements
            placement_counts = player_serves['placement'].value_counts()
            total_serves = len(player_serves)

            # calculate placement percentages
            wide_pct = placement_counts.get('W', 0) / total_serves
            t_pct = placement_counts.get('T', 0) / total_serves
            body_pct = placement_counts.get('B', 0) / total_serves

            # calculate ace rates by placement
            ace_by_placement = {}
            for placement in ['W', 'T', 'B']:
                placement_serves = player_serves[player_serves['placement'] == placement]
                if len(placement_serves) > 0:
                    ace_by_placement[placement] = placement_serves['is_ace'].sum() / len(placement_serves)
                else:
                    ace_by_placement[placement] = 0.0

            serve_patterns[player] = {
                'wide_pct': wide_pct,
                'T_pct': t_pct,
                'body_pct': body_pct,
                'ace_pct_by_placement': ace_by_placement,
                'total_serves': total_serves,
                'overall_ace_pct': player_serves['is_ace'].sum() / total_serves
            }

        self.serve_patterns = serve_patterns
        print(f"extracted serve patterns for {len(serve_patterns)} players")
        return serve_patterns

    def _infer_serve_placements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        infer serve placements from point notation column

        the match charting project uses notation like '4f1b2@' where
        the first character indicates serve placement (1-6)
        """
        df = df.copy()

        if 'point' in df.columns:
            # parse first character of point notation
            df['serve_placement'] = df['point'].astype(str).str[0].map(
                ChartingNotation.SERVE_PLACEMENTS
            )

            # convert enum to string
            df['serve_placement'] = df['serve_placement'].apply(
                lambda x: x.value if x else 'T'
            )
        else:
            # default to T (center) if no point notation
            df['serve_placement'] = 'T'

        return df

    def extract_rally_patterns(self, min_rallies: int = 30) -> Dict[str, Dict]:
        """
        extract rally characteristics for each player from mcp data

        parses rally lengths from shot notation in '1st' and '2nd' columns

        args:
            min_rallies: minimum rallies required for player to be included

        returns:
            dict mapping player name to rally statistics
            {
                'player_name': {
                    'avg_rally_length': 4.2,
                    'rally_length_dist': [0.05, 0.15, 0.20, ...],  # pmf
                    'total_rallies': 300
                }
            }
        """
        if self.charting_data is None:
            print("no charting data loaded")
            return {}

        # check for server column
        if 'server' not in self.charting_data.columns:
            print("warning: 'server' column not found - ensure matches data was merged")
            return {}

        rally_patterns = {}
        print(f"analyzing rally patterns from {len(self.charting_data):,} points...")

        # parse rally lengths from notation
        rally_data = []

        for _, row in self.charting_data.iterrows():
            server = row.get('server')
            returner = row.get('returner')

            # parse rally length from first serve notation
            first_serve = row.get('1st')
            if not pd.isna(first_serve):
                rally_length = self._count_rally_length(first_serve)

                # add rally for both players (both participate in each rally)
                if not pd.isna(server) and rally_length > 0:
                    rally_data.append({'player': server, 'rally_length': rally_length})
                if not pd.isna(returner) and rally_length > 0:
                    rally_data.append({'player': returner, 'rally_length': rally_length})

        # convert to dataframe
        rallies_df = pd.DataFrame(rally_data)

        if len(rallies_df) == 0:
            print("warning: no rallies extracted from charting data")
            return {}

        print(f"extracted {len(rallies_df):,} rally lengths from point notation")

        # group by player and compute patterns
        for player in rallies_df['player'].unique():
            player_rallies = rallies_df[rallies_df['player'] == player]

            if len(player_rallies) < min_rallies:
                continue

            # get rally lengths
            rally_lengths = player_rallies['rally_length'].values

            # compute rally length distribution
            rally_dist = np.zeros(21)  # support rallies up to 20 shots
            for length in rally_lengths:
                if 0 < length <= 20:
                    rally_dist[int(length)] += 1

            # normalize to probability mass function
            if rally_dist.sum() > 0:
                rally_dist = rally_dist / rally_dist.sum()

            rally_patterns[player] = {
                'avg_rally_length': float(np.mean(rally_lengths)),
                'rally_length_dist': rally_dist.tolist(),
                'total_rallies': len(player_rallies)
            }

        self.rally_patterns = rally_patterns
        print(f"extracted rally patterns for {len(rally_patterns)} players")
        return rally_patterns

    def get_player_serve_pattern(self, player_name: str) -> Optional[Dict]:
        """
        get serve pattern for a specific player

        args:
            player_name: player name

        returns:
            serve pattern dict or none if not found
        """
        if self.serve_patterns is None:
            self.extract_serve_patterns()

        # exact match
        if player_name in self.serve_patterns:
            return self.serve_patterns[player_name]

        # try case-insensitive match
        for name, pattern in self.serve_patterns.items():
            if name.lower() == player_name.lower():
                return pattern

        return None

    def get_player_rally_pattern(self, player_name: str) -> Optional[Dict]:
        """
        get rally pattern for a specific player

        args:
            player_name: player name

        returns:
            rally pattern dict or none if not found
        """
        if self.rally_patterns is None:
            self.extract_rally_patterns()

        # exact match
        if player_name in self.rally_patterns:
            return self.rally_patterns[player_name]

        # try case-insensitive match
        for name, pattern in self.rally_patterns.items():
            if name.lower() == player_name.lower():
                return pattern

        return None

    def get_default_serve_pattern(self) -> Dict:
        """
        get default serve pattern for unknown players

        returns average patterns across all players
        """
        if self.serve_patterns and len(self.serve_patterns) > 0:
            # compute average across all players
            all_patterns = list(self.serve_patterns.values())

            return {
                'wide_pct': np.mean([p['wide_pct'] for p in all_patterns]),
                'T_pct': np.mean([p['T_pct'] for p in all_patterns]),
                'body_pct': np.mean([p['body_pct'] for p in all_patterns]),
                'ace_pct_by_placement': {
                    'W': np.mean([p['ace_pct_by_placement'].get('W', 0.05) for p in all_patterns]),
                    'T': np.mean([p['ace_pct_by_placement'].get('T', 0.08) for p in all_patterns]),
                    'B': np.mean([p['ace_pct_by_placement'].get('B', 0.03) for p in all_patterns]),
                },
                'overall_ace_pct': np.mean([p['overall_ace_pct'] for p in all_patterns])
            }
        else:
            # hardcoded reasonable defaults
            return {
                'wide_pct': 0.35,
                'T_pct': 0.40,
                'body_pct': 0.25,
                'ace_pct_by_placement': {'W': 0.06, 'T': 0.09, 'B': 0.03},
                'overall_ace_pct': 0.06
            }

    def get_default_rally_pattern(self) -> Dict:
        """
        get default rally pattern for unknown players

        returns average patterns across all players
        """
        if self.rally_patterns and len(self.rally_patterns) > 0:
            all_patterns = list(self.rally_patterns.values())

            avg_rally_length = np.mean([p['avg_rally_length'] for p in all_patterns])

            # average rally length distribution
            all_dists = [p['rally_length_dist'] for p in all_patterns]
            avg_dist = np.mean(all_dists, axis=0)

            return {
                'avg_rally_length': avg_rally_length,
                'rally_length_dist': avg_dist.tolist()
            }
        else:
            # realistic default distribution
            dist = np.array([0, 0.08, 0.12, 0.15, 0.15, 0.13, 0.11, 0.09, 0.08, 0.06, 0.03] + [0]*10)
            return {
                'avg_rally_length': 4.5,
                'rally_length_dist': dist.tolist()
            }


def load_charting_data(data_dir: str = 'data/charting', tour: str = 'atp') -> pd.DataFrame:
    """
    convenience function to load match charting project data

    args:
        data_dir: directory containing charting csv files
        tour: 'atp' or 'wta'

    returns:
        dataframe with charting data
    """
    loader = ChartingDataLoader(data_dir)
    return loader.load_charting_data(tour)
