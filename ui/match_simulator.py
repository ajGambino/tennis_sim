"""
Match Simulator UI - Page 1

Interactive interface for running tennis match simulations with configurable
parameters and real-time visualization of results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np
import pickle
import os

from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator
from simulation.match_engine import MatchSimulator
from simulation.point_engine import PointSimulator
from simulation.serve_model import ServeModel
from simulation.rally_model import RallyModel
from simulation.return_model import ReturnModel


@st.cache_data
def load_player_list():
    """Load list of available players from match data, sorted by Elo rating"""
    try:
        import json

        # Load Elo ratings
        elo_file = 'models/elo_ratings_atp.json'
        if os.path.exists(elo_file):
            with open(elo_file, 'r') as f:
                elo_data = json.load(f)
                ratings = elo_data.get('ratings', {})

                # Calculate average Elo for each player across surfaces
                player_avg_elos = {}
                for player, surfaces in ratings.items():
                    if isinstance(surfaces, dict):
                        avg_elo = np.mean(list(surfaces.values()))
                    else:
                        avg_elo = surfaces
                    player_avg_elos[player] = avg_elo

                # Sort players by Elo (highest to lowest)
                sorted_players = sorted(player_avg_elos.keys(),
                                       key=lambda p: player_avg_elos[p],
                                       reverse=True)
                return sorted_players

        # Fallback to loading from match data if Elo file doesn't exist
        loader = DataLoader(data_dir='data')
        matches = loader.load_match_data(years=[2020, 2021, 2022, 2023], tour='atp')
        if matches is None or len(matches) == 0:
            return ["Carlos Alcaraz", "Jannik Sinner", "Novak Djokovic", "Daniil Medvedev"]

        # Get unique player names
        winners = set(matches['winner_name'].unique())
        losers = set(matches['loser_name'].unique())
        all_players = sorted(list(winners | losers))
        return all_players
    except Exception as e:
        st.error(f"Error loading players: {e}")
        return ["Carlos Alcaraz", "Jannik Sinner"]


@st.cache_data
def load_trained_models():
    """Load trained shot-level models if available"""
    serve_patterns = {}
    rally_patterns = {}

    # Load serve patterns
    serve_pkl = 'models/serve_patterns.pkl'
    if os.path.exists(serve_pkl):
        with open(serve_pkl, 'rb') as f:
            data = pickle.load(f)
            serve_patterns = data.get('serve_patterns', {})

    # Load rally patterns
    rally_pkl = 'models/rally_patterns.pkl'
    if os.path.exists(rally_pkl):
        with open(rally_pkl, 'rb') as f:
            data = pickle.load(f)
            rally_patterns = data.get('rally_patterns', {})

    return serve_patterns, rally_patterns


def run_simulations(player_a, player_b, surface, best_of, num_sims, use_shot_sim):
    """Run match simulations and return results"""

    # Load data
    loader = DataLoader(data_dir='data')
    matches = loader.load_match_data(years=[2020, 2021, 2022, 2023], tour='atp')

    if matches is None or len(matches) == 0:
        st.warning("Using synthetic demo data")
        matches = loader._create_synthetic_data()

    # Get player stats
    stats_calc = PlayerStatsCalculator(matches)
    stats_a = stats_calc.get_player_stats(player_a, surface=surface)
    stats_b = stats_calc.get_player_stats(player_b, surface=surface)

    # Initialize results
    results = {
        'a_wins': 0,
        'b_wins': 0,
        'matches': [],
        'rally_lengths': [] if use_shot_sim else None
    }

    # Load shot-level models if needed
    if use_shot_sim:
        serve_patterns, rally_patterns = load_trained_models()
        serve_model = ServeModel(serve_patterns=serve_patterns)
        return_model = ReturnModel()
        rally_model = RallyModel(rally_patterns=rally_patterns)

    # Run simulations
    for i in range(num_sims):
        if use_shot_sim:
            # Shot-level simulation
            point_sim = PointSimulator(
                rng=np.random.default_rng(i),
                use_shot_simulation=True,
                serve_model=serve_model,
                return_model=return_model,
                rally_model=rally_model
            )

            sim = MatchSimulator(
                player_a_stats=stats_a,
                player_b_stats=stats_b,
                best_of=best_of,
                seed=i,
                point_simulator=point_sim
            )
        else:
            # Point-level simulation
            sim = MatchSimulator(
                player_a_stats=stats_a,
                player_b_stats=stats_b,
                best_of=best_of,
                seed=i
            )

        match_result = sim.simulate_match()

        # Record results
        if match_result.winner == player_a:
            results['a_wins'] += 1
        else:
            results['b_wins'] += 1

        # Store match details
        results['matches'].append({
            'match_id': i,
            'winner': match_result.winner,
            'score': match_result.score,
            'aces_a': match_result.aces_a,
            'aces_b': match_result.aces_b,
            'dfs_a': match_result.double_faults_a,
            'dfs_b': match_result.double_faults_b,
            'total_points': match_result.total_points
        })

    return results


def create_win_distribution_chart(results, player_a, player_b):
    """Create win probability bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=[player_a, player_b],
            y=[results['a_wins'], results['b_wins']],
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[f"{results['a_wins']}", f"{results['b_wins']}"],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Win Distribution",
        yaxis_title="Number of Wins",
        height=400,
        showlegend=False
    )

    return fig


def render():
    """Render the Match Simulator page"""

    st.markdown('<p class="main-header"> Match Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Run multiple match simulations and analyze results</p>', unsafe_allow_html=True)

    # Load players
    players = load_player_list()

    # Input controls
    col1, col2 = st.columns(2)

    with col1:
        default_a = players.index("Carlos Alcaraz") if "Carlos Alcaraz" in players else 0
        player_a = st.selectbox("Player A", players, index=default_a, key="player_a")

    with col2:
        default_b = players.index("Jannik Sinner") if "Jannik Sinner" in players else 1
        player_b = st.selectbox("Player B", players, index=default_b, key="player_b")

    # Surface and format
    col1, col2, col3 = st.columns(3)

    with col1:
        surface = st.selectbox("Surface", ["hard", "clay", "grass"], key="surface")

    with col2:
        best_of = st.selectbox("Format", [3, 5], format_func=lambda x: f"Best of {x}", key="best_of")

    with col3:
        use_shot_sim = st.checkbox("Use Shot-Level Simulation", value=True,
                                   help="More detailed but slower (~3x)")

    # Number of simulations
    num_sims = st.slider("Number of Simulations", min_value=10, max_value=1000, value=100, step=10)

    # Run button
    if st.button("🎲 Run Simulation", type="primary", width='stretch'):

        # Validation
        if player_a == player_b:
            st.error("Please select different players")
            return

        # Run simulations with progress bar
        with st.spinner(f"Simulating {num_sims} matches..."):
            progress_bar = st.progress(0)

            # Run simulation
            results = run_simulations(player_a, player_b, surface, best_of, num_sims, use_shot_sim)
            progress_bar.progress(100)

        st.success("✅ Simulation complete!")

        # Display results
        st.markdown("### Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            win_pct_a = results['a_wins'] / num_sims * 100
            st.metric(
                label=f"{player_a} Wins",
                value=f"{results['a_wins']}/{num_sims}",
                delta=f"{win_pct_a:.1f}%"
            )

        with col2:
            win_pct_b = results['b_wins'] / num_sims * 100
            st.metric(
                label=f"{player_b} Wins",
                value=f"{results['b_wins']}/{num_sims}",
                delta=f"{win_pct_b:.1f}%"
            )

        with col3:
            avg_points = np.mean([m['total_points'] for m in results['matches']])
            st.metric(
                label="Avg Match Length",
                value=f"{avg_points:.0f} points"
            )

        # Visualization
        st.markdown("### Visualization")

        # Win distribution chart
        fig = create_win_distribution_chart(results, player_a, player_b)
        st.plotly_chart(fig, width='stretch')

        # Detailed statistics
        with st.expander("📊 View Detailed Statistics"):
            df = pd.DataFrame(results['matches'])
            st.dataframe(df, width='stretch')

            # Summary stats
            st.markdown("#### Summary Statistics")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**{player_a}**")
                st.write(f"- Avg Aces: {df['aces_a'].mean():.1f}")
                st.write(f"- Avg Double Faults: {df['dfs_a'].mean():.1f}")
              

            with col2:
                st.write(f"**{player_b}**")
                st.write(f"- Avg Aces: {df['aces_b'].mean():.1f}")
                st.write(f"- Avg Double Faults: {df['dfs_b'].mean():.1f}")
               

            # Match length statistics
            st.markdown("#### Match Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Avg Points Per Match", f"{df['total_points'].mean():.1f}")
                st.metric("Min Points", f"{df['total_points'].min()}")
                st.metric("Max Points", f"{df['total_points'].max()}")

            with col2:
                st.metric("Median Points", f"{df['total_points'].median():.1f}")
                st.metric("Std Dev Points", f"{df['total_points'].std():.1f}")

            with col3:
                st.metric("Total Points Played", f"{df['total_points'].sum()}")
                st.metric("Avg Aces Per Match", f"{(df['aces_a'].mean() + df['aces_b'].mean()):.1f}")

            # Most common scores
            st.markdown("#### Most Common Match Scores")
            score_counts = df['score'].value_counts().head(10)

            col1, col2 = st.columns([2, 1])

            with col1:
                # Create bar chart of most common scores
                fig_scores = go.Figure(data=[
                    go.Bar(
                        x=score_counts.index,
                        y=score_counts.values,
                        marker_color='#636EFA',
                        text=score_counts.values,
                        textposition='auto'
                    )
                ])
                fig_scores.update_layout(
                    title="Top 10 Most Common Scores",
                    xaxis_title="Score",
                    yaxis_title="Frequency",
                    height=300
                )
                st.plotly_chart(fig_scores, width='stretch')

            with col2:
                st.write("**Score Frequency:**")
                for score, count in score_counts.items():
                    pct = (count / len(df)) * 100
                    st.write(f"- {score}: {count} ({pct:.1f}%)")

        # Export option
        st.markdown("### Export Results")
        df = pd.DataFrame(results['matches'])
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"simulation_{player_a}_vs_{player_b}.csv",
            mime="text/csv"
        )
