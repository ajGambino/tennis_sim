"""
Model Comparison UI - Page 3

Compare point-level vs shot-level simulation modes to validate accuracy
and showcase the additional insights from shot-level modeling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from collections import defaultdict
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

                # Sort by Elo rating (highest first)
                sorted_players = sorted(player_avg_elos.items(), key=lambda x: -x[1])
                return [player for player, _ in sorted_players]
        else:
            # Fallback to default list if Elo ratings not available
            return [
                'Carlos Alcaraz', 'Novak Djokovic', 'Jannik Sinner',
                'Daniil Medvedev', 'Andrey Rublev', 'Rafael Nadal',
                'Alexander Zverev', 'Holger Rune', 'Stefanos Tsitsipas',
                'Taylor Fritz', 'Casper Ruud', 'Hubert Hurkacz'
            ]
    except Exception as e:
        st.error(f"Error loading player list: {e}")
        return ['Carlos Alcaraz', 'Jannik Sinner']


@st.cache_data
def load_data():
    """Load match data and create player stats calculator"""
    loader = DataLoader(data_dir='data')
    matches = loader.load_match_data(years=[2023], tour='atp')
    if matches is None or len(matches) == 0:
        matches = loader._create_synthetic_data()
    stats_calc = PlayerStatsCalculator(matches)
    return stats_calc


def run_single_comparison(player_a, player_b, surface, best_of, seed):
    """
    Run both point-level and shot-level simulations for a single match

    Returns:
        tuple: (point_result, shot_result, point_time_ms, shot_time_ms)
    """
    stats_calc = load_data()

    # Get player stats
    stats_a = stats_calc.get_player_stats(player_a, surface=surface.lower())
    stats_b = stats_calc.get_player_stats(player_b, surface=surface.lower())

    # Point-level simulation (legacy)
    start_time = time.time()
    point_sim = MatchSimulator(
        player_a_stats=stats_a,
        player_b_stats=stats_b,
        best_of=best_of,
        seed=seed,
        surface=surface.lower()
    )
    point_result = point_sim.simulate_match()
    point_time_ms = (time.time() - start_time) * 1000

    # Shot-level simulation
    start_time = time.time()
    serve_model = ServeModel(rng=np.random.default_rng(seed))
    return_model = ReturnModel(rng=np.random.default_rng(seed + 1))
    rally_model = RallyModel(rng=np.random.default_rng(seed + 2))

    point_simulator = PointSimulator(
        rng=np.random.default_rng(seed + 3),
        use_shot_simulation=True,
        serve_model=serve_model,
        return_model=return_model,
        rally_model=rally_model
    )

    shot_sim = MatchSimulator(
        player_a_stats=stats_a,
        player_b_stats=stats_b,
        best_of=best_of,
        seed=seed,
        point_simulator=point_simulator,
        track_point_history=True,
        surface=surface.lower()
    )
    shot_result = shot_sim.simulate_match()
    shot_time_ms = (time.time() - start_time) * 1000

    return point_result, shot_result, point_time_ms, shot_time_ms


def display_single_match_comparison(point_result, shot_result, point_time_ms, shot_time_ms):
    """Display side-by-side comparison of a single match from both modes"""

    st.markdown("### 🎾 Single Match Comparison")
    st.markdown("Same match simulated with both modes using identical seeds")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚡ Point-Level (Legacy)")
        st.markdown(f"**Winner:** {point_result.winner}")
        st.markdown(f"**Score:** {point_result.score}")

        # Key stats
        st.metric("Total Points", point_result.total_points)
        st.metric("Aces (A/B)", f"{point_result.aces_a} / {point_result.aces_b}")
        st.metric("Double Faults (A/B)", f"{point_result.double_faults_a} / {point_result.double_faults_b}")

        first_pct_a = (point_result.first_serves_in_a / max(1, point_result.first_serves_total_a)) * 100
        first_pct_b = (point_result.first_serves_in_b / max(1, point_result.first_serves_total_b)) * 100
        st.metric("1st Serve % (A/B)", f"{first_pct_a:.1f}% / {first_pct_b:.1f}%")

        st.info(f"⏱️ Simulation time: **{point_time_ms:.2f} ms**")

    with col2:
        st.markdown("#### 🎯 Shot-Level (Detailed)")
        st.markdown(f"**Winner:** {shot_result.winner}")
        st.markdown(f"**Score:** {shot_result.score}")

        # Same key stats
        st.metric("Total Points", shot_result.total_points)
        st.metric("Aces (A/B)", f"{shot_result.aces_a} / {shot_result.aces_b}")
        st.metric("Double Faults (A/B)", f"{shot_result.double_faults_a} / {shot_result.double_faults_b}")

        first_pct_a = (shot_result.first_serves_in_a / max(1, shot_result.first_serves_total_a)) * 100
        first_pct_b = (shot_result.first_serves_in_b / max(1, shot_result.first_serves_total_b)) * 100
        st.metric("1st Serve % (A/B)", f"{first_pct_a:.1f}% / {first_pct_b:.1f}%")

        st.info(f"⏱️ Simulation time: **{shot_time_ms:.2f} ms** ({shot_time_ms/point_time_ms:.1f}x slower)")


def render():
    """Main render function for Model Comparison page"""

    st.markdown("## 📊 Model Comparison")
    st.markdown("Compare point-level vs shot-level simulation modes")

    # Configuration panel
    st.markdown("### ⚙️ Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        players = load_player_list()
        player_a = st.selectbox("Player A", players, index=0, key="comp_player_a")

    with col2:
        player_b_options = [p for p in players if p != player_a]
        player_b = st.selectbox("Player B", player_b_options, index=0, key="comp_player_b")

    with col3:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="comp_surface")

    with col4:
        best_of = st.selectbox("Best of", [3, 5], key="comp_best_of")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        n_simulations = st.selectbox(
            "Number of Simulations",
            [1, 10, 100, 1000],
            help="1 = Detailed single match\n10-1000 = Statistical validation"
        )

    with col2:
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=99999)

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        run_button = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

    st.markdown("---")

    # Run comparison when button clicked
    if run_button:
        if n_simulations == 1:
            # Single match comparison
            with st.spinner("Running simulations..."):
                point_result, shot_result, point_time, shot_time = run_single_comparison(
                    player_a, player_b, surface, best_of, seed
                )

            display_single_match_comparison(point_result, shot_result, point_time, shot_time)

            # TODO: Add shot-level exclusive insights section
            st.markdown("---")
            st.info("🚧 Shot-level exclusive insights (rally length, shot types, positions) coming soon!")

        else:
            # Multiple simulations for statistical validation
            st.info(f"🚧 Statistical validation with {n_simulations} simulations coming soon!")
            st.markdown("""
            This feature will run both simulation modes multiple times and compare:
            - Win rate consistency
            - Match statistics distributions
            - Statistical significance tests
            """)

    else:
        # Show instructions
        st.info("""
        👆 Select players and match settings above, then click **Run Comparison** to see:

        - **Single Match (n=1)**: Side-by-side comparison with shot-level insights
        - **Multiple Matches (n>1)**: Statistical validation across many simulations

        This helps validate that shot-level simulation produces equivalent match outcomes
        while providing much richer tactical detail.
        """)
