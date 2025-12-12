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

        st.info(f"⏱️ Simulation time: **{shot_time_ms:.2f} ms** ({shot_time_ms/max(0.1, point_time_ms):.1f}x slower)")


def display_shot_level_insights(shot_result):
    """Display exclusive insights only available from shot-level simulation"""

    st.markdown("### 🎯 Shot-Level Exclusive Insights")
    st.markdown("Detailed analysis only possible with shot-by-shot simulation")

    if not hasattr(shot_result, 'point_history') or not shot_result.point_history:
        st.warning("Shot-level data not available. Make sure track_point_history=True")
        return

    # Extract data from point history
    rally_lengths = []
    shot_types = defaultdict(int)
    positions = defaultdict(int)
    serve_placements = defaultdict(int)

    for point in shot_result.point_history:
        if hasattr(point, 'rally_length'):
            rally_lengths.append(point.rally_length)

        if hasattr(point, 'rally') and point.rally and point.rally.shots:
            for shot in point.rally.shots:
                shot_type = str(shot.shot_type).split('.')[-1]
                shot_types[shot_type] += 1

                position = str(shot.position).split('.')[-1]
                positions[position] += 1

        if hasattr(point, 'serve_placement') and point.serve_placement:
            placement_value = point.serve_placement.value if hasattr(point.serve_placement, 'value') else str(point.serve_placement)
            serve_placements[placement_value] += 1

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Rally length distribution
        st.markdown("#### 📊 Rally Length Distribution")
        if rally_lengths:
            rally_df = pd.DataFrame({'Rally Length': rally_lengths})

            fig = px.histogram(
                rally_df,
                x='Rally Length',
                nbins=15,
                title="Points by Rally Length",
                labels={'Rally Length': 'Shots per Point'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                yaxis_title="Number of Points"
            )
            st.plotly_chart(fig, use_container_width=True)

            avg_rally = np.mean(rally_lengths)
            st.caption(f"Average rally length: **{avg_rally:.1f} shots**")
        else:
            st.info("No rally data available")

        # Court position distribution
        st.markdown("#### 📍 Court Position Distribution")
        if positions:
            pos_df = pd.DataFrame({
                'Position': list(positions.keys()),
                'Count': list(positions.values())
            })

            # Format position names
            pos_df['Position'] = pos_df['Position'].str.replace('_', ' ').str.title()

            fig = px.pie(
                pos_df,
                values='Count',
                names='Position',
                title="Where Shots Were Hit From",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No position data available")

    with col2:
        # Shot type breakdown
        st.markdown("#### 🎾 Shot Type Breakdown")
        if shot_types:
            shot_df = pd.DataFrame({
                'Shot Type': list(shot_types.keys()),
                'Count': list(shot_types.values())
            })

            # Format shot type names
            shot_df['Shot Type'] = shot_df['Shot Type'].str.replace('_', ' ').str.title()

            # Sort by count
            shot_df = shot_df.sort_values('Count', ascending=False)

            fig = px.bar(
                shot_df,
                x='Shot Type',
                y='Count',
                title="Shots by Type",
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shot type data available")

        # Serve placement effectiveness
        st.markdown("#### 🎯 Serve Placement Distribution")
        if serve_placements:
            serve_df = pd.DataFrame({
                'Placement': list(serve_placements.keys()),
                'Count': list(serve_placements.values())
            })

            # Map placement codes to names
            placement_map = {'W': 'Wide', 'T': 'T (Center)', 'B': 'Body'}
            serve_df['Placement'] = serve_df['Placement'].apply(lambda x: placement_map.get(str(x), str(x)))

            fig = px.pie(
                serve_df,
                values='Count',
                names='Placement',
                title="Serve Placement Distribution",
                color_discrete_sequence=px.colors.sequential.Greens_r
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No serve placement data available")


def run_batch_comparison(player_a, player_b, surface, best_of, seed, n_sims):
    """
    Run multiple simulations with both modes for statistical validation

    Returns:
        dict with results from both modes
    """
    stats_calc = load_data()
    stats_a = stats_calc.get_player_stats(player_a, surface=surface.lower())
    stats_b = stats_calc.get_player_stats(player_b, surface=surface.lower())

    point_results = []
    shot_results = []
    point_times = []
    shot_times = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(n_sims):
        status_text.text(f"Running simulation {i+1}/{n_sims}...")
        progress_bar.progress((i + 1) / n_sims)

        # Use different seed for each simulation
        sim_seed = seed + i

        # Point-level
        start = time.time()
        point_sim = MatchSimulator(
            player_a_stats=stats_a,
            player_b_stats=stats_b,
            best_of=best_of,
            seed=sim_seed,
            surface=surface.lower()
        )
        point_result = point_sim.simulate_match()
        point_times.append((time.time() - start) * 1000)
        point_results.append(point_result)

        # Shot-level
        start = time.time()
        serve_model = ServeModel(rng=np.random.default_rng(sim_seed))
        return_model = ReturnModel(rng=np.random.default_rng(sim_seed + 1))
        rally_model = RallyModel(rng=np.random.default_rng(sim_seed + 2))

        point_simulator = PointSimulator(
            rng=np.random.default_rng(sim_seed + 3),
            use_shot_simulation=True,
            serve_model=serve_model,
            return_model=return_model,
            rally_model=rally_model
        )

        shot_sim = MatchSimulator(
            player_a_stats=stats_a,
            player_b_stats=stats_b,
            best_of=best_of,
            seed=sim_seed,
            point_simulator=point_simulator,
            surface=surface.lower()
        )
        shot_result = shot_sim.simulate_match()
        shot_times.append((time.time() - start) * 1000)
        shot_results.append(shot_result)

    progress_bar.empty()
    status_text.empty()

    return {
        'point_results': point_results,
        'shot_results': shot_results,
        'point_times': point_times,
        'shot_times': shot_times
    }


def display_statistical_validation(batch_results, player_a, player_b):
    """Display statistical comparison across multiple simulations"""

    st.markdown("### 📈 Statistical Validation")
    st.markdown(f"Comparing {len(batch_results['point_results'])} simulations from each mode")

    point_results = batch_results['point_results']
    shot_results = batch_results['shot_results']

    # Extract metrics
    point_wins_a = sum(1 for r in point_results if r.winner == player_a)
    shot_wins_a = sum(1 for r in shot_results if r.winner == player_a)

    point_aces = [r.aces_a + r.aces_b for r in point_results]
    shot_aces = [r.aces_a + r.aces_b for r in shot_results]

    point_dfs = [r.double_faults_a + r.double_faults_b for r in point_results]
    shot_dfs = [r.double_faults_a + r.double_faults_b for r in shot_results]

    point_total_points = [r.total_points for r in point_results]
    shot_total_points = [r.total_points for r in shot_results]

    # Win rate comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            f"{player_a} Win Rate (Point-Level)",
            f"{(point_wins_a / len(point_results) * 100):.1f}%"
        )

    with col2:
        st.metric(
            f"{player_a} Win Rate (Shot-Level)",
            f"{(shot_wins_a / len(shot_results) * 100):.1f}%"
        )

    with col3:
        diff = abs(point_wins_a - shot_wins_a) / len(point_results) * 100
        st.metric(
            "Difference",
            f"{diff:.1f}%",
            delta=None,
            help="Difference in win rates between modes (should be <5% for validation over 1k sims)"
        )

    # Box plots comparing distributions
    st.markdown("#### Match Statistics Distributions")

    # Create box plots for key metrics
    metrics_df = pd.DataFrame({
        'Aces (Point)': point_aces,
        'Aces (Shot)': shot_aces,
        'DFs (Point)': point_dfs,
        'DFs (Shot)': shot_dfs,
        'Total Points (Point)': point_total_points,
        'Total Points (Shot)': shot_total_points
    })

    col1, col2 = st.columns(2)

    with col1:
        # Aces comparison
        fig = go.Figure()
        fig.add_trace(go.Box(y=point_aces, name='Point-Level', marker_color='#1f77b4'))
        fig.add_trace(go.Box(y=shot_aces, name='Shot-Level', marker_color='#ff7f0e'))
        fig.update_layout(
            title="Aces per Match",
            yaxis_title="Aces",
            showlegend=True,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # Match duration comparison
        fig = go.Figure()
        fig.add_trace(go.Box(y=point_total_points, name='Point-Level', marker_color='#1f77b4'))
        fig.add_trace(go.Box(y=shot_total_points, name='Shot-Level', marker_color='#ff7f0e'))
        fig.update_layout(
            title="Match Duration (Total Points)",
            yaxis_title="Points",
            showlegend=True,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Double faults comparison
        fig = go.Figure()
        fig.add_trace(go.Box(y=point_dfs, name='Point-Level', marker_color='#1f77b4'))
        fig.add_trace(go.Box(y=shot_dfs, name='Shot-Level', marker_color='#ff7f0e'))
        fig.update_layout(
            title="Double Faults per Match",
            yaxis_title="Double Faults",
            showlegend=True,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics table
        st.markdown("#### Summary Statistics")
        summary_df = pd.DataFrame({
            'Metric': ['Aces', 'Double Faults', 'Total Points'],
            'Point-Level Mean': [
                f"{np.mean(point_aces):.1f}",
                f"{np.mean(point_dfs):.1f}",
                f"{np.mean(point_total_points):.1f}"
            ],
            'Shot-Level Mean': [
                f"{np.mean(shot_aces):.1f}",
                f"{np.mean(shot_dfs):.1f}",
                f"{np.mean(shot_total_points):.1f}"
            ],
            'Difference': [
                f"{abs(np.mean(point_aces) - np.mean(shot_aces)):.1f}",
                f"{abs(np.mean(point_dfs) - np.mean(shot_dfs)):.1f}",
                f"{abs(np.mean(point_total_points) - np.mean(shot_total_points)):.1f}"
            ]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)


def display_performance_metrics(point_times, shot_times):
    """Display performance comparison metrics table"""

    st.markdown("### ⚙️ Performance Comparison")

    avg_point_time = np.mean(point_times)
    avg_shot_time = np.mean(shot_times)

    metrics_df = pd.DataFrame({
        'Metric': [
            'Avg Simulation Time',
            'Speed (matches/sec)',
            'Memory Overhead',
            'Lines of Code',
            'Data Sources',
            'Output Fields',
            'Use Case'
        ],
        'Point-Level': [
            f'{avg_point_time:.2f} ms',
            f'{1000/max(avg_point_time, 0.1):.0f}',
            '~1 MB',
            '~500',
            'ATP Stats',
            '~15',
            'Batch predictions'
        ],
        'Shot-Level': [
            f'{avg_shot_time:.2f} ms',
            f'{1000/max(avg_shot_time, 0.1):.0f}',
            '~5 MB',
            '~2000',
            'ATP + MCP',
            '~60',
            'Detailed analysis'
        ],
        'Difference': [
            f'{avg_shot_time/max(avg_point_time, 0.1):.1f}x slower',
            f'{(1000/max(avg_point_time, 0.1)) - (1000/max(avg_shot_time, 0.1)):.0f} fewer/sec',
            '+4 MB',
            '+1500 lines',
            '+17K matches',
            '+45 fields',
            'Complementary'
        ]
    })

    st.dataframe(metrics_df, hide_index=True, use_container_width=True)


def create_export_data(point_result, shot_result):
    """Create exportable comparison data"""

    export_df = pd.DataFrame({
        'Metric': [
            'Winner',
            'Score',
            'Total Points',
            'Aces (Player A)',
            'Aces (Player B)',
            'Double Faults (Player A)',
            'Double Faults (Player B)',
            '1st Serve % (Player A)',
            '1st Serve % (Player B)',
        ],
        'Point-Level': [
            point_result.winner,
            point_result.score,
            point_result.total_points,
            point_result.aces_a,
            point_result.aces_b,
            point_result.double_faults_a,
            point_result.double_faults_b,
            f"{(point_result.first_serves_in_a / max(1, point_result.first_serves_total_a) * 100):.1f}%",
            f"{(point_result.first_serves_in_b / max(1, point_result.first_serves_total_b) * 100):.1f}%",
        ],
        'Shot-Level': [
            shot_result.winner,
            shot_result.score,
            shot_result.total_points,
            shot_result.aces_a,
            shot_result.aces_b,
            shot_result.double_faults_a,
            shot_result.double_faults_b,
            f"{(shot_result.first_serves_in_a / max(1, shot_result.first_serves_total_a) * 100):.1f}%",
            f"{(shot_result.first_serves_in_b / max(1, shot_result.first_serves_total_b) * 100):.1f}%",
        ]
    })

    return export_df


def render():
    """Main render function for Model Comparison page"""

    st.markdown("## 📊 Model Comparison")
    st.markdown("Compare point-level vs shot-level simulation modes")

    # Configuration panel
    st.markdown("### ⚙️ Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        players = load_player_list()
        # Default to Sinner if available
        default_a_index = players.index("Jannik Sinner") if "Jannik Sinner" in players else 0
        player_a = st.selectbox("Player A", players, index=default_a_index, key="comp_player_a")

    with col2:
        player_b_options = [p for p in players if p != player_a]
        # Default to Alcaraz if available
        default_b_index = player_b_options.index("Carlos Alcaraz") if "Carlos Alcaraz" in player_b_options else 0
        player_b = st.selectbox("Player B", player_b_options, index=default_b_index, key="comp_player_b")

    with col3:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="comp_surface")

    with col4:
        best_of = st.selectbox("Best of", [3, 5], key="comp_best_of")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        n_simulations = st.selectbox(
            "Number of Simulations",
            [1, 10, 100, 1000],
            index=0,  # Default to 1
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

            # Shot-level exclusive insights
            st.markdown("---")
            display_shot_level_insights(shot_result)

            # Performance metrics
            st.markdown("---")
            display_performance_metrics([point_time], [shot_time])

            # Export functionality
            st.markdown("---")
            st.markdown("### 📥 Export Results")
            export_df = create_export_data(point_result, shot_result)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Comparison (CSV)",
                data=csv,
                file_name=f"comparison_{player_a}_vs_{player_b}.csv",
                mime="text/csv"
            )

        else:
            # Multiple simulations for statistical validation
            with st.spinner(f"Running {n_simulations} simulations..."):
                batch_results = run_batch_comparison(player_a, player_b, surface, best_of, seed, n_simulations)

            # Statistical validation
            display_statistical_validation(batch_results, player_a, player_b)

            # Performance metrics
            st.markdown("---")
            display_performance_metrics(batch_results['point_times'], batch_results['shot_times'])

            # Export functionality
            st.markdown("---")
            st.markdown("### 📥 Export Results")
            st.info("💡 Export is available for single match comparisons (n=1). For batch runs, use the summary statistics above.")

    else:
        # Show instructions
        st.info("""
        👆 Select players and match settings above, then click **Run Comparison** to see:

        - **Single Match (n=1)**: Side-by-side comparison with shot-level insights
        - **Multiple Matches (n>1)**: Statistical validation across many simulations

        This helps validate that shot-level simulation produces equivalent match outcomes
        while providing much richer tactical detail.
        """)
