"""
Single Match Viewer UI - Page 2

Detailed shot-by-shot analysis of a single tennis match.
Shows rally details, point progression, and shot sequences.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    serve_pkl = 'models/serve_patterns.pkl'
    if os.path.exists(serve_pkl):
        with open(serve_pkl, 'rb') as f:
            data = pickle.load(f)
            serve_patterns = data.get('serve_patterns', {})

    rally_pkl = 'models/rally_patterns.pkl'
    if os.path.exists(rally_pkl):
        with open(rally_pkl, 'rb') as f:
            data = pickle.load(f)
            rally_patterns = data.get('rally_patterns', {})

    return serve_patterns, rally_patterns


def simulate_single_match(player_a, player_b, surface, best_of, use_shot_sim, seed=42):
    """Simulate a single match with detailed tracking"""

    # Load data
    loader = DataLoader(data_dir='data')
    matches = loader.load_match_data(years=[2020, 2021, 2022, 2023], tour='atp')

    if matches is None or len(matches) == 0:
        matches = loader._create_synthetic_data()

    # Get player stats
    stats_calc = PlayerStatsCalculator(matches)
    stats_a = stats_calc.get_player_stats(player_a, surface=surface)
    stats_b = stats_calc.get_player_stats(player_b, surface=surface)

    # Create simulator
    if use_shot_sim:
        serve_patterns, rally_patterns = load_trained_models()

        # Create independent RNGs for each model using deterministic seed offsets
        # This ensures reproducibility - same seed always produces same results
        serve_model = ServeModel(
            serve_patterns=serve_patterns,
            rng=np.random.default_rng(seed * 2 + 1)
        )
        return_model = ReturnModel(
            rng=np.random.default_rng(seed * 2 + 2)
        )
        rally_model = RallyModel(
            rally_patterns=rally_patterns,
            rng=np.random.default_rng(seed * 2 + 3)
        )

        point_sim = PointSimulator(
            rng=np.random.default_rng(seed * 2 + 4),
            use_shot_simulation=True,
            serve_model=serve_model,
            return_model=return_model,
            rally_model=rally_model
        )

        sim = MatchSimulator(
            player_a_stats=stats_a,
            player_b_stats=stats_b,
            best_of=best_of,
            seed=seed,  # MatchSimulator uses original seed for initial serve decision
            point_simulator=point_sim,
            track_point_history=True,
            surface=surface
        )
    else:
        sim = MatchSimulator(
            player_a_stats=stats_a,
            player_b_stats=stats_b,
            best_of=best_of,
            seed=seed,
            track_point_history=True,
            surface=surface
        )

    # Run simulation
    match_result = sim.simulate_match()

    return match_result


def create_match_flow_chart(set_scores, player_a, player_b):
    """Create a visual flow chart of the match progression"""

    fig = go.Figure()

    # Create set-by-set bars
    sets = list(range(1, len(set_scores) + 1))
    scores_a = [score[0] for score in set_scores]
    scores_b = [score[1] for score in set_scores]

    fig.add_trace(go.Bar(
        name=player_a,
        x=sets,
        y=scores_a,
        marker_color='#1f77b4',
        text=scores_a,
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name=player_b,
        x=sets,
        y=scores_b,
        marker_color='#ff7f0e',
        text=scores_b,
        textposition='auto',
    ))

    fig.update_layout(
        title="Set-by-Set Progression",
        xaxis_title="Set Number",
        yaxis_title="Games Won",
        barmode='group',
        height=400,
        showlegend=True
    )

    return fig


def create_stats_comparison_chart(match_result, player_a, player_b):
    """Create radar chart comparing key statistics"""

    categories = ['Aces', '1st Serve %', '1st Srv Won', '2nd Srv Won', 'Break Points']

    # Normalize stats to 0-100 scale for radar chart
    stats_a = [
        (match_result.aces_a / max(1, match_result.aces_a + match_result.aces_b) * 100),
        (match_result.first_serves_in_a / max(1, match_result.first_serves_total_a) * 100),
        (match_result.first_serve_points_won_a / max(1, match_result.first_serve_points_total_a) * 100),
        (match_result.second_serve_points_won_a / max(1, match_result.second_serve_points_total_a) * 100),
        (match_result.break_points_won_a / max(1, match_result.break_points_total_b) * 100)
    ]

    stats_b = [
        (match_result.aces_b / max(1, match_result.aces_a + match_result.aces_b) * 100),
        (match_result.first_serves_in_b / max(1, match_result.first_serves_total_b) * 100),
        (match_result.first_serve_points_won_b / max(1, match_result.first_serve_points_total_b) * 100),
        (match_result.second_serve_points_won_b / max(1, match_result.second_serve_points_total_b) * 100),
        (match_result.break_points_won_b / max(1, match_result.break_points_total_a) * 100)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=stats_a,
        theta=categories,
        fill='toself',
        name=player_a,
        line_color='#1f77b4'
    ))

    fig.add_trace(go.Scatterpolar(
        r=stats_b,
        theta=categories,
        fill='toself',
        name=player_b,
        line_color='#ff7f0e'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500,
        title="Performance Comparison"
    )

    return fig


def format_score_line(set_scores, winner_name, player_a_name):
    """Format the match score line"""
    score_parts = []
    for games_a, games_b in set_scores:
        # Bold the winning score
        if (games_a > games_b and winner_name == player_a_name) or (games_b > games_a and winner_name != player_a_name):
            score_parts.append(f"**{games_a}-{games_b}**")
        else:
            score_parts.append(f"{games_a}-{games_b}")

    return "  ".join(score_parts)


def display_play_by_play(match_result, player_a, player_b, use_shot_sim):
    """Display detailed point-by-point play-by-play"""

    point_history = match_result.point_history

    if not point_history:
        st.warning("No point history available")
        return

    st.write(f"**Total Points Played:** {len(point_history)}")

    # Filtering options
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_rally_length = st.selectbox(
            "Filter by Rally Length",
            ["All", "Short (1-3)", "Medium (4-7)", "Long (8+)"],
            key="pbp_rally_filter"
        )

    with col2:
        filter_server = st.selectbox(
            "Filter by Server",
            ["All", player_a, player_b],
            key="pbp_server_filter"
        )

    with col3:
        show_shot_details = st.checkbox(
            "Show Shot Details",
            value=use_shot_sim,
            disabled=not use_shot_sim,
            key="pbp_shot_details",
            help="Only available with shot-level simulation"
        )

    # Apply filters
    filtered_points = []
    for idx, point in enumerate(point_history):
        # Determine server from point data if available, otherwise estimate
        if hasattr(point, 'server_name') and point.server_name:
            point_server = point.server_name
        else:
            # Fallback: estimate based on point index (not accurate but better than nothing)
            point_server = player_a if (idx % 2 == 0) else player_b

        # Filter by rally length
        if filter_rally_length != "All":
            if filter_rally_length == "Short (1-3)" and point.rally_length > 3:
                continue
            elif filter_rally_length == "Medium (4-7)" and (point.rally_length < 4 or point.rally_length > 7):
                continue
            elif filter_rally_length == "Long (8+)" and point.rally_length < 8:
                continue

        # Filter by server
        if filter_server != "All" and point_server != filter_server:
            continue

        filtered_points.append((idx + 1, point, point_server))

    total_points = len(filtered_points)
    st.write(f"**Showing:** {total_points} of {len(point_history)} points")

    # Pagination controls
    points_per_page = 20
    total_pages = max(1, (total_points + points_per_page - 1) // points_per_page)

    # Initialize page number in session state
    if 'pbp_page' not in st.session_state:
        st.session_state.pbp_page = 1

    # Reset to page 1 if filters changed (check by storing filter signature)
    filter_signature = f"{filter_rally_length}|{filter_server}|{show_shot_details}"
    if 'pbp_filter_sig' not in st.session_state or st.session_state.pbp_filter_sig != filter_signature:
        st.session_state.pbp_page = 1
        st.session_state.pbp_filter_sig = filter_signature

    # Ensure page number is within valid range
    if st.session_state.pbp_page > total_pages:
        st.session_state.pbp_page = total_pages

    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("◀ Previous", disabled=st.session_state.pbp_page <= 1, key="pbp_prev"):
            st.session_state.pbp_page -= 1

    with col2:
        st.markdown(f"<div style='text-align: center;'>**Page {st.session_state.pbp_page} of {total_pages}**</div>", unsafe_allow_html=True)

    with col3:
        if st.button("Next ▶", disabled=st.session_state.pbp_page >= total_pages, key="pbp_next"):
            st.session_state.pbp_page += 1

    st.markdown("---")

    # Calculate slice indices for current page
    start_idx = (st.session_state.pbp_page - 1) * points_per_page
    end_idx = min(start_idx + points_per_page, total_points)

    # Display points for current page
    for point_num, point, server in filtered_points[start_idx:end_idx]:
        with st.container():
            # Point header
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])

            with col1:
                st.write(f"**#{point_num}**")

            with col2:
                st.write(f"🎾 Server: **{server}**")

            with col3:
                serve_type = "1st" if point.was_first_serve else "2nd"
                st.write(f"Serve: {serve_type}")

            with col4:
                winner = server if point.server_won else (player_b if server == player_a else player_a)
                winner_emoji = "✅" if point.server_won else "🔄"
                st.write(f"{winner_emoji} Winner: **{winner}**")

            # Running score display
            if hasattr(point, 'game_score') and point.game_score:
                score_col1, score_col2 = st.columns([1, 1])

                with score_col1:
                    # Display game score and games score (after this point)
                    st.caption(f"📊 Score after point: **{point.game_score}** | Games: **{point.games_score}** (Set {point.set_number})")

                with score_col2:
                    # Display sets score if available
                    if point.sets_score and point.sets_score != "0-0":
                        st.caption(f"🏆 Sets: **{point.sets_score}**")

            # Point details
            if point.was_ace:
                st.success(f"⚡ **ACE** - {server}")
            elif point.was_double_fault:
                st.error(f"❌ **DOUBLE FAULT** - {server}")
            else:
                # Rally details
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.metric("Rally Length", f"{point.rally_length} shots")

                with col2:
                    if point.serve_placement:
                        # Get the enum value (W/T/B) not the full enum name
                        placement_value = point.serve_placement.value if hasattr(point.serve_placement, 'value') else str(point.serve_placement)
                        placement_map = {'W': 'Wide', 'T': 'T (Center)', 'B': 'Body'}
                        placement = placement_map.get(placement_value, placement_value)
                        st.write(f"📍 Serve Placement: **{placement}**")

                # Shot-by-shot details
                if show_shot_details and point.rally and point.rally.shots:
                    with st.expander(f"🔍 Shot Sequence ({len(point.rally.shots)} shots)", expanded=False):
                        shot_data = []

                        for shot in point.rally.shots:
                            shot_type = str(shot.shot_type).split('.')[-1].replace('_', ' ').title()
                            direction = str(shot.direction).split('.')[-1].replace('_', ' ').title()
                            position = str(shot.position).split('.')[-1].replace('_', ' ').title()
                            outcome = str(shot.outcome).split('.')[-1].replace('_', ' ').title()

                            # Determine outcome emoji
                            if 'Winner' in outcome:
                                outcome_emoji = "🏆"
                            elif 'Error' in outcome:
                                outcome_emoji = "❌"
                            else:
                                outcome_emoji = "➡️"

                            shot_player = player_a if shot.player == 'server' else player_b
                            if server != player_a:  # flip if player B is serving
                                shot_player = player_a if shot.player == 'returner' else player_b

                            shot_data.append({
                                'Shot #': shot.shot_number,
                                'Player': shot_player,
                                'Type': shot_type,
                                'Direction': direction,
                                'Position': position,
                                'Outcome': f"{outcome_emoji} {outcome}"
                            })

                        df_shots = pd.DataFrame(shot_data)
                        st.dataframe(df_shots, hide_index=True, width='stretch')

                        # Point ending shot highlight
                        if point.rally.point_ending_shot:
                            ending_shot = point.rally.point_ending_shot
                            ending_player = player_a if ending_shot.player == 'server' else player_b
                            if server != player_a:
                                ending_player = player_a if ending_shot.player == 'returner' else player_b

                            ending_type = str(ending_shot.shot_type).split('.')[-1].replace('_', ' ').title()
                            ending_outcome = str(ending_shot.outcome).split('.')[-1].replace('_', ' ').title()

                            if ending_shot.is_error():
                                st.error(f"❌ Point ended on {ending_player}'s {ending_type} ({ending_outcome})")
                            else:
                                st.success(f"🏆 Point ended on {ending_player}'s {ending_type} Winner!")

            st.markdown("---")


def render():
    """Render the Single Match Viewer page"""

    st.markdown('<p class="main-header"> Single Match Viewer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Simulate and analyze a single match in detail</p>', unsafe_allow_html=True)

    # Load players
    players = load_player_list()

    # Input controls in columns
    col1, col2 = st.columns(2)

    with col1:
        default_a = players.index("Carlos Alcaraz") if "Carlos Alcaraz" in players else 0
        player_a = st.selectbox("Player A", players, index=default_a, key="sm_player_a")

    with col2:
        default_b = players.index("Jannik Sinner") if "Jannik Sinner" in players else 1
        player_b = st.selectbox("Player B", players, index=default_b, key="sm_player_b")

    # Match parameters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        surface = st.selectbox("Surface", ["hard", "clay", "grass"], key="sm_surface")

    with col2:
        best_of = st.selectbox("Format", [3, 5], format_func=lambda x: f"Best of {x}", key="sm_best_of")

    with col3:
        use_shot_sim = st.checkbox("Shot-Level Detail", value=True, key="sm_shot_sim",
                                   help="Enable for rally-by-rally analysis")

    with col4:
        seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42,
                              help="Use same seed for reproducible results")

    # Simulate button
    if st.button("🎲 Simulate Match", type="primary", width='stretch'):

        if player_a == player_b:
            st.error("Please select different players")
            return

        with st.spinner("Simulating match..."):
            match_result = simulate_single_match(player_a, player_b, surface, best_of, use_shot_sim, seed)

        # Store match result and parameters in session state
        st.session_state.match_result = match_result
        st.session_state.match_player_a = player_a
        st.session_state.match_player_b = player_b
        st.session_state.match_surface = surface
        st.session_state.match_best_of = best_of
        st.session_state.match_use_shot_sim = use_shot_sim
        st.session_state.match_seed = seed
        st.session_state.pbp_page = 1  # Reset pagination to first page

        st.success("✅ Match simulation complete!")

    # Display match results if available in session state
    if 'match_result' in st.session_state:
        match_result = st.session_state.match_result
        player_a = st.session_state.match_player_a
        player_b = st.session_state.match_player_b
        surface = st.session_state.match_surface
        best_of = st.session_state.match_best_of
        use_shot_sim = st.session_state.match_use_shot_sim
        seed = st.session_state.match_seed

        # Match Result Header
        st.markdown("---")
        st.markdown("## Match Result")

        # Winner announcement
        winner_emoji = "🥇"
        st.markdown(f"### {winner_emoji} {match_result.winner} wins!")

        # Score line
        score_line = format_score_line(match_result.set_scores, match_result.winner, player_a)
        st.markdown(f"#### Final Score: {score_line}")

        # Match summary metrics
        st.markdown("### Match Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Sets Won", f"{match_result.sets_won_a}-{match_result.sets_won_b}")

        with col2:
            st.metric("Total Points", match_result.total_points)

        with col3:
            points_won_a = match_result.total_points_won_a
            points_won_b = match_result.total_points_won_b
            st.metric("Points Won", f"{points_won_a}-{points_won_b}")

        with col4:
            point_win_pct_a = (points_won_a / match_result.total_points * 100) if match_result.total_points > 0 else 0
            st.metric(f"{player_a} Point %", f"{point_win_pct_a:.1f}%")

        # Visualizations
        st.markdown("---")
        st.markdown("### 📊 Match Visualizations")

        # Set progression chart
        fig_sets = create_match_flow_chart(match_result.set_scores, player_a, player_b)
        st.plotly_chart(fig_sets, width='stretch')

        # Stats comparison radar
        fig_radar = create_stats_comparison_chart(match_result, player_a, player_b)
        st.plotly_chart(fig_radar, width='stretch')

        # Detailed Statistics Table
        st.markdown("---")
        st.markdown("### 📈 Detailed Statistics")

        # Create comparison table
        stats_data = {
            'Statistic': [
                'Aces',
                'Double Faults',
                '1st Serve In',
                '1st Serve %',
                '1st Serve Points Won',
                '2nd Serve Points Won',
                'Break Points Won',
                'Total Points Won'
            ],
            player_a: [
                str(match_result.aces_a),
                str(match_result.double_faults_a),
                f"{match_result.first_serves_in_a}/{match_result.first_serves_total_a}",
                f"{match_result.first_serves_in_a / max(1, match_result.first_serves_total_a) * 100:.1f}%",
                f"{match_result.first_serve_points_won_a}/{match_result.first_serve_points_total_a}",
                f"{match_result.second_serve_points_won_a}/{match_result.second_serve_points_total_a}",
                f"{match_result.break_points_won_a}/{match_result.break_points_total_b}",
                str(match_result.total_points_won_a)
            ],
            player_b: [
                str(match_result.aces_b),
                str(match_result.double_faults_b),
                f"{match_result.first_serves_in_b}/{match_result.first_serves_total_b}",
                f"{match_result.first_serves_in_b / max(1, match_result.first_serves_total_b) * 100:.1f}%",
                f"{match_result.first_serve_points_won_b}/{match_result.first_serve_points_total_b}",
                f"{match_result.second_serve_points_won_b}/{match_result.second_serve_points_total_b}",
                f"{match_result.break_points_won_b}/{match_result.break_points_total_a}",
                str(match_result.total_points_won_b)
            ]
        }

        df_stats = pd.DataFrame(stats_data)

        # Style the dataframe
        st.dataframe(
            df_stats,
            width='stretch',
            hide_index=True,
            column_config={
                "Statistic": st.column_config.TextColumn("Statistic", width="medium"),
                player_a: st.column_config.TextColumn(player_a, width="medium"),
                player_b: st.column_config.TextColumn(player_b, width="medium"),
            }
        )

        # Play-by-play details
        if match_result.point_history:
            st.markdown("---")
            st.markdown("### 📋 Point-by-Point Play-by-Play")

            with st.expander("🎾 View Shot-by-Shot Details", expanded=False):
                display_play_by_play(match_result, player_a, player_b, use_shot_sim)

        # Match replay info
        st.markdown("---")
        st.markdown("### 🔄 Replay This Match")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Random Seed Used:** {seed}")
            st.write("Use the same seed above to replay this exact match")

        with col2:
            # Export match summary
            export_data = {
                'Player A': [player_a],
                'Player B': [player_b],
                'Winner': [match_result.winner],
                'Score': [match_result.score],
                'Surface': [surface],
                'Format': [f'Best of {best_of}'],
                'Seed': [seed],
                'Total Points': [match_result.total_points],
                'Aces A': [match_result.aces_a],
                'Aces B': [match_result.aces_b],
                'DFs A': [match_result.double_faults_a],
                'DFs B': [match_result.double_faults_b]
            }

            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)

            st.download_button(
                label="📥 Export Match Summary",
                data=csv,
                file_name=f"match_{player_a}_vs_{player_b}_seed{seed}.csv",
                mime="text/csv",
                width='stretch'
            )
