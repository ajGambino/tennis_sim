"""
Player Statistics UI - Page 4

Interactive player statistics explorer with serve patterns, return stats,
rally characteristics, and head-to-head comparisons.
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
import json
import pickle
import os

from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator
from simulation.elo_system import EloSystem


@st.cache_data
def load_player_list():
    """Load list of available players sorted by Elo rating"""
    try:
        # Load Elo ratings
        elo_file = 'models/elo_ratings_atp.json'
        if os.path.exists(elo_file):
            with open(elo_file, 'r') as f:
                elo_data = json.load(f)
                ratings = elo_data.get('ratings', {})

                # Calculate average Elo for each player
                player_avg_elos = {}
                for player, surfaces in ratings.items():
                    if isinstance(surfaces, dict):
                        avg_elo = np.mean(list(surfaces.values()))
                    else:
                        avg_elo = surfaces
                    player_avg_elos[player] = avg_elo

                # Sort by Elo (highest to lowest)
                sorted_players = sorted(player_avg_elos.keys(),
                                       key=lambda p: player_avg_elos[p],
                                       reverse=True)
                return sorted_players

        # Fallback
        return ["Carlos Alcaraz", "Jannik Sinner", "Novak Djokovic"]
    except Exception as e:
        st.error(f"Error loading players: {e}")
        return ["Carlos Alcaraz", "Jannik Sinner"]


@st.cache_data
def load_data():
    """Load match data and compute player statistics"""
    loader = DataLoader(data_dir='data')
    matches = loader.load_match_data(years=[2020, 2021, 2022, 2023], tour='atp')

    if matches is None or len(matches) == 0:
        st.error("No match data found")
        return None, None

    stats_calculator = PlayerStatsCalculator(matches, min_matches=10)
    return matches, stats_calculator


@st.cache_data
def load_elo_ratings():
    """Load Elo ratings from JSON file"""
    elo_file = 'models/elo_ratings_atp.json'
    if os.path.exists(elo_file):
        with open(elo_file, 'r') as f:
            return json.load(f).get('ratings', {})
    return {}


@st.cache_data
def load_serve_patterns():
    """Load serve patterns from trained model"""
    serve_pkl = 'models/serve_patterns.pkl'
    if os.path.exists(serve_pkl):
        with open(serve_pkl, 'rb') as f:
            data = pickle.load(f)
            return data.get('serve_patterns', {})
    return {}


@st.cache_data
def load_rally_patterns():
    """Load rally patterns from trained model"""
    rally_pkl = 'models/rally_patterns.pkl'
    if os.path.exists(rally_pkl):
        with open(rally_pkl, 'rb') as f:
            data = pickle.load(f)
            return data.get('rally_patterns', {})
    return {}


def get_player_elo(player_name, surface, elo_ratings):
    """Get Elo rating for a player on specific surface"""
    if player_name not in elo_ratings:
        return 1500.0

    player_elo = elo_ratings[player_name]

    if isinstance(player_elo, dict):
        if surface and surface.lower() in player_elo:
            return player_elo[surface.lower()]
        else:
            # Return average across all surfaces
            return np.mean(list(player_elo.values()))
    else:
        return player_elo


def display_player_overview(player_name, surface, stats, elo_ratings, matches):
    """Display player overview with Elo and match statistics"""

    st.markdown("### 📊 Player Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Overall Elo
        overall_elo = get_player_elo(player_name, None, elo_ratings)
        st.metric("Overall Elo", f"{overall_elo:.0f}")

    with col2:
        # Surface-specific Elo
        if surface and surface != "All":
            surface_elo = get_player_elo(player_name, surface, elo_ratings)
            st.metric(f"{surface} Elo", f"{surface_elo:.0f}")
        else:
            st.metric("Matches Played", stats.matches_played)

    with col3:
        # Win percentage
        if stats.matches_played > 0:
            player_matches = matches[
                (matches['winner_name'] == player_name) |
                (matches['loser_name'] == player_name)
            ]
            if surface and surface != "All":
                player_matches = player_matches[player_matches['surface'] == surface.lower()]

            wins = len(player_matches[player_matches['winner_name'] == player_name])
            total = len(player_matches)
            win_pct = (wins / total * 100) if total > 0 else 0
            st.metric("Win %", f"{win_pct:.1f}%")
        else:
            st.metric("Win %", "N/A")

    with col4:
        # Ranking indicator
        if overall_elo >= 2000:
            st.metric("Tier", "🏆 Elite")
        elif overall_elo >= 1800:
            st.metric("Tier", "⭐ Top 20")
        elif overall_elo >= 1600:
            st.metric("Tier", "📈 Top 100")
        else:
            st.metric("Tier", "🎾 Challenger")


def display_serve_statistics(player_name, surface, stats, serve_patterns):
    """Display serve statistics with charts"""

    st.markdown("### 🎾 Serve Statistics")

    # Tour averages for comparison
    TOUR_AVG = {
        'first_serve_pct': 0.62,
        'ace_pct': 0.06,
        'df_pct': 0.04,
        'first_serve_win_pct': 0.72,
        'second_serve_win_pct': 0.53
    }

    # Main serve metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        delta = (stats.first_serve_pct - TOUR_AVG['first_serve_pct']) * 100
        st.metric("1st Serve %", f"{stats.first_serve_pct:.1%}",
                 delta=f"{delta:+.1f}%")

    with col2:
        delta = (stats.ace_pct - TOUR_AVG['ace_pct']) * 100
        st.metric("Ace %", f"{stats.ace_pct:.1%}",
                 delta=f"{delta:+.1f}%")

    with col3:
        delta = (stats.df_pct - TOUR_AVG['df_pct']) * 100
        st.metric("Double Fault %", f"{stats.df_pct:.1%}",
                 delta=f"{delta:+.1f}%", delta_color="inverse")

    st.markdown("---")

    # Serve win percentages
    col1, col2 = st.columns(2)

    with col1:
        # 1st vs 2nd serve win %
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Player',
            x=['1st Serve', '2nd Serve'],
            y=[stats.first_serve_win_pct * 100, stats.second_serve_win_pct * 100],
            marker_color='#1f77b4'
        ))

        fig.add_trace(go.Bar(
            name='Tour Average',
            x=['1st Serve', '2nd Serve'],
            y=[TOUR_AVG['first_serve_win_pct'] * 100, TOUR_AVG['second_serve_win_pct'] * 100],
            marker_color='#d62728'
        ))

        fig.update_layout(
            title=f"{player_name} - Serve Win %",
            yaxis_title="Win Percentage",
            barmode='group',
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Serve placement pattern (if available)
        if player_name in serve_patterns:
            pattern = serve_patterns[player_name]

            placements = ['Wide', 'T (Center)', 'Body']
            values = [
                pattern['wide_pct'],
                pattern['T_pct'],
                pattern['body_pct']
            ]

            fig = go.Figure(data=[go.Pie(
                labels=placements,
                values=values,
                hole=0.3,
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
            )])

            fig.update_layout(
                title=f"{player_name} - Serve Placement",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Serve placement data not available for this player")

    # Ace rate by placement (if available)
    if player_name in serve_patterns:
        pattern = serve_patterns[player_name]

        if 'ace_pct_by_placement' in pattern:
            ace_by_placement = pattern['ace_pct_by_placement']

            placements = ['Wide', 'T (Center)', 'Body']
            placement_codes = ['W', 'T', 'B']
            ace_rates = [ace_by_placement.get(code, 0) * 100 for code in placement_codes]

            fig = go.Figure(data=[go.Bar(
                x=placements,
                y=ace_rates,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                text=[f"{rate:.1f}%" for rate in ace_rates],
                textposition='auto'
            )])

            fig.update_layout(
                title=f"{player_name} - Ace Rate by Placement",
                yaxis_title="Ace Percentage",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)


def display_return_statistics(player_name, stats):
    """Display return statistics"""

    st.markdown("### 🔄 Return Statistics")

    # Tour averages
    TOUR_AVG = {
        'return_first_win_pct': 0.28,
        'return_second_win_pct': 0.47
    }

    col1, col2 = st.columns(2)

    with col1:
        delta = (stats.return_first_win_pct - TOUR_AVG['return_first_win_pct']) * 100
        st.metric("Return 1st Serve Win %", f"{stats.return_first_win_pct:.1%}",
                 delta=f"{delta:+.1f}%")

    with col2:
        delta = (stats.return_second_win_pct - TOUR_AVG['return_second_win_pct']) * 100
        st.metric("Return 2nd Serve Win %", f"{stats.return_second_win_pct:.1%}",
                 delta=f"{delta:+.1f}%")

    # Return win % comparison
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Player',
        x=['vs 1st Serve', 'vs 2nd Serve'],
        y=[stats.return_first_win_pct * 100, stats.return_second_win_pct * 100],
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        name='Tour Average',
        x=['vs 1st Serve', 'vs 2nd Serve'],
        y=[TOUR_AVG['return_first_win_pct'] * 100, TOUR_AVG['return_second_win_pct'] * 100],
        marker_color='#d62728'
    ))

    fig.update_layout(
        title=f"{player_name} - Return Performance",
        yaxis_title="Win Percentage",
        barmode='group',
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)


def display_rally_patterns(player_name, rally_patterns):
    """Display rally pattern statistics"""

    st.markdown("### 🎯 Rally Patterns")

    if player_name not in rally_patterns:
        st.info("📊 Rally pattern data not available for this player")
        return

    pattern = rally_patterns[player_name]

    col1, col2 = st.columns(2)

    with col1:
        avg_rally = pattern.get('avg_rally_length', 0)
        total_rallies = pattern.get('total_rallies', 0)

        st.metric("Average Rally Length", f"{avg_rally:.1f} shots")
        st.metric("Rallies Analyzed", f"{total_rallies:,}")

    with col2:
        # Rally length distribution
        if 'rally_length_dist' in pattern:
            dist = pattern['rally_length_dist']

            # Convert distribution to histogram data
            rally_lengths = list(range(len(dist)))
            probabilities = [d * 100 for d in dist]

            fig = go.Figure(data=[go.Bar(
                x=rally_lengths,
                y=probabilities,
                marker_color='#1f77b4'
            )])

            fig.update_layout(
                title=f"{player_name} - Rally Length Distribution",
                xaxis_title="Rally Length (shots)",
                yaxis_title="Frequency (%)",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)


def display_head_to_head(player_a, player_b, surface, stats_a, stats_b, elo_ratings):
    """Display head-to-head comparison between two players"""

    st.markdown("### ⚔️ Head-to-Head Comparison")

    # Elo-based win probability
    elo_a = get_player_elo(player_a, surface, elo_ratings)
    elo_b = get_player_elo(player_b, surface, elo_ratings)

    # Expected score formula
    expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(f"{player_a} Win Probability", f"{expected_a:.1%}")

    with col2:
        st.metric("Elo Difference", f"{abs(elo_a - elo_b):.0f}")

    with col3:
        st.metric(f"{player_b} Win Probability", f"{expected_b:.1%}")

    # Radar chart comparison
    categories = ['1st Serve %', 'Ace %', '1st Serve Win %',
                  '2nd Serve Win %', 'Return 1st %', 'Return 2nd %']

    # Normalize values to 0-100 scale
    values_a = [
        stats_a.first_serve_pct * 100,
        stats_a.ace_pct * 100 * 10,  # Scale up for visibility
        stats_a.first_serve_win_pct * 100,
        stats_a.second_serve_win_pct * 100,
        stats_a.return_first_win_pct * 100,
        stats_a.return_second_win_pct * 100
    ]

    values_b = [
        stats_b.first_serve_pct * 100,
        stats_b.ace_pct * 100 * 10,
        stats_b.first_serve_win_pct * 100,
        stats_b.second_serve_win_pct * 100,
        stats_b.return_first_win_pct * 100,
        stats_b.return_second_win_pct * 100
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_a,
        theta=categories,
        fill='toself',
        name=player_a,
        line_color='#1f77b4'
    ))

    fig.add_trace(go.Scatterpolar(
        r=values_b,
        theta=categories,
        fill='toself',
        name=player_b,
        line_color='#ff7f0e'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title=f"{player_a} vs {player_b} - Statistical Comparison",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side stats table
    st.markdown("#### Detailed Comparison")

    comparison_df = pd.DataFrame({
        'Metric': [
            'First Serve %',
            'Ace %',
            'Double Fault %',
            '1st Serve Win %',
            '2nd Serve Win %',
            'Return 1st Serve %',
            'Return 2nd Serve %'
        ],
        player_a: [
            f"{stats_a.first_serve_pct:.1%}",
            f"{stats_a.ace_pct:.1%}",
            f"{stats_a.df_pct:.1%}",
            f"{stats_a.first_serve_win_pct:.1%}",
            f"{stats_a.second_serve_win_pct:.1%}",
            f"{stats_a.return_first_win_pct:.1%}",
            f"{stats_a.return_second_win_pct:.1%}"
        ],
        player_b: [
            f"{stats_b.first_serve_pct:.1%}",
            f"{stats_b.ace_pct:.1%}",
            f"{stats_b.df_pct:.1%}",
            f"{stats_b.first_serve_win_pct:.1%}",
            f"{stats_b.second_serve_win_pct:.1%}",
            f"{stats_b.return_first_win_pct:.1%}",
            f"{stats_b.return_second_win_pct:.1%}"
        ]
    })

    st.dataframe(comparison_df, hide_index=True, use_container_width=True)


def create_export_data(player_name, surface, stats, elo_ratings, serve_patterns, rally_patterns):
    """Create exportable player statistics data"""

    elo = get_player_elo(player_name, surface, elo_ratings)

    export_data = {
        'Player': player_name,
        'Surface': surface or 'All',
        'Elo Rating': f"{elo:.0f}",
        'Matches Played': stats.matches_played,
        'First Serve %': f"{stats.first_serve_pct:.1%}",
        'Ace %': f"{stats.ace_pct:.1%}",
        'Double Fault %': f"{stats.df_pct:.1%}",
        '1st Serve Win %': f"{stats.first_serve_win_pct:.1%}",
        '2nd Serve Win %': f"{stats.second_serve_win_pct:.1%}",
        'Return 1st Serve Win %': f"{stats.return_first_win_pct:.1%}",
        'Return 2nd Serve Win %': f"{stats.return_second_win_pct:.1%}"
    }

    # Add serve patterns if available
    if player_name in serve_patterns:
        pattern = serve_patterns[player_name]
        export_data['Serve Wide %'] = f"{pattern['wide_pct']:.1%}"
        export_data['Serve T %'] = f"{pattern['T_pct']:.1%}"
        export_data['Serve Body %'] = f"{pattern['body_pct']:.1%}"

    # Add rally patterns if available
    if player_name in rally_patterns:
        pattern = rally_patterns[player_name]
        export_data['Avg Rally Length'] = f"{pattern.get('avg_rally_length', 0):.1f}"
        export_data['Total Rallies'] = pattern.get('total_rallies', 0)

    return pd.DataFrame([export_data])


def render():
    """Main render function for Player Stats page"""

    st.markdown("## 📈 Player Statistics")
    st.markdown("Explore detailed player statistics and patterns")

    # Load data
    matches, stats_calculator = load_data()
    if matches is None:
        return

    elo_ratings = load_elo_ratings()
    serve_patterns = load_serve_patterns()
    rally_patterns = load_rally_patterns()

    # Configuration panel
    st.markdown("### ⚙️ Configuration")

    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        players = load_player_list()
        # Default to Sinner
        default_index = players.index("Jannik Sinner") if "Jannik Sinner" in players else 0
        player_a = st.selectbox("Select Player", players, index=default_index, key="stats_player_a")

    with col2:
        surface = st.selectbox("Surface", ["All", "Hard", "Clay", "Grass"], key="stats_surface")

    with col3:
        comparison_mode = st.checkbox("Head-to-Head Mode", value=False)

    st.markdown("---")

    # Get player stats
    surface_filter = None if surface == "All" else surface
    stats_a = stats_calculator.get_player_stats(player_a, surface_filter)

    # Display player overview
    display_player_overview(player_a, surface_filter, stats_a, elo_ratings, matches)

    st.markdown("---")

    if comparison_mode:
        # Head-to-head comparison mode
        st.markdown("### Select Second Player")

        col1, col2 = st.columns([3, 2])

        with col1:
            player_b_options = [p for p in players if p != player_a]
            default_b = player_b_options.index("Carlos Alcaraz") if "Carlos Alcaraz" in player_b_options else 0
            player_b = st.selectbox("Opponent", player_b_options, index=default_b, key="stats_player_b")

        stats_b = stats_calculator.get_player_stats(player_b, surface_filter)

        display_head_to_head(player_a, player_b, surface_filter, stats_a, stats_b, elo_ratings)

    else:
        # Single player detailed view
        col1, col2 = st.columns(2)

        with col1:
            display_serve_statistics(player_a, surface_filter, stats_a, serve_patterns)

        with col2:
            display_return_statistics(player_a, stats_a)

        st.markdown("---")

        display_rally_patterns(player_a, rally_patterns)

    # Export functionality
    st.markdown("---")
    st.markdown("### 📥 Export Statistics")

    export_df = create_export_data(player_a, surface_filter, stats_a, elo_ratings,
                                   serve_patterns, rally_patterns)
    csv = export_df.to_csv(index=False)

    st.download_button(
        label=f"Download {player_a} Stats (CSV)",
        data=csv,
        file_name=f"player_stats_{player_a.replace(' ', '_')}.csv",
        mime="text/csv"
    )
