"""
Tournament Simulator UI - Tournament bracket simulation page

Interactive interface for running full tournament simulations with bracket visualization
and detailed match analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from typing import Dict, List, Tuple

from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator
from simulation.match_engine import MatchSimulator
from simulation.ml_predictor import MLMatchPredictor
from ui.bracket import AO_2025_MENS_R1


@st.cache_data
def load_match_data():
    """Load ATP match data (cached)"""
    loader = DataLoader(data_dir='data')
    matches = loader.load_match_data(years=[2020, 2021, 2022, 2023], tour='atp')
    if matches is None or len(matches) == 0:
        matches = loader._create_synthetic_data()
    return matches


@st.cache_resource
def load_ml_predictor():
    """Load ML match predictor model"""
    try:
        # Check if required packages are available
        try:
            import xgboost
            import sklearn
        except ImportError as ie:
            st.warning(f"ML dependencies not installed: {ie}. Install with: pip install xgboost scikit-learn")
            return None

        predictor = MLMatchPredictor(model_path='models/match_predictor_xgb.pkl')
        return predictor
    except Exception as e:
        st.warning(f"Could not load ML model: {e}")
        return None


@st.cache_resource
def load_elo_ratings():
    """Load ELO ratings dictionary with blended scores"""
    try:
        import json
        elo_path = 'models/elo_ratings_ao2025.json'
        if os.path.exists(elo_path):
            with open(elo_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.warning(f"Could not load ELO ratings: {e}")
        return {}


def get_blended_elo(player_name, surface, elo_ratings):
    """
    Get blended ELO rating: 50% overall + 50% surface-specific

    Args:
        player_name: Name of player
        surface: Surface type ('hard', 'clay', or 'grass')
        elo_ratings: Dictionary of ELO ratings

    Returns:
        Blended ELO rating (float), defaults to 1500 if not found
    """
    if not elo_ratings or player_name not in elo_ratings:
        return 1500.0

    player_data = elo_ratings[player_name]
    overall_elo = player_data.get('elo', 1500.0)
    surface_elo = player_data.get(surface, 1500.0)

    # 50/50 blend
    blended = (overall_elo + surface_elo) / 2.0
    return blended


def simulate_match_from_probability(player_a, player_b, stats_a, stats_b, surface, best_of, seed, win_prob_a):
    """
    Directly determine winner from win probability (from ML or ELO)
    Uses probability to pick winner, then runs ONE simulation to get realistic match stats
    Finally forces the winner to match the probability-based prediction
    """
    rng = np.random.default_rng(seed)

    # Determine winner based on probability
    player_a_wins = rng.random() < win_prob_a

    # Run a single simulation to get realistic match stats
    sim = MatchSimulator(
        player_a_stats=stats_a,
        player_b_stats=stats_b,
        best_of=best_of,
        seed=seed,
        track_point_history=False,
        surface=surface
    )

    match_result = sim.simulate_match()

    # Force the winner to match our probability-based prediction
    match_result.winner = player_a if player_a_wins else player_b

    return match_result


def get_tournament_data(tournament_name):
    """Get tournament bracket data"""
    if tournament_name == "Australian Open 2025 (Men's)":
        return AO_2025_MENS_R1, "hard", 5
    return None, None, None


def clean_player_name(name):
    """Remove seeding brackets and qualifiers from player names for ELO lookup"""
    import re
    # Remove patterns like [1], [32], (WC), (Q), [PR], etc.
    cleaned = re.sub(r'\s*\[.*?\]|\s*\(.*?\)', '', name).strip()
    return cleaned


def simulate_match(player_a, player_b, surface, best_of, seed, use_ml_prediction=False, elo_ratings=None):
    """Simulate a single match and return match stats"""

    # Load data (cached to avoid repeated loading)
    matches = load_match_data()

    # Get player stats
    stats_calc = PlayerStatsCalculator(matches)
    stats_a = stats_calc.get_player_stats(player_a, surface=surface)
    stats_b = stats_calc.get_player_stats(player_b, surface=surface)

    # If using ML/ELO prediction, use probability-based outcome
    if use_ml_prediction:
        # Clean player names for ELO lookup (remove [1], (WC), etc.)
        clean_name_a = clean_player_name(player_a)
        clean_name_b = clean_player_name(player_b)

        # Get blended ELO ratings (50% overall + 50% surface-specific)
        elo_a = get_blended_elo(clean_name_a, surface, elo_ratings) if elo_ratings else 1500
        elo_b = get_blended_elo(clean_name_b, surface, elo_ratings) if elo_ratings else 1500

        # Use pure ELO formula for win probability
        # This ensures the blended ELO ratings are the primary driver of predictions
        win_prob_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

        # Use probability to determine winner and get realistic stats
        match_result = simulate_match_from_probability(
            player_a, player_b, stats_a, stats_b, surface, best_of,
            seed, win_prob_a
        )

        # Store win probabilities in the match result for display
        match_result.win_prob_a = win_prob_a
        match_result.win_prob_b = 1.0 - win_prob_a

        return match_result

    # Create simulator (traditional Monte Carlo approach - point-level only)
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


def simulate_tournament(r1_matchups, surface, best_of, base_seed=42, use_ml_prediction=False, elo_ratings=None):
    """Simulate entire tournament bracket"""

    # Track all results
    results = {
        'R1': {},  # Round 1 (R128)
        'R2': {},  # Round 2 (R64)
        'R3': {},  # Round 3 (R32)
        'R4': {},  # Round 4 (R16)
        'QF': {},  # Quarter Finals
        'SF': {},  # Semi Finals
        'F': {}    # Final
    }

    # Simulate Round 1
    r1_winners = {}
    for match_id, (player1, player2) in r1_matchups.items():
        seed = base_seed + match_id
        match_result = simulate_match(player1, player2, surface, best_of, seed, use_ml_prediction, elo_ratings)
        results['R1'][match_id] = match_result
        r1_winners[match_id] = match_result.winner

    # Simulate Round 2 (R64) - pairs of R1 winners
    r2_winners = {}
    r2_match_id = 1
    for i in range(1, len(r1_matchups) + 1, 2):
        if i in r1_winners and (i + 1) in r1_winners:
            player1 = r1_winners[i]
            player2 = r1_winners[i + 1]
            seed = base_seed + 1000 + r2_match_id
            match_result = simulate_match(player1, player2, surface, best_of, seed, use_ml_prediction, elo_ratings)
            results['R2'][r2_match_id] = match_result
            r2_winners[r2_match_id] = match_result.winner
            r2_match_id += 1

    # Simulate Round 3 (R32)
    r3_winners = {}
    r3_match_id = 1
    for i in range(1, len(r2_winners) + 1, 2):
        if i in r2_winners and (i + 1) in r2_winners:
            player1 = r2_winners[i]
            player2 = r2_winners[i + 1]
            seed = base_seed + 2000 + r3_match_id
            match_result = simulate_match(player1, player2, surface, best_of, seed, use_ml_prediction, elo_ratings)
            results['R3'][r3_match_id] = match_result
            r3_winners[r3_match_id] = match_result.winner
            r3_match_id += 1

    # Simulate Round 4 (R16)
    r4_winners = {}
    r4_match_id = 1
    for i in range(1, len(r3_winners) + 1, 2):
        if i in r3_winners and (i + 1) in r3_winners:
            player1 = r3_winners[i]
            player2 = r3_winners[i + 1]
            seed = base_seed + 3000 + r4_match_id
            match_result = simulate_match(player1, player2, surface, best_of, seed, use_ml_prediction, elo_ratings)
            results['R4'][r4_match_id] = match_result
            r4_winners[r4_match_id] = match_result.winner
            r4_match_id += 1

    # Simulate Quarter Finals
    qf_winners = {}
    qf_match_id = 1
    for i in range(1, len(r4_winners) + 1, 2):
        if i in r4_winners and (i + 1) in r4_winners:
            player1 = r4_winners[i]
            player2 = r4_winners[i + 1]
            seed = base_seed + 4000 + qf_match_id
            match_result = simulate_match(player1, player2, surface, best_of, seed, use_ml_prediction, elo_ratings)
            results['QF'][qf_match_id] = match_result
            qf_winners[qf_match_id] = match_result.winner
            qf_match_id += 1

    # Simulate Semi Finals
    sf_winners = {}
    sf_match_id = 1
    for i in range(1, len(qf_winners) + 1, 2):
        if i in qf_winners and (i + 1) in qf_winners:
            player1 = qf_winners[i]
            player2 = qf_winners[i + 1]
            seed = base_seed + 5000 + sf_match_id
            match_result = simulate_match(player1, player2, surface, best_of, seed, use_ml_prediction, elo_ratings)
            results['SF'][sf_match_id] = match_result
            sf_winners[sf_match_id] = match_result.winner
            sf_match_id += 1

    # Simulate Final
    if 1 in sf_winners and 2 in sf_winners:
        player1 = sf_winners[1]
        player2 = sf_winners[2]
        seed = base_seed + 6000
        match_result = simulate_match(player1, player2, surface, best_of, seed, use_ml_prediction, elo_ratings)
        results['F'][1] = match_result

    return results


def get_match_html(player_a, player_b, winner, score, round_name, match_id, win_prob_a=None, win_prob_b=None):
    """Generate HTML for a match card in the bracket"""

    winner_a = winner == player_a
    winner_b = winner == player_b

    # Truncate long names
    def truncate_name(name, max_len=17):
        if len(name) > max_len:
            return name[:max_len-3] + "..."
        return name

    player_a_display = truncate_name(player_a)
    player_b_display = truncate_name(player_b)

    # Format win probabilities if available
    prob_a_display = f" ({win_prob_a:.0%})" if win_prob_a is not None else ""
    prob_b_display = f" ({win_prob_b:.0%})" if win_prob_b is not None else ""

    # Create unique match ID for clicking
    match_key = f"{round_name}_{match_id}"

    html = f"""
    <div class="match-card" data-match="{match_key}">
        <div class="match-player {'winner' if winner_a else 'loser'}">
            <span class="player-name">{player_a_display}</span>
            <span class="win-prob">{prob_a_display}</span>
        </div>
        <div class="match-player {'winner' if winner_b else 'loser'}">
            <span class="player-name">{player_b_display}</span>
            <span class="win-prob">{prob_b_display}</span>
        </div>
    </div>
    """
    return html


def display_bracket_horizontal(results, r1_matchups):
    """Display tournament bracket in horizontal scrollable format with clickable buttons"""

    # Create columns for each round
    rounds_data = [
        ('R1', 'Round 1', results.get('R1', {})),
        ('R2', 'Round 2', results.get('R2', {})),
        ('R3', 'Round 3', results.get('R3', {})),
        ('R4', 'Round of 16', results.get('R4', {})),
        ('QF', 'Quarter Finals', results.get('QF', {})),
        ('SF', 'Semi Finals', results.get('SF', {})),
        ('F', 'Final', results.get('F', {}))
    ]

    # Filter out empty rounds
    rounds_data = [(rk, rn, rd) for rk, rn, rd in rounds_data if rd]

    # Create columns
    cols = st.columns(len(rounds_data))

    for idx, (round_key, round_name, round_results) in enumerate(rounds_data):
        with cols[idx]:
            st.markdown(f"**{round_name}**")

            for match_id in sorted(round_results.keys()):
                match_result = round_results[match_id]

                # Get player names
                if round_key == 'R1':
                    player_a, player_b = r1_matchups[match_id]
                else:
                    player_a = match_result.player_a_name
                    player_b = match_result.player_b_name

                # Get win probabilities
                win_prob_a = getattr(match_result, 'win_prob_a', None)
                win_prob_b = getattr(match_result, 'win_prob_b', None)

                # Truncate names
                def truncate(name, max_len=15):
                    return name[:max_len-2] + ".." if len(name) > max_len else name

                # Determine favorite/underdog and add color indicators
                if win_prob_a and win_prob_b:
                    if win_prob_a > 0.5:
                        # Player A is favorite (green), Player B is underdog (red)
                        prob_a = f" 🟢 {win_prob_a:.0%}"
                        prob_b = f" 🔴 {win_prob_b:.0%}"
                    else:
                        # Player B is favorite (green), Player A is underdog (red)
                        prob_a = f" 🔴 {win_prob_a:.0%}"
                        prob_b = f" 🟢 {win_prob_b:.0%}"
                else:
                    prob_a = f" ({win_prob_a:.0%})" if win_prob_a else ""
                    prob_b = f" ({win_prob_b:.0%})" if win_prob_b else ""

                # Create match display text
                winner = match_result.winner
                if winner == player_a:
                    match_text = f"**{truncate(player_a)}{prob_a}** ✓\n\n{truncate(player_b)}{prob_b}"
                else:
                    match_text = f"{truncate(player_a)}{prob_a}\n\n**{truncate(player_b)}{prob_b}** ✓"

                # Create button
                if st.button(match_text, key=f"match_{round_key}_{match_id}", width='stretch'):
                    st.session_state.selected_match = (round_key, match_id)
                    st.rerun()

                st.markdown("")  # Small gap between matches


def display_match_stats(match_result, player_a, player_b):
    """Display match statistics (used by both modal and expanders)"""

    # Winner announcement
    st.markdown(f"**🏆 Winner:** {match_result.winner}")
    st.markdown(f"**Score:** {match_result.score}")

    st.markdown("---")

    # Match summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Sets", f"{match_result.sets_won_a}-{match_result.sets_won_b}")

    with col2:
        st.metric("Total Points", match_result.total_points)

    with col3:
        points_won_a = match_result.total_points_won_a
        points_won_b = match_result.total_points_won_b
        st.metric("Points Won", f"{points_won_a}-{points_won_b}")

    st.markdown("---")

    # Detailed stats table
    stats_data = {
        'Statistic': [
            'Aces',
            'Double Faults',
            '1st Serve In',
            '1st Serve %',
            '1st Serve Points Won',
            '2nd Serve Points Won',
            'Break Points Won'
        ],
        player_a: [
            str(match_result.aces_a),
            str(match_result.double_faults_a),
            f"{match_result.first_serves_in_a}/{match_result.first_serves_total_a}",
            f"{match_result.first_serves_in_a / max(1, match_result.first_serves_total_a) * 100:.1f}%",
            f"{match_result.first_serve_points_won_a}/{match_result.first_serve_points_total_a}",
            f"{match_result.second_serve_points_won_a}/{match_result.second_serve_points_total_a}",
            f"{match_result.break_points_won_a}/{match_result.break_points_total_b}"
        ],
        player_b: [
            str(match_result.aces_b),
            str(match_result.double_faults_b),
            f"{match_result.first_serves_in_b}/{match_result.first_serves_total_b}",
            f"{match_result.first_serves_in_b / max(1, match_result.first_serves_total_b) * 100:.1f}%",
            f"{match_result.first_serve_points_won_b}/{match_result.first_serve_points_total_b}",
            f"{match_result.second_serve_points_won_b}/{match_result.second_serve_points_total_b}",
            f"{match_result.break_points_won_b}/{match_result.break_points_total_a}"
        ]
    }

    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, hide_index=True, width='stretch')


@st.dialog("Match Details")
def display_match_details_modal(match_result, player_a, player_b):
    """Display detailed match statistics in a modal dialog"""

    st.markdown(f"### {player_a} vs {player_b}")
    display_match_stats(match_result, player_a, player_b)

    # Close button
    if st.button("Close", type="primary", width='stretch'):
        st.session_state.selected_match = None
        st.rerun()


def display_match_details_expander(match_result, player_a, player_b, round_name, match_id):
    """Display detailed match statistics in an expander"""

    with st.expander(f"📊 {round_name} - Match {match_id}: {player_a} vs {player_b}", expanded=False):
        display_match_stats(match_result, player_a, player_b)


def render():
    """Render the Tournament Simulator page"""

    st.markdown('<p class="main-header">🏆 Tournament Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Simulate complete tournament brackets</p>', unsafe_allow_html=True)

    # Tournament selection
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        tournament = st.selectbox(
            "Select Tournament",
            ["Australian Open 2025 (Men's)"],
            key="tournament_select"
        )

    with col2:
        use_ml_prediction = st.checkbox(
            "Use ELO Predictions",
            value=True,
            key="tournament_ml",
            help="Use blended ELO ratings (50% overall + 50% surface-specific) to predict match outcomes"
        )

    with col3:
        random_seed = st.checkbox(
            "Random Results",
            value=True,
            key="tournament_random",
            help="Get different results each time (unchecked = same seed for reproducibility)"
        )

    # Get tournament data
    r1_matchups, surface, best_of = get_tournament_data(tournament)

    if r1_matchups is None:
        st.error("Tournament data not available")
        return

    # Display tournament info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Surface:** {surface.capitalize()}")

    with col2:
        st.info(f"**Format:** Best of {best_of}")

    with col3:
        total_matches = len(r1_matchups) + 32 + 16 + 8 + 4 + 2 + 1
        st.info(f"**Total Matches:** {total_matches}")

    # Simulate button
    if st.button("🎾 Simulate Tournament", type="primary", width='stretch'):

        # Load ELO ratings if requested
        elo_ratings = None
        if use_ml_prediction:
            with st.spinner("Loading ELO ratings..."):
                elo_ratings = load_elo_ratings()

        with st.spinner(f"Simulating {total_matches} matches... This may take a few minutes..."):
            progress_bar = st.progress(0)

            # Determine seed: random if requested, otherwise fixed for reproducibility
            import time
            seed = int(time.time() * 1000) % 1000000 if random_seed else 42

            # Run tournament simulation
            results = simulate_tournament(
                r1_matchups, surface, best_of,
                base_seed=seed,
                use_ml_prediction=use_ml_prediction,
                elo_ratings=elo_ratings
            )

            # Store in session state
            st.session_state.tournament_results = results
            st.session_state.tournament_r1_matchups = r1_matchups
            st.session_state.tournament_name = tournament

            progress_bar.progress(100)

        st.success(f"✅ Tournament simulation complete!")

        # Display champion
        if 'F' in results and 1 in results['F']:
            champion = results['F'][1].winner
            st.markdown(f"## 🏆 Champion: **{champion}**")

    # Display results if available
    if 'tournament_results' in st.session_state:
        results = st.session_state.tournament_results
        r1_matchups = st.session_state.tournament_r1_matchups

        st.markdown("---")

        # Add CSS for champion banner
        st.markdown("""
        <style>
            .champion-banner {
                background: linear-gradient(135deg, #ffd700, #ffed4e);
                border: 3px solid #daa520;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
        </style>
        """, unsafe_allow_html=True)

        # Display champion banner
        if 'F' in results and 1 in results['F']:
            champion = results['F'][1].winner
            st.markdown(f'<div class="champion-banner">🏆 Champion: {champion} 🏆</div>', unsafe_allow_html=True)

        st.markdown("## Tournament Bracket")
        st.markdown("*Click on any match button to view detailed statistics*")

        # Initialize selected match in session state
        if 'selected_match' not in st.session_state:
            st.session_state.selected_match = None

        # Display horizontal bracket
        display_bracket_horizontal(results, r1_matchups)

        # Show match details modal if a match is selected
        if st.session_state.selected_match:
            round_key, match_id = st.session_state.selected_match

            # Get the match result
            if round_key in results and match_id in results[round_key]:
                match_result = results[round_key][match_id]

                # Get player names based on round
                if round_key == 'R1':
                    player_a, player_b = r1_matchups[match_id]
                else:
                    player_a = match_result.player_a_name
                    player_b = match_result.player_b_name

                # Display modal
                display_match_details_modal(match_result, player_a, player_b)

        # Match details section
        st.markdown("## 📋 All Match Details")
        st.markdown("*Expand any match below to view detailed statistics*")

        # Create tabs for each round
        tabs = st.tabs(["Final", "Semi Finals", "Quarter Finals", "Round of 16", "Round of 32", "Round of 64", "Round of 128"])

        # Final
        with tabs[0]:
            if 'F' in results and results['F'] and 1 in results['F']:
                match_result = results['F'][1]
                display_match_details_expander(match_result, match_result.player_a_name, match_result.player_b_name, "Final", 1)
            else:
                st.info("No final match data")

        # Semi Finals
        with tabs[1]:
            if 'SF' in results and results['SF']:
                for match_id in sorted(results['SF'].keys()):
                    match_result = results['SF'][match_id]
                    display_match_details_expander(match_result, match_result.player_a_name, match_result.player_b_name, "Semi Finals", match_id)
            else:
                st.info("No semi finals data")

        # Quarter Finals
        with tabs[2]:
            if 'QF' in results and results['QF']:
                for match_id in sorted(results['QF'].keys()):
                    match_result = results['QF'][match_id]
                    display_match_details_expander(match_result, match_result.player_a_name, match_result.player_b_name, "Quarter Finals", match_id)
            else:
                st.info("No quarter finals data")

        # Round of 16
        with tabs[3]:
            if 'R4' in results and results['R4']:
                for match_id in sorted(results['R4'].keys()):
                    match_result = results['R4'][match_id]
                    display_match_details_expander(match_result, match_result.player_a_name, match_result.player_b_name, "Round of 16", match_id)
            else:
                st.info("No round of 16 data")

        # Round of 32
        with tabs[4]:
            if 'R3' in results and results['R3']:
                for match_id in sorted(results['R3'].keys()):
                    match_result = results['R3'][match_id]
                    display_match_details_expander(match_result, match_result.player_a_name, match_result.player_b_name, "Round of 32", match_id)
            else:
                st.info("No round of 32 data")

        # Round of 64
        with tabs[5]:
            if 'R2' in results and results['R2']:
                for match_id in sorted(results['R2'].keys()):
                    match_result = results['R2'][match_id]
                    display_match_details_expander(match_result, match_result.player_a_name, match_result.player_b_name, "Round of 64", match_id)
            else:
                st.info("No round of 64 data")

        # Round of 128
        with tabs[6]:
            if 'R1' in results and results['R1']:
                for match_id in sorted(results['R1'].keys()):
                    match_result = results['R1'][match_id]
                    player_a, player_b = r1_matchups[match_id]
                    display_match_details_expander(match_result, player_a, player_b, "Round of 128", match_id)
            else:
                st.info("No round of 128 data")

        # Export option
        st.markdown("---")
        st.markdown("### Export Results")

        # Compile all results into exportable format
        export_data = []
        for round_name, matches in results.items():
            for match_id, match_result in matches.items():
                export_data.append({
                    'Round': round_name,
                    'Match ID': match_id,
                    'Winner': match_result.winner,
                    'Score': match_result.score,
                    'Sets': f"{match_result.sets_won_a}-{match_result.sets_won_b}",
                    'Total Points': match_result.total_points,
                    'Aces Winner': match_result.aces_a if match_result.winner == match_result.player_a_name else match_result.aces_b,
                    'Aces Loser': match_result.aces_b if match_result.winner == match_result.player_a_name else match_result.aces_a
                })

        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False)

        st.download_button(
            label="📥 Download Tournament Results",
            data=csv,
            file_name=f"tournament_{tournament.replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')}.csv",
            mime="text/csv",
            width='stretch'
        )
