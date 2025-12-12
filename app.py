"""
Tennis Simulation Lab - Streamlit Web Interface

Main entry point for the interactive tennis match simulation UI.
Provides browser-based interface for running simulations, comparing models,
and analyzing player statistics.

Usage:
    streamlit run app.py

Then open http://localhost:8501 in your browser.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# Import UI pages
from ui import match_simulator, single_match_viewer, model_comparison

# Page configuration
st.set_page_config(
    page_title="Tennis Simulation Lab",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🎾 Tennis Simulation Lab")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Match Simulator", "Single Match Viewer", "Model Comparison", "Player Stats"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Interactive interface for tennis match simulation using:
    - **Point-Level**: Fast Monte Carlo simulation
    - **Shot-Level**: Detailed shot-by-shot modeling with Match Charting Project data

    Built on Jeff Sackmann's ATP/WTA datasets and the Match Charting Project.
    """)

    st.markdown("---")
    st.markdown("### Quick Tips")
    st.markdown("""
    - 💡 Use shot-level for detailed analysis
    - ⚡ Use point-level for fast batch runs
    - 📊 Export results to CSV for further analysis
    - 🔄 Refresh page to reset state
    """)

# Main content area
if page == "Match Simulator":
    match_simulator.render()

elif page == "Single Match Viewer":
    single_match_viewer.render()

elif page == "Model Comparison":
    model_comparison.render()

elif page == "Player Stats":
    st.markdown('<p class="main-header">📈 Player Statistics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore player patterns and statistics</p>', unsafe_allow_html=True)
    st.info("🚧 Coming soon! This page will show detailed player stats.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Tennis Simulation Lab | Built with Streamlit | Data from Jeff Sackmann & Match Charting Project
</div>
""", unsafe_allow_html=True)
