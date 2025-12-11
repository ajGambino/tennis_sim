# Tennis Match Simulation Framework

A comprehensive point-level and shot-by-shot Monte Carlo tennis simulation engine built with Python, using Jeff Sackmann's public tennis datasets.

## Features

- **Point-level simulation**: Every point simulated using probabilistic serve/return models
- **Shot-level simulation (NEW)**: Shot-by-shot rally simulation with serve placement, return quality, and tactical modeling
- **Historical data integration**: Uses real ATP/WTA match statistics from Jeff Sackmann's datasets
- **Match Charting Project support**: Player-specific shot patterns from 17,000+ charted matches
- **Full tennis rules**: Advantage scoring, tiebreaks, set/match logic
- **Monte Carlo analysis**: Run thousands of simulations to compute win probabilities and distributions
- **Comprehensive statistics**: Aces, double faults, break points, first serve %, rally lengths, and more
- **Surface-specific stats**: Aggregate player performance by court surface (hard, clay, grass)
- **Elo rating system**: Skill-based adjustments for realistic match outcomes
- **ML-powered predictions**: XGBoost model for match outcome prediction
- **Interactive Streamlit UI**: Modern web interface with real-time simulation and visualization
- **Elo-sorted player selection**: Players ranked by strength in dropdown menus
- **Paginated shot replay**: Browse shot-by-shot details with 20 shots per page
- **Reproducible results**: All simulations use random seeds for exact reproducibility

## Phase Evolution

- **Phase 0:** Basic point-level simulation with static player stats
- **Phase 1:** Elo rating system integration for skill differentiation ([PHASE_1_COMPLETE.md](xReferences/PHASE_1_COMPLETE.md))
- **Phase 2:** ML-powered parameter estimation with XGBoost ([PHASE_2_READY.md](xReferences/PHASE_2_READY.md))
- **Phase 3A:** Shot-by-shot simulation with Match Charting Project integration ([PHASE_3A_COMPLETE.md](xReferences/PHASE_3A_COMPLETE.md))
- **Phase 3B:** Interactive Streamlit UI with Elo-sorted players and advanced visualizations ← **YOU ARE HERE**

## Installation

### 1. Clone or download this repository

```bash
git clone <your-repo-url>
cd tennis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requirements:

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- streamlit >= 1.28.0 (for web UI)
- plotly >= 5.18.0 (for visualizations)
- xgboost >= 2.0.0 (for ML predictions)
- scikit-learn >= 1.4.0 (for ML features)

### 3. Download tennis data (optional)

The framework includes synthetic demo data if no real data is available. To use actual historical data:

**Option A: Download Jeff Sackmann's ATP data**

```bash
git clone https://github.com/JeffSackmann/tennis_atp.git
cp tennis_atp/atp_matches_*.csv data/
```

**Option B: Download WTA data**

```bash
git clone https://github.com/JeffSackmann/tennis_wta.git
cp tennis_wta/wta_matches_*.csv data/
```

The framework will automatically load any CSV files matching the pattern `atp_matches_YYYY.csv` or `wta_matches_YYYY.csv` in the `data/` directory.

## Quick Start

### Streamlit Web UI (Recommended)

The easiest way to use the simulator is through the interactive Streamlit interface:

```bash
streamlit run app.py
```

This launches a web application at `http://localhost:8501` with two main pages:

**Page 1: Match Simulator**
- Run bulk Monte Carlo simulations (10-1000 matches)
- Select players from Elo-ranked dropdowns (strongest to weakest)
- Choose surface (hard, clay, grass) and format (best of 3 or 5)
- Enable shot-level simulation for detailed rally analysis
- View win probability distributions and match statistics
- See most common scores with frequency charts
- Export results to CSV

**Page 2: Single Match Viewer**
- Simulate one detailed match with complete statistics
- Set-by-set progression charts
- Performance comparison radar charts
- Detailed statistics tables
- **Shot-by-shot replay viewer** with pagination (20 shots per page)
- Filter points by rally length and server
- Expand individual rallies to see every shot with type, direction, position, and outcome
- Reproducible with custom random seeds

### Command Line Interface

For scripted or automated workflows, you can use the CLI tools:

#### Single Match Simulation

Simulate one match and see detailed statistics:

```bash
python xMonteCarlo/run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz"
```

Output includes:

- Full match box score
- Aces, double faults, first serve %, points won
- Break points won/saved
- Final score
- Random seed for reproducibility

**Options:**

```bash
# specify surface
python xMonteCarlo/run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --surface clay

# best of 5 sets
python xMonteCarlo/run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --best_of 5

# reproducible simulation with seed
python xMonteCarlo/run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --seed 12345
```

#### Bulk Monte Carlo Simulation

Run thousands of simulations to compute win probabilities:

```bash
python xMonteCarlo/run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000
```

Output includes:

- Win probability for each player
- Score distribution (most common final scores)
- Sets won distribution
- Average match statistics (aces, DFs, etc.)
- Full results saved to CSV in `results/` directory

**Options:**

```bash
# surface-specific simulation
python xMonteCarlo/run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000 --surface hard

# best of 5 sets
python xMonteCarlo/run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000 --best_of 5

# custom output filename
python xMonteCarlo/run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000 --output my_results.csv

# don't save csv (just print summary)
python xMonteCarlo/run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 1000 --no_save
```

## Project Structure

```
tennis/
├── app.py                          # Main Streamlit application entry point
├── ui/                             # Streamlit UI components
│   ├── match_simulator.py         # Bulk Monte Carlo simulation page
│   └── single_match_viewer.py     # Single match detailed analysis page
├── simulation/                     # Core simulation engine
│   ├── __init__.py                # Package exports
│   ├── data_loader.py             # Load and preprocess Sackmann CSVs
│   ├── player_stats.py            # Compute player probability parameters
│   ├── point_engine.py            # Simulate individual points
│   ├── match_engine.py            # Game/set/match logic with full rules
│   ├── simulator.py               # Bulk simulation runner
│   ├── elo_system.py              # Elo rating system
│   ├── ml_predictor.py            # XGBoost match prediction
│   ├── feature_engineering.py     # ML feature extraction
│   ├── ranking_adjuster.py        # Ranking-based adjustments
│   ├── charting_loader.py         # Match Charting Project data loader
│   ├── serve_model.py             # Serve placement and outcome modeling
│   ├── return_model.py            # Return quality modeling
│   ├── rally_model.py             # Rally progression and shot selection
│   └── shot.py                    # Shot data structures and enums
├── training/                       # ML model training scripts
│   ├── train_point_model.py       # Train XGBoost point-level model
│   ├── train_serve_model.py       # Train serve pattern models
│   └── train_rally_model.py       # Train rally pattern models
├── data/                           # Tennis match data
│   ├── (atp_matches_YYYY.csv)     # Jeff Sackmann ATP match files
│   └── charting/                  # Match Charting Project files (optional)
├── models/                         # Trained models and patterns
│   ├── elo_ratings_atp.json       # Pre-computed Elo ratings
│   ├── serve_patterns.pkl         # Trained serve models
│   └── rally_patterns.pkl         # Trained rally models
├── results/                        # Simulation output CSVs
├── analysis/                       # Analysis notebooks and scripts
├── xMonteCarlo/                    # CLI tools for Monte Carlo simulation
│   ├── run_single.py              # Single match CLI
│   ├── run_bulk.py                # Bulk simulation CLI
│   └── predict_match.py           # ML prediction CLI
├── xTesting/                       # Testing and comparison scripts
│   ├── compare_shot_vs_point.py   # Compare simulation methods
│   ├── compare_all_methods.py     # Full method comparison
│   └── test_elo.py                # Elo system tests
├── xReferences/                    # Documentation and guides
│   ├── PHASE_1_COMPLETE.md        # Elo integration documentation
│   ├── PHASE_2_READY.md           # ML prediction documentation
│   ├── PHASE_3A_COMPLETE.md       # Shot-level simulation documentation
│   ├── QUICK_REFERENCE.md         # Quick reference guide
│   └── example_api_usage.py       # API usage examples
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## How It Works

### 1. Data Loading

The `data_loader.py` module:

- Loads CSV files from Jeff Sackmann's repositories
- Supports both ATP and WTA data
- Handles multiple years of data
- Creates synthetic demo data if real data is unavailable
- Cleans and standardizes surface names and statistics

### 2. Player Statistics Calculation

The `player_stats.py` module computes probabilistic parameters for each player:

**Serve Statistics:**

- First serve % = `1stIn / svpt`
- Ace % = `ace / svpt`
- Double fault % = `df / (svpt - 1stIn)`
- First serve win % = `1stWon / 1stIn`
- Second serve win % = `2ndWon / (svpt - 1stIn)`

**Return Statistics:**

- Return first serve win % = `1 - opponent_first_serve_win_%`
- Return second serve win % = `1 - opponent_second_serve_win_%`

Statistics can be aggregated:

- By surface (hard, clay, grass, carpet)
- Across all surfaces
- With Laplace smoothing to blend actual stats with tour averages
- Fallback to tour averages when insufficient data available

### 3. Point Simulation

The `point_engine.py` module simulates each point:

1. Check for ace (based on server ace %)
2. Determine if first serve goes in (based on first serve %)
3. If first serve in:
   - Blend server's first serve win % with returner's return first win %
   - Simulate point outcome
4. If first serve out:
   - Check for double fault (based on DF %)
   - If not DF, blend second serve probabilities and simulate

The blending gives 60% weight to server stats and 40% to returner stats, reflecting that the server has more control over the point.

### 4. Match Simulation

The `match_engine.py` module implements full tennis rules:

**Game Logic:**

- Standard advantage scoring (0, 15, 30, 40, deuce, advantage)
- Tracks break points (when returner is one point from breaking)

**Set Logic:**

- First to 6 games, must win by 2
- Tiebreak at 6-6 (first to 7 points, win by 2)
- Server alternates every point in tiebreak (after first point)

**Match Logic:**

- Best of 3 or best of 5 sets
- Comprehensive statistics tracking for both players
- Random selection of who serves first

### 5. Bulk Simulation

The `simulator.py` module:

- Runs N independent match simulations
- Uses sequential seeds for reproducibility
- Aggregates results into pandas DataFrame
- Computes summary statistics:
  - Win probabilities
  - Score distributions
  - Average/median for all stats
  - Sets won distributions

## Data Sources

This framework uses data from Jeff Sackmann's tennis repositories:

- **ATP Data**: https://github.com/JeffSackmann/tennis_atp
- **WTA Data**: https://github.com/JeffSackmann/tennis_wta
- **Point-by-point Data**: https://github.com/JeffSackmann/tennis_MatchChartingProject

These datasets are updated regularly and contain:

- Match-level statistics from 1991-present (tour-level)
- Serve statistics, break points, aces, double faults, etc.
- Tournament info, surface, date, player names

## Example Output

### Single Match

```
==================================================================
Player: Novak Djokovic
Surface: hard
Matches: 487
==================================================================

Serve Statistics:
  First Serve %:        62.3%
  Ace %:                5.8%
  Double Fault %:       3.2%
  1st Serve Win %:      73.1%
  2nd Serve Win %:      55.4%

Return Statistics:
  Return 1st Serve Win %: 29.8%
  Return 2nd Serve Win %: 48.2%
==================================================================

[... Player B stats ...]

======================================================================
Novak Djokovic vs Carlos Alcaraz
Winner: Novak Djokovic
Score: 6-4, 4-6, 7-6
======================================================================

Statistic                 Novak Djokovic  Carlos Alcaraz
----------------------------------------------------------------------
Aces                                   7              12
Double Faults                          3               4
First Serve %                      61.2%          65.3%
1st Serve Points Won              54/76          48/82
2nd Serve Points Won              23/49          19/44
Break Points Won                    3/8            2/7
Total Points Won                    102             95
======================================================================

random seed used: 1847293847
```

### Bulk Simulation

```
======================================================================
MONTE CARLO SIMULATION RESULTS
======================================================================

Total Simulations: 5,000

Novak Djokovic vs Carlos Alcaraz
----------------------------------------------------------------------

Win Probability:
  Novak Djokovic                    56.24% (2,812 wins)
  Carlos Alcaraz                    43.76% (2,188 wins)

Most Common Scores:
  6-4, 4-6, 6-4                     8.32%
  6-3, 6-4                          7.89%
  6-4, 6-3                          7.12%
  4-6, 6-3, 6-4                     6.54%
  6-2, 6-4                          5.87%

Sets Won Distribution (Novak Djokovic):
  0 sets: 6.23%
  1 set: 37.53%
  2 sets: 56.24%

Average Match Statistics:
Statistic                 Novak Djokovic  Carlos Alcaraz
----------------------------------------------------------------------
Aces (mean)                          8.7            10.2
Aces (median)                        8.0            10.0
Double Faults (mean)                 3.4             3.8
Double Faults (median)               3.0             4.0
First Serve % (mean)                62.1%          64.8%
Total Points (mean)                 93.2            89.1
======================================================================
```

## Advanced Usage

### Programmatic API

You can also use the framework programmatically in your own Python code:

```python
from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator
from simulation.match_engine import MatchSimulator
from simulation.point_engine import PointSimulator
from simulation.serve_model import ServeModel
from simulation.return_model import ReturnModel
from simulation.rally_model import RallyModel
import pickle

# Load data
loader = DataLoader(data_dir='data')
matches = loader.load_match_data(years=[2020, 2021, 2022, 2023], tour='atp')

# Calculate player stats
stats_calc = PlayerStatsCalculator(matches)
djokovic_stats = stats_calc.get_player_stats('Novak Djokovic', surface='hard')
alcaraz_stats = stats_calc.get_player_stats('Carlos Alcaraz', surface='hard')

# Single point-level simulation
sim = MatchSimulator(
    player_a_stats=djokovic_stats,
    player_b_stats=alcaraz_stats,
    best_of=3,
    seed=42
)
result = sim.simulate_match()
print(f"Winner: {result.winner}")
print(f"Score: {result.score}")

# Shot-level simulation with trained models
with open('models/serve_patterns.pkl', 'rb') as f:
    serve_data = pickle.load(f)
with open('models/rally_patterns.pkl', 'rb') as f:
    rally_data = pickle.load(f)

serve_model = ServeModel(serve_patterns=serve_data.get('serve_patterns', {}))
return_model = ReturnModel()
rally_model = RallyModel(rally_patterns=rally_data.get('rally_patterns', {}))

point_sim = PointSimulator(
    use_shot_simulation=True,
    serve_model=serve_model,
    return_model=return_model,
    rally_model=rally_model
)

sim_detailed = MatchSimulator(
    player_a_stats=djokovic_stats,
    player_b_stats=alcaraz_stats,
    best_of=3,
    seed=42,
    point_simulator=point_sim,
    track_point_history=True
)
result = sim_detailed.simulate_match()

# Access shot-by-shot details
for point in result.point_history:
    print(f"Rally length: {point.rally_length}")
    if point.rally and point.rally.shots:
        for shot in point.rally.shots:
            print(f"  Shot {shot.shot_number}: {shot.shot_type} -> {shot.outcome}")
```

### Custom Player Stats

You can override historical stats with custom projections:

```python
from simulation.player_stats import PlayerStats

# create custom stats
custom_stats = PlayerStats(
    player_name='Custom Player',
    surface='hard',
    matches_played=0,
    first_serve_pct=0.65,
    ace_pct=0.08,
    df_pct=0.03,
    first_serve_win_pct=0.75,
    second_serve_win_pct=0.56,
    return_first_win_pct=0.32,
    return_second_win_pct=0.50
)

# use in simulation
sim = MatchSimulator(custom_stats, alcaraz_stats)
result = sim.simulate_match()
```

## Limitations and Future Enhancements

**Current Limitations:**

- Does not account for player fatigue over long matches
- Does not model momentum or streakiness within a match
- No head-to-head historical adjustments
- No recent form or injury weighting
- Court speed and environmental conditions not yet modeled

**Recently Added:**

- ✅ Elo rating system for skill differentiation
- ✅ Shot-by-shot simulation with Match Charting Project data
- ✅ ML-powered match prediction with XGBoost
- ✅ Interactive Streamlit UI with visualizations
- ✅ Surface-specific player statistics
- ✅ Serve placement and return quality modeling

**Potential Future Enhancements:**

- Model time-varying probabilities (fatigue, momentum shifts)
- Add court speed ratings (fast/slow hard court)
- Weather and altitude adjustments
- Support for doubles matches
- Head-to-head learning and tactical adjustments
- Recent form weighting with time decay
- Player injury status integration

## Contributing

This is a demo/educational project. Feel free to fork and extend for your own use.

## License

This code is provided as-is for educational purposes. Jeff Sackmann's tennis data has its own license - please see his repositories for details.

## Credits

- Tennis data: [Jeff Sackmann](https://github.com/JeffSackmann)
- Simulation framework: Built for educational/research purposes

## Questions?

For issues or questions, please open an issue in this repository.
