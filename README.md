# Tennis Match Simulation Framework

A comprehensive point-level Monte Carlo tennis simulation engine built with Python, using Jeff Sackmann's public tennis datasets.

## Features

- **Point-level simulation**: Every point simulated using probabilistic serve/return models
- **Historical data integration**: Uses real ATP/WTA match statistics from Jeff Sackmann's datasets
- **Full tennis rules**: Advantage scoring, tiebreaks, set/match logic
- **Monte Carlo analysis**: Run thousands of simulations to compute win probabilities and distributions
- **Comprehensive statistics**: Aces, double faults, break points, first serve %, and more
- **Surface-specific stats**: Aggregate player performance by court surface (hard, clay, grass)
- **Reproducible results**: All simulations use random seeds for exact reproducibility

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

### Single Match Simulation

Simulate one match and see detailed statistics:

```bash
python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz"
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
python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --surface clay

# best of 5 sets
python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --best_of 5

# reproducible simulation with seed
python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --seed 12345
```

### Bulk Monte Carlo Simulation

Run thousands of simulations to compute win probabilities:

```bash
python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000
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
python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000 --surface hard

# best of 5 sets
python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000 --best_of 5

# custom output filename
python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 5000 --output my_results.csv

# don't save csv (just print summary)
python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 1000 --no_save
```

## Project Structure

```
tennis/
├── data/                           # place csv files here
│   └── (atp_matches_YYYY.csv files)
├── results/                        # simulation output csvs saved here
├── simulation/                     # core simulation package
│   ├── __init__.py                # package exports
│   ├── data_loader.py             # load and preprocess sackmann csvs
│   ├── player_stats.py            # compute player probability parameters
│   ├── point_engine.py            # simulate individual points
│   ├── match_engine.py            # game/set/match logic with full rules
│   └── simulator.py               # bulk simulation runner
├── run_single.py                  # cli for single match simulation
├── run_bulk.py                    # cli for monte carlo simulations
├── requirements.txt               # python dependencies
└── README.md                      # this file
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
- **Point-by-point Data**: https://github.com/JeffSackmann/tennis_MatchChartingProject (optional)

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
from simulation.data_loader import load_match_data
from simulation.player_stats import PlayerStatsCalculator
from simulation.match_engine import MatchSimulator
from simulation.simulator import run_simulations, analyze_results

# load data
match_data = load_match_data(data_dir='data', tour='atp')

# calculate player stats
stats_calc = PlayerStatsCalculator(match_data)
djokovic_stats = stats_calc.get_player_stats('Novak Djokovic', surface='hard')
alcaraz_stats = stats_calc.get_player_stats('Carlos Alcaraz', surface='hard')

# single simulation
sim = MatchSimulator(djokovic_stats, alcaraz_stats, best_of=3, seed=42)
result = sim.simulate_match()
print(f"Winner: {result.winner}")
print(f"Score: {result.score}")

# bulk simulations
results_df = run_simulations(djokovic_stats, alcaraz_stats, n=10000)
summary = analyze_results(results_df, 'Novak Djokovic', 'Carlos Alcaraz')
print(f"Win probability: {summary['a_win_pct']:.2%}")
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
- Does not model momentum or streakiness
- Return stats derived from opponent serve stats (not direct return measurements)
- No head-to-head adjustment
- No recent form weighting

**Potential Enhancements:**
- Incorporate point-by-point charting data for more granular modeling
- Add player ranking/Elo adjustments
- Model time-varying probabilities (fatigue, momentum)
- Add court speed and conditions
- Support for doubles
- Head-to-head historical adjustments
- Injury/form adjustments

## Contributing

This is a demo/educational project. Feel free to fork and extend for your own use.

## License

This code is provided as-is for educational purposes. Jeff Sackmann's tennis data has its own license - please see his repositories for details.

## Credits

- Tennis data: [Jeff Sackmann](https://github.com/JeffSackmann)
- Simulation framework: Built for educational/research purposes

## Questions?

For issues or questions, please open an issue in this repository.
