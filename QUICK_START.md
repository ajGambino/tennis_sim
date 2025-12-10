# Quick Start Guide

Get up and running with tennis simulations in 60 seconds.

## Installation

```bash
# 1. navigate to project directory
cd tennis

# 2. install dependencies
pip install -r requirements.txt
```

## Run Your First Simulation

### Single Match
```bash
python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz"
```

You'll see:
- Player statistics (serve %, aces, etc.)
- Full match box score
- Winner and final score
- All match statistics

### Monte Carlo (1000 matches)
```bash
python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 1000
```

You'll see:
- Win probability for each player
- Most common final scores
- Average match statistics
- Results automatically saved to CSV

## Common Options

### Specify Surface
```bash
# clay court
python run_single.py --playerA "Novak Djokovic" --playerB "Rafael Nadal" --surface clay

# grass court
python run_single.py --playerA "Roger Federer" --playerB "Andy Murray" --surface grass
```

### Best of 5 Sets
```bash
python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --best_of 5
```

### Reproducible Simulations
```bash
# use same seed to get exact same match
python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --seed 42
```

### Large-Scale Simulations
```bash
# 10,000 simulations
python run_bulk.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 10000
```

## Using Real Data (Optional)

The framework works immediately with synthetic data. To use real historical data:

```bash
# download jeff sackmann's atp data
git clone https://github.com/JeffSackmann/tennis_atp.git

# copy csv files to data directory
cp tennis_atp/atp_matches_*.csv data/

# run simulation (will now use real data)
python run_single.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz"
```

## Programmatic Usage

```python
from simulation.data_loader import load_match_data
from simulation.player_stats import PlayerStatsCalculator
from simulation.match_engine import MatchSimulator

# load data
match_data = load_match_data()

# calculate player stats
stats_calc = PlayerStatsCalculator(match_data)
player_a = stats_calc.get_player_stats('Novak Djokovic')
player_b = stats_calc.get_player_stats('Carlos Alcaraz')

# simulate match
sim = MatchSimulator(player_a, player_b, best_of=3)
result = sim.simulate_match()

print(f"Winner: {result.winner}")
print(f"Score: {result.score}")
```

## Results Location

All simulation results are saved to:
```
results/
  PlayerA_vs_PlayerB_1000sims_TIMESTAMP.csv
```

Each row contains full match statistics for one simulation.

## Need Help?

See [README.md](README.md) for complete documentation.

## What's Next?

- Try different player matchups
- Compare performance on different surfaces
- Run large-scale Monte Carlo simulations
- Use the programmatic API for custom analysis
- See [example_api_usage.py](example_api_usage.py) for advanced usage
