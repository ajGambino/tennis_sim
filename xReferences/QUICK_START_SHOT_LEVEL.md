# Quick Start: Shot-Level Simulation

**5-minute guide to using the new shot-by-shot tennis simulation**

---

## Option 1: Use Built-In Defaults (No Setup Required)

```bash
# Compare point-level vs shot-level simulation
python compare_shot_vs_point.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --n 100
```

**That's it!** The system uses built-in default serve/rally patterns.

---

## Option 2: Use Match Charting Project Data (Full Pipeline)

### Step 1: Download Charting Data (~2 minutes)

**Windows:**
```bash
download_charting_data.bat
```

**Mac/Linux:**
```bash
chmod +x download_charting_data.sh
./download_charting_data.sh
```

**Or manually:**
```bash
git clone https://github.com/JeffSackmann/tennis_MatchChartingProject
mkdir -p data/charting
cp tennis_MatchChartingProject/charting-*.csv data/charting/
```

### Step 2: Train Models (~1 minute)

```bash
# Train serve placement model
python training/train_serve_model.py

# Train rally model
python training/train_rally_model.py
```

**Output:**
- `models/serve_patterns.pkl` - Player-specific serve patterns
- `models/rally_patterns.pkl` - Rally characteristics

### Step 3: Validate (~30 seconds)

```bash
python analysis/validate_shot_simulation.py --validate all --num_sims 500
```

### Step 4: Run Simulations

```bash
# Compare modes
python compare_shot_vs_point.py --playerA "Novak Djokovic" --playerB "Rafael Nadal" --n 1000

# Single match with shot details
python run_single.py --playerA "Roger Federer" --playerB "Andy Murray" --use_shot_sim
```

---

## Python API Usage

### Enable Shot-Level Simulation

```python
from simulation.point_engine import PointSimulator
from simulation.data_loader import DataLoader
from simulation.player_stats import PlayerStatsCalculator

# Load data and compute stats
data_loader = DataLoader()
matches = data_loader.load_match_data(years=[2023])
stats_calc = PlayerStatsCalculator(matches)

stats_sinner = stats_calc.get_player_stats("Jannik Sinner", surface='hard')
stats_alcaraz = stats_calc.get_player_stats("Carlos Alcaraz", surface='hard')

# Create shot-level simulator
point_sim = PointSimulator(use_shot_simulation=True)

# Simulate point
result = point_sim.simulate_point(
    server_stats=stats_sinner,
    returner_stats=stats_alcaraz,
    server_name="Jannik Sinner",
    returner_name="Carlos Alcaraz",
    surface='hard'
)

# Access shot-level details
print(f"Rally length: {result.rally_length} shots")
print(f"Serve placement: {result.serve_placement}")
print(f"Point-ending shot: {result.point_ending_shot_type}")

if result.rally:
    print(f"\nShot sequence:")
    for shot in result.rally.shots:
        print(f"  Shot {shot.shot_number}: {shot.shot_type.value} {shot.direction.value} → {shot.outcome.value}")
```

### Use Trained Models

```python
import pickle
from simulation.serve_model import ServeModel
from simulation.return_model import ReturnModel
from simulation.rally_model import RallyModel

# Load trained patterns
with open('models/serve_patterns.pkl', 'rb') as f:
    serve_data = pickle.load(f)

# Create models with trained patterns
serve_model = ServeModel(
    serve_patterns=serve_data['serve_patterns'],
    default_pattern=serve_data['default_pattern']
)
return_model = ReturnModel()
rally_model = RallyModel()

# Create simulator with trained models
point_sim = PointSimulator(
    use_shot_simulation=True,
    serve_model=serve_model,
    return_model=return_model,
    rally_model=rally_model
)
```

---

## What You Get

### Point-Level (Legacy)
```
Server won: True
Was ace: False
Rally length: 0
Serve placement: None
```

### Shot-Level (New)
```
Server won: True
Was ace: False
Rally length: 6 shots
Serve placement: T (down the T)
Point-ending shot: FOREHAND winner

Shot sequence:
  1: SERVE → Wide
  2: BACKHAND return → Cross-court (in play)
  3: FOREHAND → Down-the-line (in play)
  4: BACKHAND → Cross-court (in play)
  5: FOREHAND (inside baseline) → Down-the-line (in play)
  6: FOREHAND → Cross-court (WINNER)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `simulation/shot.py` | Data structures (Shot, Rally, ServeOutcome) |
| `simulation/serve_model.py` | Serve placement model |
| `simulation/return_model.py` | Return quality model |
| `simulation/rally_model.py` | Shot-by-shot rally simulation |
| `simulation/point_engine.py` | Updated point simulator (supports both modes) |
| `training/train_serve_model.py` | Extract serve patterns from MCP data |
| `analysis/validate_shot_simulation.py` | Validation tools |
| `compare_shot_vs_point.py` | Compare both simulation modes |

---

## CLI Commands

```bash
# Compare simulation modes
python compare_shot_vs_point.py --playerA "Player A" --playerB "Player B" --n 100

# Validate serve placements
python analysis/validate_shot_simulation.py --validate serve --player "Novak Djokovic"

# Validate rally lengths
python analysis/validate_shot_simulation.py --validate rally --num_sims 500

# Run all validations
python analysis/validate_shot_simulation.py --validate all

# Train serve model
python training/train_serve_model.py --tour atp --min_points 50

# Train rally model
python training/train_rally_model.py --tour atp --min_rallies 30
```

---

## Performance

| Simulation Mode | Time per Match | Overhead |
|----------------|----------------|----------|
| Point-Level    | ~5-10 ms       | Baseline |
| Shot-Level     | ~15-30 ms      | 2-3x     |

**Why slower?** Shot-level simulates 4-5 shots per point vs 1 outcome in point-level.

**Optimization coming in Phase 3B:** Target <10 ms per match.

---

## Next Steps

- **Read full docs:** [PHASE_3A_SHOT_LEVEL.md](PHASE_3A_SHOT_LEVEL.md)
- **Phase 3B goals:** Momentum, fatigue, advanced tactics, player styles
- **Experiment:** Try different players, surfaces, tournament scenarios

---

**Ready to simulate shot-by-shot!** 🎾
