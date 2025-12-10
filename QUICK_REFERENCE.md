# Tennis Simulation - Quick Reference

## Common Commands

### Single Match Simulation

```bash
# Basic simulation
python run_single.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz"

# With Elo adjustment (RECOMMENDED)
python run_single.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo

# Specify surface
python run_single.py --playerA "Rafael Nadal" --playerB "Novak Djokovic" --use_elo --surface clay

# Best of 5 sets
python run_single.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo --best_of 5

# Reproducible (with seed)
python run_single.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo --seed 12345
```

### Bulk Monte Carlo Simulation

```bash
# 1000 simulations (default)
python run_bulk.py --playerA "Jannik Sinner" --playerB "Gabriel Diallo" --use_elo

# 5000 simulations
python run_bulk.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo --n 5000

# With surface
python run_bulk.py --playerA "Rafael Nadal" --playerB "Novak Djokovic" --use_elo --surface clay --n 10000

# Custom output filename
python run_bulk.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo --n 5000 --output my_results.csv

# Don't save CSV (just print summary)
python run_bulk.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo --n 1000 --no_save
```

### WTA (Women's Tennis)

```bash
# Single match
python run_single.py --playerA "Iga Swiatek" --playerB "Aryna Sabalenka" --use_elo --tour wta

# Bulk simulation
python run_bulk.py --playerA "Iga Swiatek" --playerB "Coco Gauff" --use_elo --tour wta --n 5000
```

### Testing & Comparison

```bash
# Test Elo system
python test_elo.py

# Compare baseline vs Elo
python compare_elo.py --playerA "Jannik Sinner" --playerB "Gabriel Diallo" --n 1000

# Different matchup
python compare_elo.py --playerA "Novak Djokovic" --playerB "Carlos Alcaraz" --n 2000 --surface hard
```

## Flags Explained

| Flag | Description | Example |
|------|-------------|---------|
| `--playerA` | First player name | `"Jannik Sinner"` |
| `--playerB` | Second player name | `"Carlos Alcaraz"` |
| `--use_elo` | Enable Elo adjustments (RECOMMENDED) | Just add flag |
| `--surface` | Court surface | `hard`, `clay`, `grass` |
| `--best_of` | Match format | `3` (default) or `5` |
| `--n` | Number of simulations (bulk only) | `1000`, `5000`, `10000` |
| `--seed` | Random seed for reproducibility | `12345` |
| `--tour` | Tour type | `atp` (default) or `wta` |
| `--output` | Custom output filename (bulk) | `my_results.csv` |
| `--no_save` | Don't save CSV (bulk) | Just add flag |
| `--elo_file` | Custom Elo ratings file | `models/my_elo.json` |

## When to Use Elo

**Always use `--use_elo` unless:**
- You're specifically testing baseline performance
- You're comparing baseline vs Elo results

**Why?** Elo adjustments make predictions much more realistic for matchups between players of different skill levels.

## Output Files

### Single Match
- Prints full match statistics to console
- No file saved

### Bulk Simulation
- Saves CSV to `results/` directory
- Filename format: `PlayerA_vs_PlayerB_surface_elo_Nsims_timestamp.csv`
- Example: `Jannik_Sinner_vs_Carlos_Alcaraz_hard_elo_5000sims_20241209_193000.csv`

## Elo Ratings Cache

- First run builds Elo ratings (~3 seconds for ATP)
- Saved to `models/elo_ratings_atp.json` or `models/elo_ratings_wta.json`
- Subsequent runs load instantly (<0.1 seconds)
- Delete cache file to rebuild from scratch

## Example Workflows

### Quick Single Match
```bash
python run_single.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo
```

### Tournament Final Simulation
```bash
python run_bulk.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo --surface hard --best_of 5 --n 10000
```

### Clay Court Specialist Analysis
```bash
python run_bulk.py --playerA "Rafael Nadal" --playerB "Novak Djokovic" --use_elo --surface clay --n 5000
```

### Head-to-Head Comparison
```bash
# First run baseline
python run_bulk.py --playerA "Player A" --playerB "Player B" --n 2000 --output baseline.csv

# Then run with Elo
python run_bulk.py --playerA "Player A" --playerB "Player B" --use_elo --n 2000 --output elo.csv

# Or use comparison tool
python compare_elo.py --playerA "Player A" --playerB "Player B" --n 2000
```

## Troubleshooting

### Player Not Found
- Check spelling (case-sensitive)
- Player must exist in historical data (2015-2024)
- Try without surface filter first

### Low Matches Warning
```
warning: Player X has only 5 matches on clay, using fallback stats
```
- Normal for players with limited data on specific surface
- Elo still helps, but predictions less reliable

### Build Elo Taking Long
- First run builds from 26k+ matches (~3 seconds)
- Subsequent runs use cache (instant)
- Delete `models/elo_ratings_*.json` to rebuild

## Pro Tips

1. **Always use Elo** for realistic results
2. **Use surface filters** for more accurate predictions (e.g., Nadal on clay)
3. **Run 5000+ simulations** for stable win probabilities
4. **Use seeds** to reproduce exact match results
5. **Compare baseline vs Elo** to see the improvement

## Performance

| Operation | Time |
|-----------|------|
| Build Elo (26k matches) | ~3 seconds |
| Load cached Elo | <0.1 seconds |
| Single match simulation | <0.1 seconds |
| 1000 simulations | ~3 seconds |
| 5000 simulations | ~15 seconds |
| 10000 simulations | ~30 seconds |

## What's Next?

**Phase 2** (in development):
- XGBoost ML models for point prediction
- Feature engineering from match history
- Fatigue modeling
- Point-by-point data integration
- Further performance optimization

---

**Quick start:** `python run_bulk.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz" --use_elo --n 5000`
