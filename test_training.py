"""
quick test of training pipeline with small dataset
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

from simulation.data_loader import load_match_data
from simulation.elo_system import EloSystem
from simulation.feature_engineering import create_training_dataset

print("testing feature engineering...")

# load small amount of data
match_data = load_match_data(tour='atp', years=[2023])

# build elo
print("\nbuilding elo...")
elo_system = EloSystem(surface_specific=True)
if os.path.exists('models/elo_ratings_atp.json'):
    elo_system.load_ratings('models/elo_ratings_atp.json')
else:
    # use full data for elo
    full_data = load_match_data(tour='atp')
    elo_system.build_from_matches(full_data, verbose=False)

# create small training set
print("\ncreating training data...")
X, y = create_training_dataset(
    match_data,
    elo_ratings=elo_system.ratings,
    min_date=datetime(2023, 1, 1),
    max_date=datetime(2023, 3, 1)
)

print(f"\ntraining data shape: {X.shape}")
print(f"target distribution: {y.value_counts()}")
print(f"\nfeatures:\n{list(X.columns)}")
print("\ntest successful!")
