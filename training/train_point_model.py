"""
train xgboost model for point-level prediction

trains ml model to predict match outcomes, replacing static probability blending
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import pickle

# add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.data_loader import load_match_data
from simulation.elo_system import EloSystem
from simulation.feature_engineering import create_training_dataset


def train_model(X_train, y_train, X_val, y_val):
    """
    train xgboost model

    args:
        X_train: training features
        y_train: training labels
        X_val: validation features
        y_val: validation labels

    returns:
        trained model
    """
    print("\ntraining xgboost model...")

    # xgboost parameters with improved regularization
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 3,  # reduced from 4 to prevent overfitting
        'learning_rate': 0.1,
        'n_estimators': 200,  # increased to allow early stopping to find optimal
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,  # increased from 3 for more regularization
        'gamma': 0.1,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'tree_method': 'hist',
        'enable_categorical': False,
        'early_stopping_rounds': 20  # stop if no improvement for 20 rounds
    }

    model = xgb.XGBClassifier(**params)

    # train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    return model


def evaluate_model(model, X, y, dataset_name='Test'):
    """
    evaluate model performance

    args:
        model: trained model
        X: features
        y: true labels
        dataset_name: name for printing

    returns:
        dict with metrics
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'log_loss': log_loss(y, y_pred_proba),
        'auc_roc': roc_auc_score(y, y_pred_proba),
        'brier_score': brier_score_loss(y, y_pred_proba)
    }

    print(f"\n{dataset_name} Set Performance:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Log Loss:     {metrics['log_loss']:.4f}")
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  Brier Score:  {metrics['brier_score']:.4f}")

    return metrics


def plot_feature_importance(model, feature_names, top_n=20):
    """plot feature importance"""
    try:
        import matplotlib.pyplot as plt

        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.barh(range(top_n), importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        os.makedirs('training/plots', exist_ok=True)
        plt.savefig('training/plots/feature_importance.png')
        print("\nfeature importance plot saved to training/plots/feature_importance.png")
    except ImportError:
        print("\nmatplotlib not available, skipping feature importance plot")


def main():
    """main training pipeline"""
    print("="*80)
    print("TENNIS MATCH PREDICTION - ML MODEL TRAINING")
    print("="*80)

    # load data
    print("\nloading match data...")
    match_data = load_match_data(tour='atp')

    # convert tourney_date to datetime
    if 'tourney_date' in match_data.columns:
        match_data['tourney_date'] = pd.to_datetime(
            match_data['tourney_date'].astype(str),
            format='%Y%m%d',
            errors='coerce'
        )

    # create training/validation/test split
    print("\ncreating training/validation/test split...")

    train_cutoff = datetime(2022, 1, 1)
    val_cutoff = datetime(2023, 1, 1)
    test_cutoff = datetime(2024, 1, 1)

    # build elo ratings incrementally to prevent time leakage
    print("\nbuilding elo ratings for training set (2020-2021)...")
    train_data = match_data[match_data['tourney_date'] < train_cutoff]
    elo_system_train = EloSystem(surface_specific=True)
    elo_system_train.build_from_matches(train_data, verbose=False)

    print("building elo ratings for validation set (up to 2022)...")
    val_data = match_data[match_data['tourney_date'] < val_cutoff]
    elo_system_val = EloSystem(surface_specific=True)
    elo_system_val.build_from_matches(val_data, verbose=False)

    print("building elo ratings for test set (up to 2023)...")
    test_data = match_data[match_data['tourney_date'] < test_cutoff]
    elo_system_test = EloSystem(surface_specific=True)
    elo_system_test.build_from_matches(test_data, verbose=False)

    # save final elo ratings for production use
    os.makedirs('models', exist_ok=True)
    elo_file = 'models/elo_ratings_atp.json'
    elo_system_test.save_ratings(elo_file)

    # training set: 2020-2021
    X_train, y_train = create_training_dataset(
        match_data,
        elo_ratings=elo_system_train.ratings,
        min_date=datetime(2020, 1, 1),
        max_date=train_cutoff
    )

    # validation set: 2022
    X_val, y_val = create_training_dataset(
        match_data,
        elo_ratings=elo_system_val.ratings,
        min_date=train_cutoff,
        max_date=val_cutoff
    )

    # test set: 2023
    X_test, y_test = create_training_dataset(
        match_data,
        elo_ratings=elo_system_test.ratings,
        min_date=val_cutoff,
        max_date=test_cutoff
    )

    print(f"\nDataset sizes:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")

    # train model
    base_model = train_model(X_train, y_train, X_val, y_val)

    # apply platt scaling for better calibration
    print("\napplying platt scaling calibration...")
    model = CalibratedClassifierCV(
        base_model,
        method='sigmoid',  # platt scaling
        cv='prefit'  # use pre-fitted model
    )
    model.fit(X_val, y_val)  # calibrate on validation set

    # evaluate
    print("\nevaluating base model (before calibration)...")
    train_metrics = evaluate_model(base_model, X_train, y_train, 'Training')
    val_metrics = evaluate_model(base_model, X_val, y_val, 'Validation')
    test_metrics_base = evaluate_model(base_model, X_test, y_test, 'Test (Base)')

    print("\nevaluating calibrated model...")
    test_metrics = evaluate_model(model, X_test, y_test, 'Test (Calibrated)')

    # feature importance (use base model, not calibrated wrapper)
    plot_feature_importance(base_model, X_train.columns)

    # save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/match_predictor_xgb.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,  # calibrated model
            'base_model': base_model,  # original xgboost model
            'features': list(X_train.columns),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'test_metrics_base': test_metrics_base,  # metrics before calibration
            'trained_date': datetime.now().isoformat()
        }, f)

    print(f"\nmodel saved to {model_path}")

    # save feature names
    feature_list_path = 'models/feature_names.txt'
    with open(feature_list_path, 'w') as f:
        for feat in X_train.columns:
            f.write(f"{feat}\n")
    print(f"feature names saved to {feature_list_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
