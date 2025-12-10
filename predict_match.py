"""
ml-based match prediction cli

predict match outcomes using trained ml model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from simulation.ml_predictor import predict_match_from_names


def main():
    """run ml prediction from command line"""

    parser = argparse.ArgumentParser(
        description='predict tennis match outcome using ml model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python predict_match.py --playerA "Jannik Sinner" --playerB "Carlos Alcaraz"
  python predict_match.py --playerA "Jannik Sinner" --playerB "Gabriel Diallo" --surface hard
  python predict_match.py --playerA "Iga Swiatek" --playerB "Aryna Sabalenka" --tour wta
        """
    )

    parser.add_argument('--playerA', type=str, required=True,
                       help='name of player a')
    parser.add_argument('--playerB', type=str, required=True,
                       help='name of player b')
    parser.add_argument('--surface', type=str, default='hard',
                       choices=['hard', 'clay', 'grass'],
                       help='surface type (default: hard)')
    parser.add_argument('--tour', type=str, default='atp',
                       choices=['atp', 'wta'],
                       help='tour type (default: atp)')
    parser.add_argument('--model', type=str, default='models/match_predictor_xgb.pkl',
                       help='path to ml model (default: models/match_predictor_xgb.pkl)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"ML MATCH PREDICTION")
    print(f"{'='*70}\n")

    print(f"loading data and model...")

    # get prediction
    result = predict_match_from_names(
        args.playerA,
        args.playerB,
        args.surface,
        args.tour,
        args.model
    )

    # print results
    print(f"\n{'-'*70}")
    print(f"MATCH: {result['player_a']} vs {result['player_b']}")
    print(f"Surface: {result['surface']}")
    print(f"{'-'*70}")

    print(f"\nElo Ratings:")
    print(f"  {result['player_a']}: {result['elo_a']:.0f}")
    print(f"  {result['player_b']}: {result['elo_b']:.0f}")
    print(f"  Differential: {result['elo_a'] - result['elo_b']:+.0f}")

    print(f"\n{'-'*70}")
    print(f"ML MODEL PREDICTION")
    print(f"{'-'*70}")
    print(f"  {result['player_a']} Win Probability: {result['player_a_win_prob']:.1%}")
    print(f"  {result['player_b']} Win Probability: {result['player_b_win_prob']:.1%}")
    print(f"  Confidence: {result['ml_confidence']:.1%}")

    print(f"\n{'-'*70}")
    print(f"COMPARISON")
    print(f"{'-'*70}")
    print(f"  ML Prediction:  {result['ml_win_prob_a']:.1%} - {1-result['ml_win_prob_a']:.1%}")
    print(f"  Elo Prediction: {result['elo_win_prob_a']:.1%} - {1-result['elo_win_prob_a']:.1%}")
    print(f"  Difference:     {result['difference']:+.1%}")

    # interpretation
    print(f"\n{'-'*70}")
    print(f"INTERPRETATION")
    print(f"{'-'*70}")

    if abs(result['difference']) < 0.05:
        print(f"  ML and Elo predictions are very close - strong agreement")
    elif result['difference'] > 0:
        print(f"  ML is more optimistic about {result['player_a']} than Elo")
    else:
        print(f"  ML is less optimistic about {result['player_a']} than Elo")

    if result['ml_confidence'] > 0.6:
        print(f"  ML model is highly confident in this prediction")
    elif result['ml_confidence'] > 0.3:
        print(f"  ML model has moderate confidence")
    else:
        print(f"  ML model sees this as a close matchup")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
