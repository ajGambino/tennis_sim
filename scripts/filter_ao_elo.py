"""
Filter ELO ratings to only include players in the Australian Open 2025 bracket
"""

import json
import re
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.bracket import AO_2025_MENS_R1


def clean_player_name(name):
    """Remove seeding brackets and qualifiers from player names"""
    # Remove patterns like [1], [32], (WC), (Q), [PR], etc.
    cleaned = re.sub(r'\s*\[.*?\]|\s*\(.*?\)', '', name).strip()
    return cleaned


def main():
    # Extract all unique player names from the bracket
    bracket_players = set()
    for match_id, (player1, player2) in AO_2025_MENS_R1.items():
        bracket_players.add(clean_player_name(player1))
        bracket_players.add(clean_player_name(player2))

    print(f"Found {len(bracket_players)} unique players in AO 2025 bracket")

    # Load the full ELO ratings
    elo_path = Path(__file__).parent.parent / 'models' / 'elo_ratings_new.json'
    with open(elo_path, 'r') as f:
        all_elos = json.load(f)

    print(f"Loaded {len(all_elos)} total players from elo_ratings_new.json")

    # Create a normalized name lookup for fuzzy matching
    # Normalize: lowercase, remove hyphens, remove accents
    def normalize_name(name):
        import unicodedata
        # Remove accents
        name = ''.join(c for c in unicodedata.normalize('NFD', name)
                      if unicodedata.category(c) != 'Mn')
        # Lowercase and remove hyphens
        return name.lower().replace('-', ' ').strip()

    elo_lookup = {}
    for original_name in all_elos.keys():
        normalized = normalize_name(original_name)
        elo_lookup[normalized] = original_name

    # Filter to only AO bracket players
    filtered_elos = {}
    missing_players = []

    for player in bracket_players:
        # Clean the player name from bracket (this is what the simulator will use)
        clean_bracket_name = clean_player_name(player)

        # Try exact match first
        if player in all_elos:
            filtered_elos[clean_bracket_name] = all_elos[player]
        else:
            # Try normalized match
            normalized = normalize_name(player)
            if normalized in elo_lookup:
                original_name = elo_lookup[normalized]
                filtered_elos[clean_bracket_name] = all_elos[original_name]
                print(f"  Matched '{player}' -> '{original_name}' (stored as '{clean_bracket_name}')")
            else:
                missing_players.append(player)

    print(f"\nFiltered to {len(filtered_elos)} players with ELO ratings")

    if missing_players:
        print(f"\nWARNING: {len(missing_players)} players not found in ELO ratings (will default to 1500):")
        for player in sorted(missing_players):
            print(f"  - {player}")

    # Save filtered ratings
    output_path = Path(__file__).parent.parent / 'models' / 'elo_ratings_ao2025.json'
    with open(output_path, 'w') as f:
        json.dump(filtered_elos, f, indent=2)

    print(f"\nSaved filtered ELO ratings to: {output_path}")
    print(f"  {len(filtered_elos)} players included")


if __name__ == '__main__':
    main()
