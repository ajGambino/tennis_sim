#!/bin/bash
# Download Match Charting Project Data
# This script downloads shot-by-shot tennis data for shot-level simulation

echo "========================================================================"
echo "MATCH CHARTING PROJECT DATA DOWNLOAD"
echo "========================================================================"
echo ""
echo "This will download the Match Charting Project repository containing"
echo "shot-by-shot data for ~17,000 ATP and ~5,000 WTA matches."
echo ""
echo "Repository: https://github.com/JeffSackmann/tennis_MatchChartingProject"
echo "Size: ~50-100 MB"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed"
    echo ""
    echo "Please install Git:"
    echo "  - Ubuntu/Debian: sudo apt-get install git"
    echo "  - macOS: brew install git"
    echo ""
    echo "Alternatively, download manually:"
    echo "1. Visit https://github.com/JeffSackmann/tennis_MatchChartingProject"
    echo "2. Click 'Code' -> 'Download ZIP'"
    echo "3. Extract to data/charting/"
    echo ""
    exit 1
fi

# Create data/charting directory
if [ ! -d "data/charting" ]; then
    echo "Creating data/charting directory..."
    mkdir -p data/charting
fi

# Check if already downloaded
if [ -f "data/charting/charting-m-points.csv" ]; then
    echo ""
    echo "WARNING: Charting data appears to already exist in data/charting/"
    echo ""
    read -p "Do you want to re-download (this will overwrite existing files)? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Download cancelled."
        exit 0
    fi
fi

echo ""
echo "Downloading Match Charting Project..."
echo ""

# Clone repository to temp location
if [ -d "temp_charting" ]; then
    rm -rf temp_charting
fi

git clone --depth 1 https://github.com/JeffSackmann/tennis_MatchChartingProject.git temp_charting

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to clone repository"
    echo ""
    exit 1
fi

# Copy CSV files to data/charting
echo ""
echo "Copying charting CSV files..."
cp temp_charting/charting-m-points.csv data/charting/
cp temp_charting/charting-w-points.csv data/charting/
cp temp_charting/charting-m-matches.csv data/charting/
cp temp_charting/charting-w-matches.csv data/charting/

# Clean up temp directory
echo "Cleaning up..."
rm -rf temp_charting

echo ""
echo "========================================================================"
echo "DOWNLOAD COMPLETE"
echo "========================================================================"
echo ""
echo "Match Charting Project data has been downloaded to: data/charting/"
echo ""
echo "Files:"
echo "  - charting-m-points.csv  (ATP point-by-point data)"
echo "  - charting-w-points.csv  (WTA point-by-point data)"
echo "  - charting-m-matches.csv (ATP match metadata)"
echo "  - charting-w-matches.csv (WTA match metadata)"
echo ""
echo "Next steps:"
echo "  1. Train serve model:  python training/train_serve_model.py"
echo "  2. Train rally model:  python training/train_rally_model.py"
echo "  3. Validate:           python analysis/validate_shot_simulation.py"
echo ""
echo "========================================================================"
