@echo off
REM Download Match Charting Project Data
REM This script downloads shot-by-shot tennis data for shot-level simulation

echo ========================================================================
echo MATCH CHARTING PROJECT DATA DOWNLOAD
echo ========================================================================
echo.
echo This will download the Match Charting Project repository containing
echo shot-by-shot data for ~17,000 ATP and ~5,000 WTA matches.
echo.
echo Repository: https://github.com/JeffSackmann/tennis_MatchChartingProject
echo Size: ~50-100 MB
echo.

REM Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo.
    echo Please install Git from: https://git-scm.com/download/win
    echo.
    echo Alternatively, download manually:
    echo 1. Visit https://github.com/JeffSackmann/tennis_MatchChartingProject
    echo 2. Click "Code" -^> "Download ZIP"
    echo 3. Extract to data/charting/
    echo.
    pause
    exit /b 1
)

REM Create data/charting directory
if not exist "data\charting" (
    echo Creating data\charting directory...
    mkdir data\charting
)

REM Check if already downloaded
if exist "data\charting\charting-m-points.csv" (
    echo.
    echo WARNING: Charting data appears to already exist in data\charting\
    echo.
    choice /C YN /M "Do you want to re-download (this will overwrite existing files)"
    if errorlevel 2 (
        echo.
        echo Download cancelled.
        pause
        exit /b 0
    )
)

echo.
echo Downloading Match Charting Project...
echo.

REM Clone repository to temp location
if exist "temp_charting" (
    rmdir /s /q temp_charting
)

git clone --depth 1 https://github.com/JeffSackmann/tennis_MatchChartingProject.git temp_charting

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to clone repository
    echo.
    pause
    exit /b 1
)

REM Copy CSV files to data/charting
echo.
echo Copying charting CSV files...
copy temp_charting\charting-m-points.csv data\charting\ >nul
copy temp_charting\charting-w-points.csv data\charting\ >nul
copy temp_charting\charting-m-matches.csv data\charting\ >nul
copy temp_charting\charting-w-matches.csv data\charting\ >nul

REM Clean up temp directory
echo Cleaning up...
rmdir /s /q temp_charting

echo.
echo ========================================================================
echo DOWNLOAD COMPLETE
echo ========================================================================
echo.
echo Match Charting Project data has been downloaded to: data\charting\
echo.
echo Files:
echo   - charting-m-points.csv  (ATP point-by-point data)
echo   - charting-w-points.csv  (WTA point-by-point data)
echo   - charting-m-matches.csv (ATP match metadata)
echo   - charting-w-matches.csv (WTA match metadata)
echo.
echo Next steps:
echo   1. Train serve model:  python training\train_serve_model.py
echo   2. Train rally model:  python training\train_rally_model.py
echo   3. Validate:           python analysis\validate_shot_simulation.py
echo.
echo ========================================================================

pause
