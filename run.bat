@echo off
REM Data Partitioning Experiments - Simple Runner Script (Windows)
REM This script provides easy access to the main functionality

echo 🚀 Data Partitioning Experiments
echo ================================

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run setup first:
    echo   python main.py --setup
    pause
    exit /b 1
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if "%VIRTUAL_ENV%"=="" (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated: %VIRTUAL_ENV%

REM Run the main script with all arguments
python main.py %*
