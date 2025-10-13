@echo off
REM Data Partitioning Experiments - Simple Runner Script (Windows)
REM This script provides easy access to the main functionality

echo 🚀 Data Partitioning Experiments
echo ================================

REM Check if virtual environment exists (optional)
if exist "venv" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
    
    REM Check if activation was successful
    if "%VIRTUAL_ENV%"=="" (
        echo ❌ Failed to activate virtual environment
        pause
        exit /b 1
    )
    
    echo ✅ Virtual environment activated: %VIRTUAL_ENV%
) else (
    echo ℹ️  No virtual environment found, using system Python
)

REM Run the main script with all arguments
python main.py %*
