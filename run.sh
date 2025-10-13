#!/bin/bash

# Data Partitioning Experiments - Simple Runner Script
# This script provides easy access to the main functionality

set -e  # Exit on any error

echo "🚀 Data Partitioning Experiments"
echo "================================"

# Check if virtual environment exists (optional)
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
    
    # Check if activation was successful
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi
    
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "ℹ️  No virtual environment found, using system Python"
fi

# Run the main script with all arguments
python main.py "$@"
