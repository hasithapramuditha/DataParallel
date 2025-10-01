#!/bin/bash

# Data Partitioning Experiments - Simple Runner Script
# This script provides easy access to the main functionality

set -e  # Exit on any error

echo "🚀 Data Partitioning Experiments"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first:"
    echo "  python main.py --setup"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if activation was successful
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Run the main script with all arguments
python main.py "$@"
