#!/bin/bash

# Navigate to the backend directory
cd "$(dirname "$0")"

echo "=== AI4Bharat Backend Setup & Startup ==="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.12+ and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies from requirements.txt..."
echo "This might take a few minutes if torch/transformers are downloaded..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements."
    exit 1
fi

# Start FastAPI server
echo "Starting FastAPI server on http://127.0.0.1:8000..."
python main.py
