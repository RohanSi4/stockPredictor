#!/bin/bash

# to run:
#./setup_instructions.sh

# Instructions to the user
echo "This script will create a virtual environment, activate it, and install all required dependencies."

# Step 1: Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Step 2: Activate the virtual environment
echo "Activating the virtual environment..."
source .venv/bin/activate

# Step 3: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Instructions to the user
echo ""
echo "Setup is complete!"
echo "To activate the virtual environment in the future, run:"
echo "source .venv/bin/activate"
echo "To deactivate the virtual environment, run:"
echo "deactivate"
