#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create virtual environment in current directory (./venv)
python3 -m venv VQLS_env

# Activate the environment
# For macOS/Linux
source VQLS_env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
