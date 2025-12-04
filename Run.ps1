# Exit on error
$ErrorActionPreference = "Stop"

# Create virtual environment
python -m venv VQLS_env

# Activate environment
.\VQLS_env\Scripts\Activate.ps1

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
