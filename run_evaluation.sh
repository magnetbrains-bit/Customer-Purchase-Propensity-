#!/bin/bash

echo "🚀 Running Customer Propensity Model Evaluation..."
echo

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️ Virtual environment not found, using system Python"
fi

# Install/upgrade required packages
echo "📦 Installing/upgrading required packages..."
pip install -r requirements.txt

# Run the evaluation script
echo
echo "🔮 Starting model evaluation..."
python model_evaluation.py

echo
echo "✅ Evaluation complete! Check the generated PNG files."
