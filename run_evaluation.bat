@echo off
echo 🚀 Running Customer Propensity Model Evaluation...
echo.

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ⚠️ Virtual environment not found, using system Python
)

REM Install/upgrade required packages
echo 📦 Installing/upgrading required packages...
pip install -r requirements.txt

REM Run the evaluation script
echo.
echo 🔮 Starting model evaluation...
python model_evaluation.py

echo.
echo ✅ Evaluation complete! Check the generated PNG files.
pause
