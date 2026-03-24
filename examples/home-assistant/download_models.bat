@echo off
REM Windows batch script to download models
REM Run this before starting the app to avoid UI timeouts

echo.
echo ========================================
echo Model Downloader for Home Assistant
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .venv
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if huggingface_hub is installed
python -c "import huggingface_hub" 2>nul
if errorlevel 1 (
    echo.
    echo huggingface_hub is not installed
    echo Installing required packages...
    pip install huggingface_hub
)

REM Run the download script
echo.
echo Starting model downloads...
echo This may take 30-60 minutes depending on your internet connection
echo.
python download_models.py

REM Keep window open to see the output
pause
