# Windows PowerShell script to download models
# Run this before starting the app to avoid UI timeouts

Write-Host ""
Write-Host "========================================"
Write-Host "Model Downloader for Home Assistant"
Write-Host "========================================"
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Error: Virtual environment not found at .venv" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory"
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Check if huggingface_hub is installed
try {
    python -m pip show huggingface_hub | Out-Null
} catch {
    Write-Host ""
    Write-Host "huggingface_hub is not installed" -ForegroundColor Yellow
    Write-Host "Installing required packages..."
    python -m pip install huggingface_hub
}

# Run the download script
Write-Host ""
Write-Host "Starting model downloads..." -ForegroundColor Green
Write-Host "This may take 30-60 minutes depending on your internet connection"
Write-Host ""

python download_models.py

Write-Host ""
Write-Host "Download complete!" -ForegroundColor Green
Read-Host "Press Enter to exit"
