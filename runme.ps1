# SafeEdge PPE Vision - Project Launcher
# Run: powershell -ExecutionPolicy Bypass -File runme.ps1

Set-Location $PSScriptRoot

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = "python"
}
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   SafeEdge PPE Vision - Project Launcher  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Live Detector  (opens video in continuous loop, like CCTV)" -ForegroundColor Green
Write-Host "  2. Web Dashboard  (Streamlit browser app)" -ForegroundColor Yellow
Write-Host "  3. Single Image   (process one image file)" -ForegroundColor Magenta
Write-Host ""

$choice = Read-Host "Enter choice (1 / 2 / 3)"

switch ($choice) {

    "1" {
        Write-Host ""
        Write-Host "Starting Live Detector (continuous loop active)..." -ForegroundColor Green
        Write-Host "Tip: Press 'Q' or 'ESC' on your keyboard anytime to exit!" -ForegroundColor Yellow
        Write-Host "Press ENTER to auto-pick first video in data\ folder"
        $vid = Read-Host "Video path (or press ENTER)"
        if ($vid -eq "") {
            & $pythonPath live_detect.py
        } else {
            & $pythonPath live_detect.py --video $vid
        }
    }

    "2" {
        Write-Host ""
        Write-Host "Starting Web Dashboard..." -ForegroundColor Yellow
        Write-Host "Open browser at: http://localhost:8501" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop the server"
        & $pythonPath -m streamlit run app.py --browser.gatherUsageStats false
    }

    "3" {
        Write-Host ""
        Write-Host "Single Image Mode" -ForegroundColor Magenta
        $img = Read-Host "Image path (press ENTER for sample)"
        if ($img -eq "") { $img = "data\sample_workers.jpg" }
        & $pythonPath main.py --image $img
        Write-Host "Done! Check the output\ folder." -ForegroundColor Green
    }

    default {
        Write-Host "Invalid choice. Enter 1, 2 or 3." -ForegroundColor Red
    }
}