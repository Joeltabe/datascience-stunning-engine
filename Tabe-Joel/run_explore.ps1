# Quick Data Explorer - No typing long paths!
# Double-click this file in Windows Explorer or run: .\run_explore.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SHOPCAM DATA EXPLORER" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$python = "C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe"

# Check if Python exists
if (-not (Test-Path $python)) {
    Write-Host "ERROR: Python not found at $python" -ForegroundColor Red
    Write-Host "Please update the path in this script" -ForegroundColor Yellow
    pause
    exit
}

# Run the exploration script
& $python explore_shopcam_data.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Exploration Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

pause
