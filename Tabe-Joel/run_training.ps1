# Quick Training Script - No typing long paths!
# Double-click this file in Windows Explorer or run: .\run_training.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SHOPCAM ML MODEL TRAINER" -ForegroundColor Cyan
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

# Run the training script
& $python train_shopcam_sklearn.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

pause
