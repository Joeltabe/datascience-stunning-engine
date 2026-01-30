param(
    [string]$Branch = 'main',
    [switch]$DryRun
)

# Map of files to exact commit messages (messages taken verbatim from the provided list)
$items = @(
    @{ path = 'app.py'; message = 'Add main application script implementing core entry point and primary workflows for data processing and demos.' },
    @{ path = 'areas.csv'; message = 'Add geographic `areas.csv` dataset used by analysis notebooks (area mappings / region metadata).' },
    @{ path = 'BigData_DataBase_Lab.ipynb'; message = 'Add Big Data / Database lab notebook demonstrating queries, exercises, and example datasets.' },
    @{ path = 'DailyDelhiClimateTest.csv'; message = 'Add test split for Daily Delhi Climate dataset used for model evaluation.' },
    @{ path = 'DailyDelhiClimateTrain.csv'; message = 'Add training split for Daily Delhi Climate dataset used for time-series training.' },
    @{ path = 'full_data.csv'; message = 'Add aggregated `full_data.csv` used across notebooks as the consolidated experiment dataset.' },
    @{ path = 'homeprices.csv'; message = 'Add sample housing prices dataset for regression demos and tutorials.' },
    @{ path = 'insurance_data.csv'; message = 'Add insurance dataset used for EDA and modeling examples.' },
    @{ path = 'Lab2Landmark.ipynb'; message = 'Add Lab 2 notebook for landmark exercise including instructions and example runs.' },
    @{ path = 'LabWordcountLandmark (1).ipynb'; message = 'Add duplicate copy of `LabWordcountLandmark` notebook (kept as a separate backup/iteration).' },
    @{ path = 'LabWordcountLandmark.ipynb'; message = 'Add primary wordcount landmark notebook showing Spark/MapReduce wordcount examples and analysis.' },
    @{ path = 'Learning Pandas.ipynb'; message = 'Add pandas learning notebook with examples, exercises, and data-wrangling patterns.' },
    @{ path = 'LSTM_Daily_Climate_Forecasting.ipynb'; message = 'Add LSTM forecasting notebook implementing model architecture and forecasting pipeline for daily climate.' },
    @{ path = 'LSTM_Daily_Climate_Train_Test.ipynb'; message = 'Add training & testing notebook for LSTM daily-climate experiments, including metrics and plots.' },
    @{ path = 'Machine_Learning_learn.ipynb'; message = 'Add general machine learning experiments notebook with assorted models and demo workflows.' },
    @{ path = 'NCHS_-_Leading_Causes_of_Death__United_States.csv'; message = 'Add NCHS leading-causes-of-death dataset for public-health analysis examples.' },
    @{ path = 'Pandas_learn.ipynb'; message = 'Add supplemental pandas practice notebook with hands-on examples.' },
    @{ path = 'README.md'; message = 'Add project README describing repository purpose, dataset list, and basic usage instructions.' },
    @{ path = 'requirements.txt'; message = 'Add `requirements.txt` listing Python dependencies required to run notebooks and scripts.' },
    @{ path = 'salaries.csv'; message = 'Add sample salaries dataset used for regression/classification tutorials.' },
    @{ path = 'sample_corpus.txt'; message = 'Add sample text corpus for wordcount and text-processing labs.' },
    @{ path = 'setup_emr.py'; message = 'Add EMR setup script for provisioning/configuring AWS EMR clusters used in Spark exercises.' },
    @{ path = 'Spark_Text_Exercises.ipynb'; message = 'Add Spark text-processing exercises notebook with step-by-step examples.' },
    @{ path = 'spark_text_lab.py'; message = 'Add Spark job script implementing text-processing pipeline used in labs.' },
    @{ path = 'test1.ipynb'; message = 'Add sandbox notebook for quick experiments and testing snippets.' },
    @{ path = 'time-series.ipynb'; message = 'Add time-series analysis notebook with models and forecasting demonstrations.' },
    @{ path = 'Titanic.csv'; message = 'Add Titanic dataset for classification examples and model tutorials.' },
    @{ path = 'weather_data.csv'; message = 'Add weather dataset used by climate/time-series notebooks.' },
    @{ path = 'welcome.txt'; message = 'Add workspace welcome notes and brief overview for collaborators.' }
)

# Auto-discover untracked and modified files and append them to the list
$existingPaths = $items | ForEach-Object { $_.path }

try {
    $gitUntracked = git ls-files --others --exclude-standard 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
} catch { $gitUntracked = @() }

try {
    $gitModified = git ls-files -m 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
} catch { $gitModified = @() }

$candidates = @($gitUntracked + $gitModified) | Select-Object -Unique
foreach ($f in $candidates) {
    if ($existingPaths -contains $f) { continue }
    if ($f -like 'untracked_backup_*' -or $f -eq '.gitignore') { continue }
    if ($gitUntracked -contains $f) {
        $msg = "Add $f (auto-generated commit message)"
    } else {
        $msg = "Update $f (auto-generated commit message)"
    }
    $items += @{ path = $f; message = $msg }
}

# We'll ensure a .gitignore entry for __pycache__ and commit it
$gitignorePath = '.gitignore'
$gitignoreMessage = 'Do not commit: add `.gitignore` entry to exclude `__pycache__` and avoid committing compiled Python bytecode.'

function Run-GitCommand {
    param($cmd)
    if ($DryRun) { Write-Host "DRYRUN: $cmd"; return 0 }
    iex $cmd
    return $LASTEXITCODE
}

Write-Host "Branch: $Branch`n" -ForegroundColor Cyan

# Fetch remote and detect if local and remote have diverged. Abort if so to avoid accidental overwrites.
try {
    git fetch origin $Branch 2>$null
} catch {
    # ignore fetch errors; we'll check divergence downstream
}

function Get-DivergenceCounts {
    param($branch)
    $counts = $null
    try {
        $counts = git rev-list --left-right --count "origin/$branch...$branch" 2>$null
    } catch {
        return $null
    }
    if (-not $counts) { return $null }
    $parts = $counts -split "\s+"
    if ($parts.Length -ge 2) { return @{ ahead = [int]$parts[1]; behind = [int]$parts[0] } }
    return $null
}

$div = Get-DivergenceCounts -branch $Branch
if ($div -ne $null -and ($div.ahead -gt 0 -and $div.behind -gt 0)) {
    Write-Host "Local and remote have diverged (ahead: $($div.ahead), behind: $($div.behind))." -ForegroundColor Red
    Write-Host "Resolve divergence before running this script. Suggested commands:" -ForegroundColor Yellow
    Write-Host "  git fetch origin" -ForegroundColor Cyan
    Write-Host "  git status --short --branch" -ForegroundColor Cyan
    Write-Host "  git pull --rebase origin $Branch  # or git merge origin/$Branch" -ForegroundColor Cyan
    exit 1
}

# Create or update .gitignore with __pycache__/
if (-not (Test-Path $gitignorePath)) {
    if ($DryRun) { Write-Host "DRYRUN: Create $gitignorePath with __pycache__/" }
    else { "__pycache__/" | Out-File -FilePath $gitignorePath -Encoding utf8 }
}
else {
    $existing = Get-Content $gitignorePath -ErrorAction SilentlyContinue
    if ($existing -notcontains '__pycache__/') {
        if ($DryRun) { Write-Host "DRYRUN: Append __pycache__/ to $gitignorePath" }
        else { Add-Content -Path $gitignorePath -Value '__pycache__/' }
    }
}

# Commit .gitignore if there are changes
if (Test-Path $gitignorePath) {
    if (-not $DryRun) { git add -- $gitignorePath }
    if ($DryRun) {
        Write-Host "DRYRUN: git commit -m \"$gitignoreMessage\" -- $gitignorePath"
    } else {
        $out = git commit -m "$gitignoreMessage" -- $gitignorePath 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { Write-Host ".gitignore: $out" -ForegroundColor Yellow } else {
            $pushOut = git push origin $Branch 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) { Write-Host "Push .gitignore failed:`n$pushOut" -ForegroundColor Red }
        }
    }
}

foreach ($item in $items) {
    $file = $item.path
    $msg = $item.message

    if (-not (Test-Path $file)) {
        Write-Host "Skipping missing file: $file" -ForegroundColor Yellow
        continue
    }

    Write-Host "Processing: $file" -ForegroundColor Green
    if ($DryRun) {
        Write-Host "DRYRUN: git add -- `"$file`""
        Write-Host "DRYRUN: git commit -m `"$msg`" -- `"$file`""
        Write-Host "DRYRUN: git push origin $Branch`n"
        continue
    }

    git add -- "$file"
    $commitOutput = (git commit -m "$msg" -- "$file" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        if ($commitOutput -match 'nothing to commit') {
            Write-Host ("No changes to commit for {0}" -f $file) -ForegroundColor Yellow
            continue
        }
        Write-Host ("Commit failed for {0}`n{1}" -f $file, $commitOutput) -ForegroundColor Red
        continue
    }

    $pushOutput = (git push origin $Branch 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("Push failed for {0}`n{1}" -f $file, $pushOutput) -ForegroundColor Red
        exit 1
    }
    Write-Host ("Committed & pushed: {0}`n" -f $file) -ForegroundColor Cyan
}

Write-Host "All done." -ForegroundColor Green
