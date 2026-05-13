param(
    [string]$RepoRoot = "D:\MARL\MRAL-Server\MARL\pymarl",
    [string]$ResultsRoot = "D:\MARL\MRAL-Server\MARL\pymarl-results",
    [switch]$CopyPaperFigures
)

$ErrorActionPreference = "Stop"

$sacredDir = Join-Path $ResultsRoot "sacred"
$diagnosticsDir = Join-Path $ResultsRoot "diagnostics"
$figuresDir = Join-Path $ResultsRoot "figures"
$paperFiguresDir = Join-Path $RepoRoot "paper\latex\figures"

if (-not (Test-Path $RepoRoot)) {
    throw "Missing repo root: $RepoRoot"
}
if (-not (Test-Path $sacredDir)) {
    throw "Missing Sacred directory. Download server results to: $sacredDir"
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $diagnosticsDir, $figuresDir | Out-Null

Write-Host "Regenerating summaries..."
python scripts\summarize_marl_transfer_adaptation.py `
    --sacred-dir $sacredDir `
    --output-dir $diagnosticsDir `
    --maps 5m_vs_6m,3s5z `
    --primary-configs qmix,qmix_attnres_l2,qmix_attncomm_l2_other,qmix_attncomm_l2_self `
    --seeds 1,2,3 `
    --include-cross `
    --cross-pairs qmix:qmix_attnres_l2,qmix:qmix_attncomm_l2_other,qmix:qmix_attncomm_l2_self

Write-Host "Regenerating learning curves..."
python scripts\plot_marl_transfer_curves.py `
    --sacred-dir $sacredDir `
    --output-dir $figuresDir `
    --seeds 1,2,3

Write-Host "Regenerating AttnComm heatmaps..."
python scripts\plot_comm_attn_heatmaps.py `
    --sacred-dir $sacredDir `
    --output-dir $figuresDir

$primaryPath = Join-Path $diagnosticsDir "marl_transfer_primary_qmix_table.csv"
$missingPath = Join-Path $diagnosticsDir "marl_transfer_missing_or_partial.csv"
$aggregatePath = Join-Path $diagnosticsDir "marl_transfer_cross_algorithm_aggregate.csv"

if (-not (Test-Path $primaryPath)) {
    throw "Missing generated primary table: $primaryPath"
}

$required = @(
    "qmix_attnres_l2",
    "qmix_attncomm_l2_other",
    "qmix_attncomm_l2_self"
)

$primary = Import-Csv $primaryPath
$problems = @()
foreach ($config in $required) {
    $row = $primary | Where-Object { $_.map -eq "3s5z" -and $_.config -eq $config } | Select-Object -First 1
    if ($null -eq $row) {
        $problems += "Missing primary summary row: 3s5z $config"
        continue
    }
    if ([int]$row.complete_seeds -ne 3) {
        $problems += "Expected 3 complete seeds for 3s5z $config, got $($row.complete_seeds)/$($row.expected_seeds), missing_or_partial=$($row.missing_or_partial)"
    }
}

if (Test-Path $missingPath) {
    $badMissing = Import-Csv $missingPath | Where-Object {
        $_.map -eq "3s5z" -and (
            $_.config -eq "qmix_attnres_l2" -or
            $_.config -eq "qmix_attncomm_l2_other" -or
            $_.config -eq "qmix_attncomm_l2_self"
        )
    }
    foreach ($row in $badMissing) {
        $problems += "Still missing/partial: group=$($row.group) map=$($row.map) config=$($row.config) seed=$($row.seed) issue=$($row.issue) run_id=$($row.run_id)"
    }
}

$expectedFigures = @(
    "3s5z_qmix_win_curve.pdf",
    "3s5z_qmix_attncomm_l2_other_l0_attncomm_attention_heatmap.pdf",
    "3s5z_qmix_attncomm_l2_other_l1_attncomm_attention_heatmap.pdf",
    "3s5z_qmix_attncomm_l2_self_l0_attncomm_attention_heatmap.pdf",
    "3s5z_qmix_attncomm_l2_self_l1_attncomm_attention_heatmap.pdf"
)
foreach ($name in $expectedFigures) {
    $path = Join-Path $figuresDir $name
    if (-not (Test-Path $path)) {
        $problems += "Missing figure: $path"
    }
}

Write-Host ""
Write-Host "3s5z primary rows:"
$primary |
    Where-Object { $_.map -eq "3s5z" -and ($required -contains $_.config -or $_.config -eq "qmix") } |
    Format-Table map, config, complete_seeds, expected_seeds, final_win_mean, best_win_mean, win_auc_mean, wall_hours_mean -AutoSize

if (Test-Path $aggregatePath) {
    Write-Host "3s5z paired deltas:"
    Import-Csv $aggregatePath |
        Where-Object { $_.map -eq "3s5z" -and ($required -contains $_.candidate) } |
        Format-Table map, baseline, candidate, paired_seeds, mean_delta_final_win, mean_delta_win_auc, mean_wall_time_ratio, paper_reading -AutoSize
}

if ($CopyPaperFigures) {
    New-Item -ItemType Directory -Force -Path $paperFiguresDir | Out-Null
    foreach ($name in $expectedFigures) {
        Copy-Item -Force -Path (Join-Path $figuresDir $name) -Destination (Join-Path $paperFiguresDir $name)
    }
    Write-Host "Copied selected 3s5z figures to $paperFiguresDir"
}

if ($problems.Count -gt 0) {
    Write-Host ""
    Write-Host "Validation failed:"
    $problems | ForEach-Object { Write-Host " - $_" }
    exit 1
}

Write-Host ""
Write-Host "Validation passed. 3s5z rerun summaries and figures are complete."
