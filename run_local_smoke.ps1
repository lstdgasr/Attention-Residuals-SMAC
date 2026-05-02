$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResultsDir = "D:\MARL\pymarl-results"
$PythonExe = "D:\Anaconda\envs\pymarl-sc2\python.exe"

if (-not $env:SC2PATH) {
    $env:SC2PATH = "D:\StarCraft II"
}

if (-not (Test-Path $env:SC2PATH)) {
    throw "SC2PATH does not exist: $env:SC2PATH"
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable does not exist: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

Push-Location $RepoRoot
try {
    & $PythonExe src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m local_results_path="$ResultsDir" use_cuda=False t_max=5000 test_nepisode=4 save_model=False
}
finally {
    Pop-Location
}
