# Dot-source this file: . ./scripts/env.ps1. All writes stay inside the project.
$env:WAVEFORGE_PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$env:PIP_CACHE_DIR = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/pip'
$env:PIP_CONFIG_FILE = 'NUL'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PIP_REQUIRE_VIRTUALENV = 'true'
$env:MPLCONFIGDIR = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/matplotlib'
$env:MPLBACKEND = 'Agg'
$env:NUMBA_CACHE_DIR = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/numba'
$env:XDG_CACHE_HOME = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache'
$env:XDG_CONFIG_HOME = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/config'
$env:XDG_DATA_HOME = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/data'
$env:TMPDIR = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/tmp'
$env:TMP = $env:TMPDIR
$env:TEMP = $env:TMPDIR
$env:PYTHONPYCACHEPREFIX = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/python'
$env:PYTHONUSERBASE = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/python-userbase'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONHASHSEED = '0'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:COVERAGE_FILE = Join-Path $env:WAVEFORGE_PROJECT_ROOT '.cache/coverage/.coverage'
$waveforgeDirectories = @(
    '.cache/pip', '.cache/matplotlib', '.cache/numba', '.cache/pytest',
    '.cache/python', '.cache/tmp', '.cache/config', '.cache/data',
    '.cache/python-userbase', '.cache/coverage', '.cache/experiments', '.external', '.tools',
    'results', 'figures', 'logs', 'data/generated'
)
foreach ($waveforgeDirectory in $waveforgeDirectories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $env:WAVEFORGE_PROJECT_ROOT $waveforgeDirectory) | Out-Null
}
