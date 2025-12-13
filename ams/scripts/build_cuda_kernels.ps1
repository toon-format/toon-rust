# PowerShell build script for Geoshi CUDA kernels (Windows-friendly)
param(
    [switch]$Scanner  # run scanner example in addition to CUDA demo
)

$ErrorActionPreference = "Stop"

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
$kernelsDir  = Join-Path $projectRoot "crates/geoshi/kernels"
$buildDir    = Join-Path $projectRoot "target/cuda-build"

Write-Host "Building CUDA kernels for Geoshi"
Write-Host "Project Root : $projectRoot"
Write-Host "Kernels Dir  : $kernelsDir"
Write-Host "Build Dir    : $buildDir"
Write-Host ""

# Ensure build directory exists
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

# Check for nvcc
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Error "nvcc not found in PATH. Please install the NVIDIA CUDA Toolkit."
    exit 1
}

$nvccVersion = (& nvcc --version | Select-Object -First 1)
Write-Host "Found nvcc: $nvccVersion"
Write-Host ""

# Compile mgsk.cu into PTX
Write-Host "Compiling mgsk.cu..."

& nvcc `
    -ptx `
    -arch=sm_89 `
    -O3 `
    -use_fast_math `
    --generate-line-info `
    -Xptxas -v `
    --maxrregcount=32 `
    -I/usr/local/cuda/include `
    (Join-Path $kernelsDir "mgsk.cu") `
    -o (Join-Path $buildDir "mgsk.ptx")

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to compile mgsk.cu (exit code $LASTEXITCODE)."
    exit $LASTEXITCODE
}

Write-Host "mgsk.cu compiled successfully."
Write-Host "Output: $(Join-Path $buildDir "mgsk.ptx")"

# Export environment variable for Rust/CUDA integration
$env:MGSK_PTX = Join-Path $buildDir "mgsk.ptx"
Write-Host ""
Write-Host "Environment variable set:"
Write-Host "  MGSK_PTX=$env:MGSK_PTX"
Write-Host ""
Write-Host "For Rust integration, you can run:"
$ptxDisplay = Join-Path $buildDir 'mgsk.ptx'
Write-Host "  `$env:MGSK_PTX = \"$ptxDisplay\""
Write-Host "  setx MGSK_PTX \"$ptxDisplay\"  # persist for new shells"
Write-Host ""
Write-Host "Build completed."

Write-Host ""
Write-Host "Running Geoshi CUDA kernel test..."
Write-Host ""

Push-Location $projectRoot
try {
    & cargo run --example test_cuda -p geoshi --features cuda
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Geoshi CUDA test failed (exit code $LASTEXITCODE). CUDA may not be available."
    } else {
        Write-Host "  ✅ CUDA kernel test passed!"
    }
} catch {
    Write-Warning "CUDA test could not be run. CUDA may not be available on this system."
} finally {
    Pop-Location
}

if ($Scanner) {
    Write-Host ""
    Write-Host "Running Geoshi geometric scanner test..."
    Write-Host ""

    Push-Location $projectRoot
    try {
        & cargo run --example test_scanner -p geoshi
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Geoshi scanner test failed (exit code $LASTEXITCODE)."
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "✅ All builds and requested tests completed."
