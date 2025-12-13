<#
Run Clippy with strict pedantic lints via PowerShell.

Usage:
    ./scripts/clippy.ps1 [-Workspace] [-Targets] [-Features] [-ExtraArgs <args>]

Default runs: cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic
#>
param (
    [switch]$Workspace,
    [switch]$Targets,
    [switch]$Features,
    [string]$ExtraArgs = ""
)

# Build the cargo clippy base args
$baseArgs = @("clippy")

if ($Workspace) { $baseArgs += "--workspace" }
if ($Targets) { $baseArgs += "--all-targets" }
if ($Features) { $baseArgs += "--all-features" }

# Default clippy flags to enforce: deny warnings and enable pedantic
$clippyFlags = "-D warnings -W clippy::pedantic"
if ($ExtraArgs -ne "") { $clippyFlags += " $ExtraArgs" }

# NOTE: Important `--` separator is required to pass flags to clippy/rustc
$cmd = "cargo $($baseArgs -join ' ') -- $clippyFlags"
Write-Host "Running: $cmd"

# Execute the command and propagate the exit code
Invoke-Expression $cmd
exit $LASTEXITCODE
