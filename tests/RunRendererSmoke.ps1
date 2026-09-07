param([string]$BuildDirectory = "$PSScriptRoot/../cmake-build-release")
$ErrorActionPreference = 'Stop'
$renderBuild = (Resolve-Path -LiteralPath $BuildDirectory).Path
$renderExe = Join-Path $renderBuild 'renderer_smoke.exe'
if (-not (Test-Path -LiteralPath $renderExe)) { throw "Build the renderer_smoke target first." }
$previousValidation = $env:GABGL_DX12_VALIDATION
try {
    $env:GABGL_DX12_VALIDATION = '1'
    foreach ($backend in @('dx12', 'opengl')) {
        $stdout = Join-Path $renderBuild "$backend-pipeline-check.stdout.log"
        $stderr = Join-Path $renderBuild "$backend-pipeline-check.stderr.log"
        Write-Output "Starting $backend integration test"
        $launch = @{
            FilePath = $renderExe
            WorkingDirectory = $renderBuild
            WindowStyle = 'Hidden'
            RedirectStandardOutput = $stdout
            RedirectStandardError = $stderr
            PassThru = $true
        }
        if ($backend -eq 'opengl') { $launch.ArgumentList = '--opengl' }
        $renderProcess = Start-Process @launch
        # Retain the process handle so Windows PowerShell can read ExitCode
        # after a short-lived child has already exited.
        $null = $renderProcess.Handle
        if (-not $renderProcess.WaitForExit(120000)) {
            $renderProcess.Kill()
            throw "$backend integration test timed out. See $stdout and $stderr"
        }
        $renderProcess.Refresh()
        $logs = (Get-Content -LiteralPath $stdout, $stderr -Raw) -join "`n"
        if ($renderProcess.ExitCode -ne 0 -or $logs -match '\[ERROR\]|smoke FAIL' -or $logs -notmatch 'smoke PASS') {
            throw "$backend integration test failed (exit $($renderProcess.ExitCode)). See $stdout and $stderr"
        }
        ($logs -split "`n" | Where-Object { $_ -match 'smoke PASS|D3D12 validation' }) | Write-Output
    }
} finally {
    $env:GABGL_DX12_VALIDATION = $previousValidation
}
