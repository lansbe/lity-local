$ErrorActionPreference = "Stop"

function Get-PythonRunner {
    # Preferred Windows launcher: py -3
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Command = "py"; Args = @("-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Command = "python"; Args = @() }
    }
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        return @{ Command = "python3"; Args = @() }
    }
    return $null
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
    $Uv = "uv"
    $UvArgs = @()
} else {
    $Python = Get-PythonRunner
    if ($null -eq $Python) {
        Write-Host "Python 3 is not installed or not available in PATH."
        Write-Host "Install Python from https://www.python.org/downloads/windows/ and rerun this script."
        exit 1
    }

    & $Python.Command @($Python.Args + @("-m", "uv", "--version")) *> $null
    if ($LASTEXITCODE -eq 0) {
        $Uv = $Python.Command
        $UvArgs = @($Python.Args + @("-m", "uv"))
    } else {
        Write-Host "uv is required. Install it with:"
        Write-Host "powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
        exit 1
    }
}

& $Uv @UvArgs sync --extra desktop --extra web --extra packaging

# Build the web UI so PyInstaller can bundle it (web_dist). Skipped with a
# warning if npm is missing; the Qt UI still packages without it.
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Push-Location frontend
    npm install
    npm run build
    Pop-Location
} else {
    Write-Host "WARNING: npm not found - skipping web UI build."
    Write-Host "         The packaged app's '--ui web' mode needs frontend/ built first."
}

& $Uv @UvArgs run pyinstaller packaging/pyinstaller/lity.spec --noconfirm

Write-Host "Build complete: dist\Lity"
