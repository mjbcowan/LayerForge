<#
.SYNOPSIS
    One-time install of the LayerForge napari plugin into the official
    signed napari bundle, from the internal network share.

.DESCRIPTION
    Run this once per machine, after installing napari from its official
    signed Windows installer (https://github.com/napari/napari/releases).
    It runs `pip install` against the LayerForge wheel on the internal
    share, using the bundled napari environment's own Python — no custom
    installer, no code signing, nothing beyond what installing napari
    itself already required.

.PARAMETER NapariPythonPath
    Path to python.exe inside the installed napari bundle (check your
    napari Start Menu shortcut's "Open file location" if unsure).

.PARAMETER WheelPath
    Path (UNC) to the LayerForge wheel on the internal share, e.g.
    "\\share\layerforge\layerforge-0.4.0-py3-none-any.whl".

.EXAMPLE
    .\install-layerforge.ps1 -NapariPythonPath "$env:LOCALAPPDATA\napari\python.exe" -WheelPath "\\share\layerforge\layerforge-0.4.0-py3-none-any.whl"
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$NapariPythonPath,

    [Parameter(Mandatory = $true)]
    [string]$WheelPath
)

if (-not (Test-Path $NapariPythonPath)) {
    throw "Could not find napari's bundled python at '$NapariPythonPath'."
}
if (-not (Test-Path $WheelPath)) {
    throw "Could not find the LayerForge wheel at '$WheelPath'."
}

& $NapariPythonPath -m pip install --upgrade $WheelPath
