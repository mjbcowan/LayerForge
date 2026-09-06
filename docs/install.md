# Installation instructions

LayerForge ships as a **napari plugin**, not a standalone app. Colleagues
install the official napari desktop bundle once, then add LayerForge to it
with a single command.

## 1. Install napari (official, signed installer)

Download the appropriate installer from napari's GitHub releases:
https://github.com/napari/napari/releases

- **Windows**: Authenticode-signed under NumFOCUS.
- **macOS**: signed and notarized `.pkg` by the napari team.

We do not build or sign our own installer — LayerForge rides on top of
napari's existing, already-signed distribution.

## 2. Get the LayerForge wheel

**Where to get it (pick whichever your team has set up):**

- **GitHub Releases (always available, no extra setup)** — every published
  release attaches a `layerforge-<version>-py3-none-any.whl` file directly
  to it: https://github.com/mjbcowan/LayerForge/releases — click the
  latest release, then download the `.whl` from its **Assets**. This repo
  is private, so you'll need GitHub access to the repo (the same access
  you'd use to browse the code) — ask your team to add you if you can't
  see it.
- **Internal network share (if your team has wired this up)** — some teams
  additionally mirror the wheel to a network share on release (see
  `.github/workflows/publish-wheel.yml`, `publish-to-internal-share` job),
  e.g. `\\share\layerforge\layerforge-0.4.0-py3-none-any.whl`. This is
  opt-in infrastructure (a self-hosted CI runner + a configured share
  path) — ask your team whether it's set up before assuming a wheel is
  there.

## 3. Install the LayerForge plugin (one-time, per machine)

Once you have the `.whl` file (downloaded from GitHub, or from the share):

**Option A — napari's built-in Plugin Manager (recommended for most users)**

`Plugins > Install/Uninstall Plugins...`, then drag the `.whl` file onto
the dialog (or paste its path into the text box at the bottom) and click
**Install**. No terminal required — see `docs/annotation_protocol.md` for
a fully click-by-click walkthrough.

**Option B — one-line command**

Run this once, using the Python bundled *inside* the napari installation
(not your system Python):

```
<path-to-napari-bundle>\python.exe -m pip install --upgrade <path-to-downloaded-wheel>\layerforge-0.4.0-py3-none-any.whl
```

Convenience wrapper scripts are provided so nobody has to remember the
bundle's python path by hand:

- Windows: `scripts/install-layerforge.ps1`
- macOS/Linux: `scripts/install-layerforge.sh`

```powershell
.\scripts\install-layerforge.ps1 -NapariPythonPath "$env:LOCALAPPDATA\napari\python.exe" -WheelPath "<path-to-downloaded-wheel>\layerforge-0.4.0-py3-none-any.whl"
```

## 3. Using it

Open napari, then `File > Open File(s)...` (or drag-and-drop) a
`.tif`/`.tiff`/`.png`/`.jpg`/`.jpeg` image — LayerForge's reader loads it
into a `SpatialData` object automatically. Then open
`Plugins > LayerForge > LayerForge annotation panel` to define classes,
choose labels/shapes mode, and launch the annotation session.

## Updates

No custom update checker is needed: once installed, LayerForge shows up in
napari's own Plugin Manager like any other plugin, including its installed
version (see the "LayerForge vX.Y.Z" label in the annotation panel) and
update status once the internal share is wired in as a plugin source.

## Developing LayerForge itself

```bash
pip install git+https://<PAT>@github.com/mjbcowan/LayerForge@main
# or, for local development:
pip install -e ".[dev]"
```
