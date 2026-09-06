#!/usr/bin/env bash
# One-time install of the LayerForge napari plugin into the official signed
# napari bundle (macOS/Linux), from the internal network share.
#
# Usage:
#   ./install-layerforge.sh <path-to-napari-bundle-python> <path-to-layerforge-wheel>
#
# Example:
#   ./install-layerforge.sh \
#       /Applications/napari.app/Contents/Resources/bin/python3 \
#       /Volumes/share/layerforge/layerforge-0.4.0-py3-none-any.whl
set -euo pipefail

NAPARI_PYTHON="${1:?Usage: $0 <path-to-napari-bundle-python> <path-to-layerforge-wheel>}"
WHEEL_PATH="${2:?Usage: $0 <path-to-napari-bundle-python> <path-to-layerforge-wheel>}"

if [ ! -x "$NAPARI_PYTHON" ]; then
    echo "Could not find napari's bundled python at '$NAPARI_PYTHON'." >&2
    exit 1
fi
if [ ! -f "$WHEEL_PATH" ]; then
    echo "Could not find the LayerForge wheel at '$WHEEL_PATH'." >&2
    exit 1
fi

"$NAPARI_PYTHON" -m pip install --upgrade "$WHEEL_PATH"
