"""Validate the npe2 plugin manifest (src/LayerForge/napari.yaml)."""
from pathlib import Path

from npe2 import PluginManifest

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "src" / "LayerForge" / "napari.yaml"


def test_manifest_is_valid():
    manifest = PluginManifest.from_file(MANIFEST_PATH)
    assert manifest.name == "LayerForge"

    command_ids = {c.id for c in manifest.contributions.commands}
    assert "LayerForge.get_reader" in command_ids
    assert "LayerForge.make_annotation_widget" in command_ids

    reader_commands = {r.command for r in manifest.contributions.readers}
    assert "LayerForge.get_reader" in reader_commands

    widget_commands = {w.command for w in manifest.contributions.widgets}
    assert "LayerForge.make_annotation_widget" in widget_commands
