"""
End-to-end tests that drive LayerForge through napari's real plugin
machinery rather than calling its Python functions directly.

The reader is registered with napari via an npe2 manifest, so
``viewer.open(..., plugin="LayerForge")`` exercises a chain the unit tests in
test_reader.py never touch: npe2 manifest discovery -> command resolution ->
``ReaderContribution.exec`` -> ``_reader_function`` -> SpatialData
construction -> napari layer creation. Bugs have shipped in that gap before
(see test_open_file_with_spaces_in_name), so it is covered here.

These tests need a real Qt/OpenGL context. On Linux CI that is provided by
QT_QPA_PLATFORM=offscreen plus a headless GL display; on macOS do *not* set
QT_QPA_PLATFORM=offscreen, which leaves vispy without a GL context.
"""
import numpy as np
import pytest
import tifffile

from LayerForge._version import __version__


def test_layerforge_reader_is_discoverable_by_npe2():
    """The manifest is registered via the napari.manifest entry point."""
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()

    # raises KeyError if the napari.manifest entry point did not register
    manifest = pm.get_manifest("LayerForge")
    assert manifest.package_version == __version__
    assert [r.command for r in manifest.contributions.readers] == [
        "LayerForge.get_reader"
    ]


def test_viewer_open_with_layerforge_plugin_preserves_pixels(make_napari_viewer, tmp_path):
    rng = np.random.default_rng(7)
    original = rng.integers(0, 4095, size=(3, 64, 48), dtype=np.uint16)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, original)

    viewer = make_napari_viewer()
    layers = viewer.open(path, plugin="LayerForge")

    assert len(layers) == 3  # one layer per channel
    for ch_idx, layer in enumerate(layers):
        np.testing.assert_array_equal(np.asarray(layer.data), original[ch_idx])
        assert layer.metadata["image_key"] == "stack"
        assert layer.metadata["channel_index"] == ch_idx


def test_open_file_with_spaces_in_name(make_napari_viewer, tmp_path):
    """
    Regression test for the reported MultipleReaderError/ValidationError crash:
    opening a microscope export whose filename contains spaces used to abort
    File > Open, because the filename stem was used verbatim as a SpatialData
    element name and SpatialData rejects spaces.

    Driven through viewer.open() rather than load_image_as_sdata() because the
    failure only surfaced once npe2 dispatched to the reader.
    """
    original = np.zeros((1, 64, 48), dtype=np.uint16)
    path = tmp_path / "13-39-06_FINGER001 Sample 4 PIP 1point6X zoom L transverse Z0319.ome.tif"
    tifffile.imwrite(path, original)

    viewer = make_napari_viewer()
    with pytest.warns(UserWarning, match="does not allow in element names"):
        layers = viewer.open(path, plugin="LayerForge")

    assert len(layers) == 1
    expected_key = (
        "13-39-06_FINGER001_Sample_4_PIP_1point6X_zoom_L_transverse_Z0319.ome"
    )
    assert layers[0].metadata["image_key"] == expected_key
    assert expected_key in layers[0].metadata["sdata"].images


def test_open_large_single_plane_builds_multiscale_layer(make_napari_viewer, tmp_path):
    """
    A single 2D light-sheet plane — the shape in the user's report (2560x2160,
    one plane) — must load as a multiscale layer rather than failing the
    (c, y, x) transpose.
    """
    path = tmp_path / "plane.ome.tif"
    tifffile.imwrite(path, np.zeros((900, 900), dtype=np.uint16))

    viewer = make_napari_viewer()
    layers = viewer.open(path, plugin="LayerForge")

    assert len(layers) == 1
    assert layers[0].multiscale
    assert layers[0].data[0].shape == (900, 900)
