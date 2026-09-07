"""
Tests for the npe2 reader contribution (_reader.py) — headless, no napari GUI.
"""
import warnings

import numpy as np
import pytest
import spatialdata
import tifffile
from hypothesis import given, settings
from hypothesis import strategies as hyp_st
from spatialdata.models import Image2DModel, get_model
from spatialdata.testing import assert_spatial_data_objects_are_identical

from LayerForge._reader import (
    SUPPORTED_SUFFIXES,
    _sanitize_element_name,
    _tiff_memmap_would_fail,
    default_scale_factors,
    load_image_as_sdata,
    napari_get_reader,
)


@pytest.mark.parametrize(
    "height,width,expected",
    [
        (900, 900, [2, 2]),  # matches the skin tutorial: 900 -> 450 -> 225
        (128, 128, None),  # already at/below the default min_size
        (256, 256, None),  # exactly at the threshold — no halving needed
        (4096, 2048, [2, 2, 2, 2]),
    ],
)
def test_default_scale_factors(height, width, expected):
    assert default_scale_factors(height, width) == expected


def test_default_scale_factors_custom_min_size():
    assert default_scale_factors(100, 100, min_size=32) == [2, 2]


def test_napari_get_reader_recognises_supported_suffixes(tmp_path):
    for suffix in SUPPORTED_SUFFIXES:
        path = tmp_path / f"image{suffix}"
        path.touch()
        assert napari_get_reader(str(path)) is not None


def test_napari_get_reader_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "image.svg"
    path.touch()
    assert napari_get_reader(str(path)) is None


def test_napari_get_reader_rejects_multi_file_selection(tmp_path):
    paths = [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]
    assert napari_get_reader(paths) is None


def test_load_image_as_sdata_grayscale_tiff(tmp_path):
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 255, size=(3, 64, 64), dtype=np.uint8)
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, mask)

    sdata = load_image_as_sdata(path)

    assert "sample" in sdata.images
    xarr = sdata.images["sample"]
    # small image (<= default min_size) -> single-scale DataArray, no pyramid
    assert xarr.sizes["y"] == 64
    assert xarr.sizes["x"] == 64
    assert xarr.sizes["c"] == 3


def test_load_image_as_sdata_builds_multiscale_pyramid(tmp_path):
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 255, size=(1, 900, 900), dtype=np.uint8)
    path = tmp_path / "big.tif"
    tifffile.imwrite(path, mask)

    sdata = load_image_as_sdata(path)

    element = sdata.images["big"]
    # multiscale elements expose their pyramid levels via a mapping interface
    assert sorted(element.keys()) == ["scale0", "scale1", "scale2"]


def test_load_image_as_sdata_custom_image_key(tmp_path):
    mask = np.zeros((1, 32, 32), dtype=np.uint8)
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, mask)

    sdata = load_image_as_sdata(path, image_key="custom")

    assert "custom" in sdata.images


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("Sample 4 PIP transverse", "Sample_4_PIP_transverse"),
        ("scan (1), 50%#2", "scan__1___50__2"),
        ("plane.ome", "plane.ome"),  # dots are legal — must survive untouched
        ("already-valid_name.1", "already-valid_name.1"),
        ("Müller_scan", "Müller_scan"),  # isalnum() is Unicode-aware
        ("__hidden", "_hidden"),  # "__" prefix is reserved by SpatialData
        ("   ", "image"),  # nothing but separators left
        ("", "image"),  # e.g. a file named ".tif"
        (".", "image"),
        ("..", "image"),
    ],
)
def test_sanitize_element_name(stem, expected):
    assert _sanitize_element_name(stem) == expected


def test_load_image_as_sdata_sanitizes_filename_with_spaces(tmp_path):
    """
    Regression test: SpatialData rejects element names containing anything but
    alphanumerics, '_', '-' and '.', so a microscope export whose filename has
    spaces used to crash File > Open with a ValidationError.
    """
    path = tmp_path / "13-39-06_FINGER001 Sample 4 PIP transverse Z0319.ome.tif"
    tifffile.imwrite(path, np.zeros((64, 48), dtype=np.uint16))

    with pytest.warns(UserWarning, match="does not"):
        sdata = load_image_as_sdata(path)

    assert "13-39-06_FINGER001_Sample_4_PIP_transverse_Z0319.ome" in sdata.images


def test_load_image_as_sdata_keeps_valid_stem_unchanged(tmp_path):
    """An already-valid stem is used verbatim, with no warning."""
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, np.zeros((1, 32, 32), dtype=np.uint8))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = load_image_as_sdata(path)

    assert "sample" in sdata.images


@pytest.fixture
def read_only_tiff(tmp_path):
    """A TIFF the process cannot open read-write, as on read-only media."""
    path = tmp_path / "readonly.tif"
    tifffile.imwrite(path, np.zeros((3, 64, 64), dtype=np.uint16))
    path.chmod(0o444)
    yield path
    path.chmod(0o644)


def test_load_image_as_sdata_read_only_tiff(read_only_tiff):
    """
    Regression test: spatialdata_io memory-maps TIFFs read-write ('r+'), which
    raises OSError for files on read-only media (Errno 30) or without write
    permission (Errno 13). The image is still readable, so loading must succeed.
    """
    sdata = load_image_as_sdata(read_only_tiff)

    xarr = sdata.images["readonly"]
    assert (xarr.sizes["c"], xarr.sizes["y"], xarr.sizes["x"]) == (3, 64, 64)


def test_load_image_as_sdata_single_plane_2d_tiff(tmp_path):
    """A 2D single-plane TIFF (e.g. one light-sheet plane) loads as c=1."""
    path = tmp_path / "plane.ome.tif"
    tifffile.imwrite(path, np.zeros((64, 48), dtype=np.uint16))

    sdata = load_image_as_sdata(path)

    xarr = sdata.images["plane.ome"]
    assert (xarr.sizes["c"], xarr.sizes["y"], xarr.sizes["x"]) == (1, 64, 48)


def test_tiff_memmap_would_fail_on_unwritable_file(read_only_tiff):
    assert _tiff_memmap_would_fail(read_only_tiff, ["c", "y", "x"]) is True


def test_tiff_memmap_would_fail_on_axis_count_mismatch(tmp_path):
    path = tmp_path / "plane.tif"
    tifffile.imwrite(path, np.zeros((64, 48), dtype=np.uint16))

    assert _tiff_memmap_would_fail(path, ["c", "y", "x"]) is True
    assert _tiff_memmap_would_fail(path, ["y", "x"]) is False


def test_tiff_memmap_would_fail_is_false_for_ordinary_tiff(tmp_path):
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, np.zeros((3, 64, 48), dtype=np.uint16))

    assert _tiff_memmap_would_fail(path, ["c", "y", "x"]) is False


def test_tiff_memmap_would_fail_is_false_for_non_tiff(tmp_path):
    path = tmp_path / "image.png"
    path.touch()

    assert _tiff_memmap_would_fail(path, ["y", "x", "c"]) is False


def test_load_image_as_sdata_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "image.svg"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_image_as_sdata(path)


# ---------------------------------------------------------------------------
# napari reader contract
#
# Shape taken from the napari-plugin-template's generated test_reader.py: a
# reader plugin's public contract is "napari_get_reader returns a callable
# that returns a list of (data, kwargs, layer_type) tuples", and the pixels
# must survive the trip. Asserting only on array *shapes* would let a reader
# that transposed axes or returned zeros pass.
# ---------------------------------------------------------------------------

def test_reader_returns_layer_data_tuples_and_preserves_pixels(tmp_path):
    rng = np.random.default_rng(3)
    original = rng.integers(0, 4095, size=(3, 64, 48), dtype=np.uint16)
    path = tmp_path / "contract.tif"
    tifffile.imwrite(path, original)

    reader = napari_get_reader(str(path))
    assert callable(reader)

    layer_data_list = reader(str(path))
    assert isinstance(layer_data_list, list)
    assert len(layer_data_list) == 3  # one layer per channel

    for ch_idx, layer_data_tuple in enumerate(layer_data_list):
        assert isinstance(layer_data_tuple, tuple)
        assert len(layer_data_tuple) == 3
        data, add_kwargs, layer_type = layer_data_tuple
        assert layer_type == "image"
        assert isinstance(add_kwargs, dict)
        np.testing.assert_array_equal(np.asarray(data), original[ch_idx])


def test_reader_layers_carry_sdata_metadata(tmp_path):
    """
    Every layer must carry {"sdata", "image_key"} in its metadata — the
    convention napari-spatialdata uses, and what _widgets._sdata_from_viewer
    relies on to recover the SpatialData object from the viewer.
    """
    path = tmp_path / "meta.tif"
    tifffile.imwrite(path, np.zeros((2, 32, 32), dtype=np.uint8))

    layer_data_list = napari_get_reader(str(path))(str(path))

    for ch_idx, (_data, add_kwargs, _type) in enumerate(layer_data_list):
        metadata = add_kwargs["metadata"]
        assert isinstance(metadata["sdata"], spatialdata.SpatialData)
        assert metadata["image_key"] in metadata["sdata"].images
        assert metadata["channel_index"] == ch_idx


def test_reader_preserves_pixels_for_multiscale_image(tmp_path):
    """Pyramid building must not disturb scale0, which is what napari shows first."""
    rng = np.random.default_rng(4)
    original = rng.integers(0, 255, size=(1, 900, 900), dtype=np.uint8)
    path = tmp_path / "pyramid.tif"
    tifffile.imwrite(path, original)

    (data, _kwargs, _type) = napari_get_reader(str(path))(str(path))[0]

    # multiscale layers hand napari a list of levels, highest resolution first
    assert isinstance(data, list)
    np.testing.assert_array_equal(np.asarray(data[0]), original[0])


def test_memmap_and_non_memmap_paths_agree_on_pixels(tmp_path):
    """
    _read_image_element has two branches — spatialdata_io's memory-mapped TIFF
    fast path, and the use_tiff_memmap=False fallback taken for read-only
    files. They must produce identical pixels, or a read-only drive would
    silently yield different data from the same file.
    """
    rng = np.random.default_rng(5)
    original = rng.integers(0, 4095, size=(3, 64, 48), dtype=np.uint16)

    writable = tmp_path / "writable.tif"
    read_only = tmp_path / "read_only.tif"
    tifffile.imwrite(writable, original)
    tifffile.imwrite(read_only, original)
    read_only.chmod(0o444)
    try:
        assert _tiff_memmap_would_fail(writable, ["c", "y", "x"]) is False
        assert _tiff_memmap_would_fail(read_only, ["c", "y", "x"]) is True

        via_memmap = load_image_as_sdata(writable).images["writable"]
        via_fallback = load_image_as_sdata(read_only).images["read_only"]

        np.testing.assert_array_equal(via_memmap.to_numpy(), via_fallback.to_numpy())
        np.testing.assert_array_equal(via_memmap.to_numpy(), original)
    finally:
        read_only.chmod(0o644)


# ---------------------------------------------------------------------------
# SpatialData contract
# ---------------------------------------------------------------------------

def test_loaded_element_is_a_valid_spatialdata_image(tmp_path):
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, np.zeros((3, 64, 48), dtype=np.uint8))

    element = load_image_as_sdata(path).images["sample"]

    assert get_model(element) is Image2DModel
    Image2DModel.validate(element)  # raises if the element is malformed


def test_sdata_round_trips_through_zarr_for_messy_filename(tmp_path):
    """
    Element names are restricted because they become group paths inside the
    zarr store, so the real test of a sanitized name is that the object it
    keys can actually be written and read back.
    """
    path = tmp_path / "Sample 4 (PIP), 1.6X zoom.tif"
    tifffile.imwrite(path, np.zeros((1, 64, 64), dtype=np.uint8))

    with pytest.warns(UserWarning):
        sdata = load_image_as_sdata(path)

    zarr_path = tmp_path / "out.zarr"
    sdata.write(zarr_path)

    assert_spatial_data_objects_are_identical(sdata, spatialdata.read_zarr(zarr_path))


# ---------------------------------------------------------------------------
# Property-based test of the name sanitizer
# ---------------------------------------------------------------------------

@given(hyp_st.text())
@settings(max_examples=500, deadline=None)
def test_sanitize_element_name_always_produces_a_valid_name(raw):
    """
    The invariant, checked against spatialdata's own validator rather than a
    reimplementation of its rules: whatever the stem, the sanitized name must
    be accepted by SpatialData. Importing the private validator is deliberate
    — if upstream tightens the rules, this test is where we find out.
    """
    from spatialdata._core.validation import check_valid_name

    check_valid_name(_sanitize_element_name(raw))


@given(hyp_st.text(min_size=1))
@settings(max_examples=200, deadline=None)
def test_sanitize_element_name_is_idempotent(raw):
    once = _sanitize_element_name(raw)
    assert _sanitize_element_name(once) == once
