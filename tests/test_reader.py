"""
Tests for the npe2 reader contribution (_reader.py) — headless, no napari GUI.
"""
import numpy as np
import pytest
import tifffile

from LayerForge._reader import (
    SUPPORTED_SUFFIXES,
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
