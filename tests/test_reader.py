"""
Tests for the npe2 reader contribution (_reader.py) — headless, no napari GUI.
"""
import numpy as np
import pytest
import tifffile

from LayerForge._reader import (
    SUPPORTED_SUFFIXES,
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


def test_load_image_as_sdata_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "image.svg"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_image_as_sdata(path)
