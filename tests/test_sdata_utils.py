"""
Tests for sdata_utils — all headless (no napari, no GUI).
"""
import numpy as np
import pytest
import spatialdata
from spatialdata.models import Image2DModel

from reimagined_guacamole.sdata_utils import (
    get_scale0_xarray,
    get_scale0_shape,
    get_channel_names,
    copy_transform,
    wrap_to_multiscale_labels,
    class_id_colormap,
    print_sdata_summary,
)


@pytest.fixture
def demo_sdata():
    """Minimal 2-channel, multiscale SpatialData object."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(2, 64, 64), dtype=np.uint8)
    image = Image2DModel.parse(
        data,
        dims=["c", "y", "x"],
        c_coords=["DAPI", "GFP"],
        scale_factors=[2],
    )
    return spatialdata.SpatialData(images={"image": image})


def test_get_scale0_shape(demo_sdata):
    h, w = get_scale0_shape(demo_sdata, "image")
    assert h == 64
    assert w == 64


def test_get_scale0_xarray(demo_sdata):
    xarr = get_scale0_xarray(demo_sdata, "image")
    assert "y" in xarr.dims
    assert "x" in xarr.dims
    assert xarr.sizes["y"] == 64


def test_get_channel_names(demo_sdata):
    names = get_channel_names(demo_sdata, "image")
    assert names == ["DAPI", "GFP"]


def test_wrap_to_multiscale_labels(demo_sdata):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[:32, :] = 1
    element = wrap_to_multiscale_labels(mask, demo_sdata, scale_factors=[2])
    assert element is not None


def test_class_id_colormap():
    cmap = class_id_colormap(3)
    assert 0 in cmap                        # background always present
    assert cmap[0][3] == 0.0               # background is transparent
    assert len(cmap) == 4                  # background + 3 classes
    for i in range(1, 4):
        assert len(cmap[i]) == 4           # RGBA
        assert all(0.0 <= v <= 1.0 for v in cmap[i])


def test_print_sdata_summary_does_not_raise(demo_sdata, capsys):
    print_sdata_summary(demo_sdata)
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_copy_transform_does_not_raise(demo_sdata):
    """copy_transform should run without error on a source image and a new labels element."""
    mask = np.zeros((64, 64), dtype=np.int32)
    element = wrap_to_multiscale_labels(mask, demo_sdata)
    src = demo_sdata.images["image"]
    copy_transform(src, element)  # should not raise
