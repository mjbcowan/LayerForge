"""
Tests for annotate_sdata — headless only.
All GUI tests use napari.Viewer(show=False) to avoid requiring a display server.
"""
import numpy as np
import pytest
import napari
import spatialdata
from spatialdata.models import Image2DModel

from reimagined_guacamole.annotate_sdata import (
    add_labels_layer,
    labels_layer_to_sdata,
    rasterize_shapes_to_labels,
    labels_to_shapes,
)


@pytest.fixture
def demo_sdata():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 255, size=(2, 64, 64), dtype=np.uint8)
    image = Image2DModel.parse(
        data,
        dims=["c", "y", "x"],
        c_coords=["DAPI", "GFP"],
        scale_factors=[2],
    )
    return spatialdata.SpatialData(images={"image": image})


@pytest.fixture
def headless_viewer():
    """napari Viewer in headless (no-show) mode."""
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


def test_add_labels_layer_shape(demo_sdata, headless_viewer):
    layer = add_labels_layer(headless_viewer, demo_sdata, image_key="image")
    assert layer.data.shape == (64, 64)
    assert layer.data.dtype == np.int32


def test_labels_layer_to_sdata_roundtrip(demo_sdata, headless_viewer):
    """Paint a synthetic mask, export to sdata, verify element is stored."""
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[:32, :] = 1
    mask[32:, :] = 2
    layer = headless_viewer.add_labels(mask, name="test_labels")

    labels_layer_to_sdata(layer, demo_sdata, image_key="image", element_name="test_labels_out")

    assert "test_labels_out" in demo_sdata.labels


def test_rasterize_shapes_creates_labels(demo_sdata):
    """Create a shapes element, rasterize it, check labels element exists."""
    import geopandas as gpd
    from shapely.geometry import box
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import get_transformation

    polygon = box(10, 10, 54, 54)
    gdf = gpd.GeoDataFrame({"class_id": [1], "geometry": [polygon]})
    ref_transform = get_transformation(
        demo_sdata.images["image"], to_coordinate_system="global"
    )
    shapes_element = ShapesModel.parse(
        gdf, transformations={"global": ref_transform}
    )
    demo_sdata.shapes["test_shapes"] = shapes_element

    rasterize_shapes_to_labels(
        demo_sdata,
        shapes_key="test_shapes",
        image_key="image",
        element_name="rasterized",
    )
    assert "rasterized" in demo_sdata.labels


def test_labels_to_shapes_roundtrip(demo_sdata):
    """Create a labels element, vectorize to shapes, check class_ids preserved."""
    from reimagined_guacamole.sdata_utils import wrap_to_multiscale_labels

    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:30, 10:30] = 1
    mask[40:60, 40:60] = 2

    element = wrap_to_multiscale_labels(mask, demo_sdata)
    demo_sdata.labels["test_labels_vec"] = element

    labels_to_shapes(
        demo_sdata,
        labels_key="test_labels_vec",
        image_key="image",
        element_name="vectorized_shapes",
    )
    assert "vectorized_shapes" in demo_sdata.shapes
    gdf = demo_sdata.shapes["vectorized_shapes"]
    assert set(gdf["class_id"].unique()) == {1, 2}
