"""
Tests for annotate_sdata.py — the label/shape conversion, morphology
measurement and session-orchestration logic.

Viewers come from napari's own ``make_napari_viewer`` fixture (shipped as a
``pytest11`` entry point on ``napari.utils._testsupport``, so no conftest
wiring is needed) rather than a hand-rolled ``napari.Viewer(show=False)``: it
handles Qt app setup/teardown and fails tests that leak viewers or widgets.
See CONTRIBUTING.md, "Use napari's own viewer fixture".

The session functions (``run_annotation_session`` /
``run_refinement_session``) export their results from a callback wired to the
Qt window's ``destroyed`` signal. They are tested through that real signal —
only the blocking event loop is stubbed out — because the wiring is the part
that breaks. ``qtbot.waitUntil`` pumps the event queue until the export lands.
"""
import numpy as np
import pytest
import spatialdata
from spatialdata import join_spatialelement_table
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, get_model
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import get_transformation

from LayerForge import annotate_sdata as mod
from LayerForge.annotate_sdata import (
    _flush_flattened_mask,
    _run_napari_event_loop,
    _scale0_label_array,
    add_labels_layer,
    add_shapes_layer,
    labels_layer_to_sdata,
    labels_to_shapes,
    load_labels_for_refinement,
    measure_label_morphology,
    open_in_napari,
    rasterize_shapes_to_labels,
    run_annotation_session,
    run_refinement_session,
    shapes_layer_to_sdata,
)
from LayerForge.sdata_utils import wrap_to_multiscale_labels

# Mirrors the _CLASS_COLORS table inside add_shapes_layer.
CLASS_1_HEX = "#e6194b"
CLASS_2_HEX = "#3cb44b"
BACKGROUND_HEX = "#ffffff"  # index 0 — what class 14 wraps around to


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_sdata():
    """A two-channel multiscale image, the common case."""
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
def single_scale_sdata():
    """
    A plain (non-pyramidal) image element.

    Not an edge case: ``_reader.default_scale_factors`` returns ``None`` for
    images at or below 256px, so anything small opened through the plugin is a
    bare DataArray rather than a multiscale DataTree. Every function that
    resolves an image's scale0 extent has to cope with both.
    """
    rng = np.random.default_rng(3)
    data = rng.integers(0, 255, size=(1, 32, 48), dtype=np.uint8)
    image = Image2DModel.parse(data, dims=["c", "y", "x"], c_coords=["DAPI"])
    return spatialdata.SpatialData(images={"image": image})


@pytest.fixture
def viewer(make_napari_viewer):
    """A napari viewer with cleanup handled by napari's own test support."""
    return make_napari_viewer()


@pytest.fixture
def patched_viewer_constructor(make_napari_viewer, monkeypatch):
    """
    Route ``napari.Viewer(...)`` calls made *inside* LayerForge through
    napari's own fixture.

    ``open_in_napari`` constructs its own viewer, which is the behaviour under
    test — but a bare ``napari.Viewer()`` re-runs npe2 plugin discovery and
    dies with ``Command 'LayerForge.get_reader' already registered`` as soon as
    any fixture-made viewer exists in the same session. ``make_napari_viewer``
    passes ``block_plugin_discovery=True`` and registers the viewer for
    teardown, so patching the constructor leaves the real code path intact
    while keeping viewers tracked.
    """
    import napari

    monkeypatch.setattr(napari, "Viewer", lambda **kwargs: make_napari_viewer(**kwargs))


@pytest.fixture
def two_class_labels(demo_sdata):
    """*demo_sdata* with a two-class multiscale Labels element added."""
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:30, 10:30] = 1
    mask[40:60, 40:60] = 2
    demo_sdata.labels["painted"] = wrap_to_multiscale_labels(mask, demo_sdata)
    return demo_sdata, mask


def _class_polygon(y0, x0, y1, x1):
    """A rectangle as napari shape data, i.e. (row, col) vertex pairs."""
    return np.array(
        [[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float
    )


# ---------------------------------------------------------------------------
# open_in_napari
# ---------------------------------------------------------------------------

def test_open_in_napari_adds_one_layer_per_channel(
    demo_sdata, patched_viewer_constructor
):
    """Every channel becomes its own layer, named from the ``c`` coordinate."""
    opened = open_in_napari(demo_sdata, image_key="image")

    assert [layer.name for layer in opened.layers] == ["DAPI", "GFP"]
    for layer in opened.layers:
        assert layer.metadata["sdata"] is demo_sdata
        assert layer.metadata["image_key"] == "image"
    # scale0 pixels survive the multiscale pyramid unpacking
    expected = np.asarray(demo_sdata.images["image"]["scale0"].ds["image"].values)
    np.testing.assert_array_equal(np.asarray(opened.layers[0].data[0]), expected[0])


# ---------------------------------------------------------------------------
# add_labels_layer
# ---------------------------------------------------------------------------

def test_add_labels_layer_shape(demo_sdata, viewer):
    layer = add_labels_layer(viewer, demo_sdata, image_key="image")
    assert layer.data.shape == (64, 64)
    assert layer.data.dtype == np.int32
    assert not layer.data.any()  # starts empty


def test_add_labels_layer_single_scale_image(single_scale_sdata, viewer):
    """
    Regression test: this used to raise KeyError('scale0'). The fallback for a
    bare DataArray caught only (TypeError, AttributeError), but subscripting a
    DataArray with a missing key raises KeyError — so the branch was dead code
    and every non-pyramidal image crashed here.
    """
    layer = add_labels_layer(viewer, single_scale_sdata, image_key="image")
    assert layer.data.shape == (32, 48)


# ---------------------------------------------------------------------------
# add_shapes_layer — the docked Qt class selector
# ---------------------------------------------------------------------------

def _panel_widgets(viewer):
    """Return the (spinbox, tag button, swatch, labels) of the class selector."""
    from qtpy.QtWidgets import QLabel, QPushButton, QSpinBox

    panel = viewer.window.dock_widgets["Class selector"]
    spin = panel.findChildren(QSpinBox)[0]
    buttons = panel.findChildren(QPushButton)
    tag_btn = next(b for b in buttons if "Tag last" in b.text())
    swatch = next(b for b in buttons if b is not tag_btn)
    labels = panel.findChildren(QLabel)
    return spin, tag_btn, swatch, labels


def _label_texts(labels):
    return [lab.text() for lab in labels]


def test_add_shapes_layer_creates_layer_and_panel(viewer):
    layer = add_shapes_layer(viewer, layer_name="my_shapes")

    assert layer.name == "my_shapes"
    assert layer._annotator_class_ids == []
    assert "Class selector" in viewer.window.dock_widgets

    _spin, _tag, swatch, labels = _panel_widgets(viewer)
    assert CLASS_1_HEX in swatch.styleSheet()
    assert "Polygons tagged: 0 / drawn: 0" in _label_texts(labels)


def test_add_shapes_layer_class_change_updates_swatch_and_name(viewer):
    add_shapes_layer(viewer, class_labels={1: "artefact", 2: "epidermis"})
    spin, _tag, swatch, labels = _panel_widgets(viewer)

    assert "<i>artefact</i>" in _label_texts(labels)

    spin.setValue(2)
    assert CLASS_2_HEX in swatch.styleSheet()
    assert "<i>epidermis</i>" in _label_texts(labels)

    # a class with no name defined clears the label rather than showing a stale one
    spin.setValue(3)
    assert "<i>epidermis</i>" not in _label_texts(labels)


def test_add_shapes_layer_class_colours_wrap_around(viewer):
    """There are 14 colours; class 14 wraps back to index 0."""
    add_shapes_layer(viewer)
    spin, _tag, swatch, _labels = _panel_widgets(viewer)

    spin.setValue(14)
    assert BACKGROUND_HEX in swatch.styleSheet()


def test_tag_last_with_no_polygons_reports_instead_of_raising(viewer):
    layer = add_shapes_layer(viewer)
    _spin, tag_btn, _swatch, labels = _panel_widgets(viewer)

    tag_btn.click()

    assert "No polygons drawn yet." in _label_texts(labels)
    assert layer._annotator_class_ids == []


def test_tag_last_assigns_current_class_to_newest_polygon(viewer):
    layer = add_shapes_layer(viewer, class_labels={2: "epidermis"})
    spin, tag_btn, _swatch, labels = _panel_widgets(viewer)

    layer.add(_class_polygon(0, 0, 10, 10), shape_type="polygon")
    spin.setValue(2)
    tag_btn.click()

    assert layer._annotator_class_ids == [2]
    assert layer.features["class_id"].tolist() == [2]
    assert "Polygons tagged: 1 / drawn: 1" in _label_texts(labels)
    assert "Tagged #1 → class 2 epidermis" in _label_texts(labels)


def test_tag_last_backfills_untagged_earlier_polygons(viewer):
    """
    Polygons drawn before the button is first pressed are back-filled with
    class 1 so the class_id list stays aligned with the shape list.
    """
    layer = add_shapes_layer(viewer)
    spin, tag_btn, _swatch, _labels = _panel_widgets(viewer)

    layer.add(_class_polygon(0, 0, 10, 10), shape_type="polygon")
    layer.add(_class_polygon(20, 20, 30, 30), shape_type="polygon")
    spin.setValue(3)
    tag_btn.click()

    assert layer._annotator_class_ids == [1, 3]
    assert layer.features["class_id"].tolist() == [1, 3]


def test_tag_last_twice_retags_rather_than_appending(viewer):
    """Pressing T again corrects the last shape instead of adding an entry."""
    layer = add_shapes_layer(viewer)
    spin, tag_btn, _swatch, _labels = _panel_widgets(viewer)

    layer.add(_class_polygon(0, 0, 10, 10), shape_type="polygon")
    spin.setValue(2)
    tag_btn.click()
    spin.setValue(5)
    tag_btn.click()

    assert layer._annotator_class_ids == [5]


def test_tag_last_recolours_shapes_per_class(viewer):
    from napari.utils.colormaps.standardize_color import transform_color

    layer = add_shapes_layer(viewer)
    spin, tag_btn, _swatch, _labels = _panel_widgets(viewer)

    layer.add(_class_polygon(0, 0, 10, 10), shape_type="polygon")
    tag_btn.click()
    layer.add(_class_polygon(20, 20, 30, 30), shape_type="polygon")
    spin.setValue(2)
    tag_btn.click()

    np.testing.assert_allclose(layer.edge_color[0], transform_color(CLASS_1_HEX)[0])
    np.testing.assert_allclose(layer.edge_color[1], transform_color(CLASS_2_HEX)[0])


# ---------------------------------------------------------------------------
# labels_layer_to_sdata
# ---------------------------------------------------------------------------

def test_labels_layer_to_sdata_roundtrip(demo_sdata, viewer, tmp_path):
    """Painted values, the model, and the transform all survive the export."""
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[:32, :] = 1
    mask[32:, :] = 2
    layer = viewer.add_labels(mask, name="test_labels")

    labels_layer_to_sdata(
        layer, demo_sdata, image_key="image", element_name="test_labels_out"
    )

    element = demo_sdata.labels["test_labels_out"]
    assert get_model(element) is Labels2DModel
    np.testing.assert_array_equal(np.asarray(element.values), mask)
    assert get_transformation(element, to_coordinate_system="global") == (
        get_transformation(demo_sdata.images["image"], to_coordinate_system="global")
    )

    # honest test of usability: the element must survive a zarr round-trip
    zarr_path = tmp_path / "labels.zarr"
    demo_sdata.write(zarr_path)
    assert_spatial_data_objects_are_identical(
        demo_sdata, spatialdata.read_zarr(zarr_path)
    )


def test_labels_layer_to_sdata_with_scale_factors_is_multiscale(demo_sdata, viewer):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[:32, :] = 1
    layer = viewer.add_labels(mask, name="test_labels")

    labels_layer_to_sdata(
        layer,
        demo_sdata,
        image_key="image",
        element_name="pyramid",
        scale_factors=[2],
    )

    element = demo_sdata.labels["pyramid"]
    assert set(element.keys()) == {"scale0", "scale1"}
    np.testing.assert_array_equal(
        np.asarray(element["scale0"].ds["image"].values), mask
    )


# ---------------------------------------------------------------------------
# shapes_layer_to_sdata
# ---------------------------------------------------------------------------

def test_shapes_layer_to_sdata_swaps_vertex_order(demo_sdata, viewer):
    """
    napari stores vertices as (row, col) == (y, x); shapely wants (x, y).
    Asserting on the coordinates is the only way to catch a missing swap —
    a square is symmetric enough to hide it, so use a non-square rectangle.
    """
    layer = viewer.add_shapes(
        [_class_polygon(2, 5, 8, 30)], shape_type="polygon", name="s"
    )

    shapes_layer_to_sdata(
        layer, demo_sdata, image_key="image", element_name="shapes_out"
    )

    gdf = demo_sdata.shapes["shapes_out"]
    assert get_model(gdf) is ShapesModel
    minx, miny, maxx, maxy = gdf.geometry.iloc[0].bounds
    assert (minx, maxx) == (5.0, 30.0)  # x came from the napari column axis
    assert (miny, maxy) == (2.0, 8.0)  # y came from the napari row axis


def test_shapes_layer_to_sdata_defaults_all_shapes_to_class_one(demo_sdata, viewer):
    layer = viewer.add_shapes(
        [_class_polygon(0, 0, 10, 10), _class_polygon(20, 20, 30, 30)],
        shape_type="polygon",
        name="s",
    )

    shapes_layer_to_sdata(layer, demo_sdata, image_key="image", element_name="out")

    assert demo_sdata.shapes["out"]["class_id"].tolist() == [1, 1]


def test_shapes_layer_to_sdata_accepts_rectangles(demo_sdata, viewer):
    layer = viewer.add_shapes(
        [_class_polygon(0, 0, 10, 10)], shape_type="rectangle", name="s"
    )

    shapes_layer_to_sdata(
        layer, demo_sdata, image_key="image", element_name="out", class_ids=[7]
    )

    assert demo_sdata.shapes["out"]["class_id"].tolist() == [7]


def test_shapes_layer_to_sdata_rejects_unsupported_shape_type(demo_sdata, viewer):
    layer = viewer.add_shapes(
        [_class_polygon(0, 0, 10, 10)], shape_type="ellipse", name="s"
    )

    with pytest.raises(ValueError, match="is not supported"):
        shapes_layer_to_sdata(layer, demo_sdata, image_key="image")

    assert "shapes_semantic" not in demo_sdata.shapes


def test_shapes_layer_to_sdata_rejects_class_id_length_mismatch(demo_sdata, viewer):
    layer = viewer.add_shapes(
        [_class_polygon(0, 0, 10, 10)], shape_type="polygon", name="s"
    )

    with pytest.raises(ValueError, match=r"len\(class_ids\)=2"):
        shapes_layer_to_sdata(
            layer, demo_sdata, image_key="image", class_ids=[1, 2]
        )


def test_shapes_layer_to_sdata_stores_nothing_when_no_shapes_drawn(
    demo_sdata, viewer, capsys
):
    layer = viewer.add_shapes(name="s")

    shapes_layer_to_sdata(layer, demo_sdata, image_key="image", element_name="out")

    assert "out" not in demo_sdata.shapes
    assert "No shapes were drawn" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# rasterize_shapes_to_labels
# ---------------------------------------------------------------------------

def _add_shapes_element(sdata, gdf, key="test_shapes"):
    ref = get_transformation(sdata.images["image"], to_coordinate_system="global")
    sdata.shapes[key] = ShapesModel.parse(gdf, transformations={"global": ref})
    return key


def test_rasterize_shapes_burns_class_ids_at_the_right_pixels(demo_sdata):
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"class_id": [1], "geometry": [box(10, 20, 30, 40)]}  # x 10-30, y 20-40
    )
    _add_shapes_element(demo_sdata, gdf)

    rasterize_shapes_to_labels(
        demo_sdata, shapes_key="test_shapes", image_key="image", element_name="rast"
    )

    mask = np.asarray(demo_sdata.labels["rast"].values)
    assert mask.shape == (64, 64)
    assert mask[30, 20] == 1  # inside: row 30 (y), col 20 (x)
    assert mask[5, 5] == 0  # outside
    assert mask[20, 35] == 0  # would be painted if x/y were swapped


def test_rasterize_shapes_smaller_polygon_wins_on_overlap(demo_sdata):
    """
    Large shapes are burned first so a small region nested inside a big one
    stays visible — otherwise fine detail is silently swallowed.
    """
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {
            "class_id": [2, 1],  # small one listed first, to prove sorting happens
            "geometry": [box(20, 20, 30, 30), box(0, 0, 60, 60)],
        }
    )
    _add_shapes_element(demo_sdata, gdf)

    rasterize_shapes_to_labels(
        demo_sdata, shapes_key="test_shapes", image_key="image", element_name="rast"
    )

    mask = np.asarray(demo_sdata.labels["rast"].values)
    assert mask[25, 25] == 2  # inner, smaller polygon survives
    assert mask[5, 5] == 1  # outer polygon elsewhere


def test_rasterize_shapes_single_scale_image_and_scale_factors(single_scale_sdata):
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame({"class_id": [1], "geometry": [box(2, 2, 20, 20)]})
    _add_shapes_element(single_scale_sdata, gdf)

    rasterize_shapes_to_labels(
        single_scale_sdata,
        shapes_key="test_shapes",
        image_key="image",
        element_name="rast",
        scale_factors=[2],
    )

    element = single_scale_sdata.labels["rast"]
    assert set(element.keys()) == {"scale0", "scale1"}
    assert element["scale0"].ds["image"].shape == (32, 48)


# ---------------------------------------------------------------------------
# labels_to_shapes
# ---------------------------------------------------------------------------

def test_labels_to_shapes_preserves_classes_and_indexes_by_instance(two_class_labels):
    sdata, _mask = two_class_labels

    labels_to_shapes(
        sdata, labels_key="painted", image_key="image", element_name="vectorized"
    )

    gdf = sdata.shapes["vectorized"]
    assert get_model(gdf) is ShapesModel
    assert sorted(gdf["class_id"].unique()) == [1, 2]
    # spatialdata treats a shapes element's index as its instance key
    assert gdf.index.tolist() == gdf["instance_id"].tolist()
    assert gdf.index.tolist() == [1, 2]


def test_labels_to_shapes_separates_touching_regions_of_different_classes(demo_sdata):
    """
    Two adjacent blocks of different classes must not be merged into one
    instance — the per-class connected-component pass exists for this.
    """
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:30, 10:20] = 1
    mask[10:30, 20:30] = 2  # shares an edge with the class-1 block
    demo_sdata.labels["touching"] = wrap_to_multiscale_labels(mask, demo_sdata)

    labels_to_shapes(
        demo_sdata, labels_key="touching", image_key="image", element_name="vec"
    )

    gdf = demo_sdata.shapes["vec"]
    assert len(gdf) == 2
    assert sorted(gdf["class_id"].tolist()) == [1, 2]


def test_labels_to_shapes_reads_multiscale_labels(demo_sdata):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:30, 10:30] = 1
    demo_sdata.labels["pyramid"] = wrap_to_multiscale_labels(
        mask, demo_sdata, scale_factors=[2]
    )

    labels_to_shapes(
        demo_sdata, labels_key="pyramid", image_key="image", element_name="vec"
    )

    assert demo_sdata.shapes["vec"]["class_id"].tolist() == [1]


# ---------------------------------------------------------------------------
# measure_label_morphology
# ---------------------------------------------------------------------------

@pytest.fixture
def measured(two_class_labels):
    """Labels vectorised to shapes and measured into a table."""
    sdata, mask = two_class_labels
    labels_to_shapes(
        sdata, labels_key="painted", image_key="image", element_name="regions"
    )
    measure_label_morphology(
        sdata,
        labels_key="painted",
        shapes_key="regions",
        image_key="image",
        element_name="morphology",
    )
    return sdata, mask


def test_morphology_table_joins_to_its_shapes_element(measured):
    """
    Regression test for two bugs that made the documented join impossible:
    `instance_id` present as both obs index name and column (which made
    spatialdata's internal reset_index raise), and a shapes element left on a
    default RangeIndex so only some rows matched.
    """
    sdata, _mask = measured

    elements, joined = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names="regions",
        table_name="morphology",
        how="inner",
    )

    assert joined.n_obs == 2  # every region matched, not just one
    assert elements["regions"].index.tolist() == [1, 2]


def test_morphology_measures_expected_geometry(measured):
    sdata, _mask = measured
    obs = sdata.tables["morphology"].obs

    assert obs["instance_id"].tolist() == [1, 2]
    assert obs["class_id"].tolist() == [1, 2]
    # each painted block is 20x20 pixels
    np.testing.assert_allclose(obs["area"].to_numpy(), [400.0, 400.0])
    np.testing.assert_allclose(obs["extent"].to_numpy(), [1.0, 1.0])
    assert (obs["region"] == "regions").all()


def test_morphology_has_per_channel_intensity_columns(measured):
    sdata, _mask = measured
    obs = sdata.tables["morphology"].obs

    for channel in ("DAPI", "GFP"):
        for prop in ("mean_intensity", "max_intensity", "min_intensity"):
            assert f"{channel}_{prop}" in obs.columns

    # intensities are read from the real image, not left at zero
    scale0 = np.asarray(sdata.images["image"]["scale0"].ds["image"].values)
    expected = scale0[0][10:30, 10:30].mean()
    np.testing.assert_allclose(obs["DAPI_mean_intensity"].iloc[0], expected)


def test_morphology_sanitises_channel_names_for_column_use(demo_sdata):
    """Channel names from a microscope often carry spaces and slashes."""
    rng = np.random.default_rng(5)
    image = Image2DModel.parse(
        rng.integers(0, 255, size=(1, 64, 64), dtype=np.uint8),
        dims=["c", "y", "x"],
        c_coords=["GFP 488/20"],
        scale_factors=[2],
    )
    sdata = spatialdata.SpatialData(images={"image": image})
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:30, 10:30] = 1
    sdata.labels["painted"] = wrap_to_multiscale_labels(mask, sdata)
    labels_to_shapes(sdata, labels_key="painted", image_key="image", element_name="r")

    measure_label_morphology(
        sdata, labels_key="painted", shapes_key="r", image_key="image",
        element_name="morphology",
    )

    assert "GFP_488_20_mean_intensity" in sdata.tables["morphology"].obs.columns


def test_morphology_table_survives_zarr_roundtrip(measured, tmp_path):
    sdata, _mask = measured
    zarr_path = tmp_path / "measured.zarr"

    sdata.write(zarr_path)
    reloaded = spatialdata.read_zarr(zarr_path)

    assert reloaded.tables["morphology"].n_obs == 2
    assert reloaded.shapes["regions"].index.tolist() == [1, 2]


def test_morphology_reads_multiscale_labels(demo_sdata):
    """The pyramidal Labels element is what a real session produces."""
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:30, 10:30] = 1
    demo_sdata.labels["pyramid"] = wrap_to_multiscale_labels(
        mask, demo_sdata, scale_factors=[2]
    )
    labels_to_shapes(
        demo_sdata, labels_key="pyramid", image_key="image", element_name="r"
    )

    measure_label_morphology(
        demo_sdata,
        labels_key="pyramid",
        shapes_key="r",
        image_key="image",
        element_name="morphology",
    )

    obs = demo_sdata.tables["morphology"].obs
    assert obs["instance_id"].tolist() == [1]
    np.testing.assert_allclose(obs["area"].to_numpy(), [400.0])


def test_morphology_reads_single_scale_labels(single_scale_sdata):
    mask = np.zeros((32, 48), dtype=np.int32)
    mask[5:15, 5:15] = 1
    single_scale_sdata.labels["painted"] = Labels2DModel.parse(
        mask,
        dims=["y", "x"],
        transformations={
            "global": get_transformation(
                single_scale_sdata.images["image"], to_coordinate_system="global"
            )
        },
    )
    labels_to_shapes(
        single_scale_sdata, labels_key="painted", image_key="image", element_name="r"
    )

    measure_label_morphology(
        single_scale_sdata,
        labels_key="painted",
        shapes_key="r",
        image_key="image",
        element_name="morphology",
    )

    assert single_scale_sdata.tables["morphology"].n_obs == 1


# ---------------------------------------------------------------------------
# _scale0_label_array
# ---------------------------------------------------------------------------

def test_scale0_label_array_reads_multiscale(demo_sdata):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:30, 10:30] = 4
    demo_sdata.labels["pyramid"] = wrap_to_multiscale_labels(
        mask, demo_sdata, scale_factors=[2]
    )

    np.testing.assert_array_equal(_scale0_label_array(demo_sdata, "pyramid"), mask)


def test_scale0_label_array_reads_single_scale(demo_sdata):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[0:5, 0:5] = 3
    demo_sdata.labels["flat"] = wrap_to_multiscale_labels(mask, demo_sdata)

    np.testing.assert_array_equal(_scale0_label_array(demo_sdata, "flat"), mask)


# ---------------------------------------------------------------------------
# _flush_flattened_mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".npy"])
def test_flush_flattened_mask_writes_mask_and_zarr(demo_sdata, tmp_path, suffix):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[:32, :] = 1
    output_path = tmp_path / f"mask{suffix}"

    _flush_flattened_mask(mask, demo_sdata, output_path)

    assert output_path.exists()
    if suffix == ".npy":
        roundtripped = np.load(output_path)
    else:
        import tifffile
        roundtripped = tifffile.imread(output_path)
    np.testing.assert_array_equal(roundtripped, mask)

    zarr_path = output_path.with_suffix(".zarr")
    assert zarr_path.exists()


def test_flush_flattened_mask_creates_missing_parent_directory(demo_sdata, tmp_path):
    output_path = tmp_path / "nested" / "deeper" / "mask.npy"

    _flush_flattened_mask(np.zeros((8, 8), dtype=np.int32), demo_sdata, output_path)

    assert output_path.exists()


def test_flush_flattened_mask_rejects_unsupported_suffix(demo_sdata, tmp_path):
    mask = np.zeros((8, 8), dtype=np.int32)
    with pytest.raises(ValueError, match="output_path must end in"):
        _flush_flattened_mask(mask, demo_sdata, tmp_path / "mask.bmp")


# ---------------------------------------------------------------------------
# _run_napari_event_loop
# ---------------------------------------------------------------------------

def test_run_napari_event_loop_starts_the_loop_when_none_is_running(monkeypatch, qtbot):
    import napari

    calls = []
    monkeypatch.setattr(napari, "run", lambda: calls.append("run"))

    _run_napari_event_loop()

    assert calls == ["run"]


def test_run_napari_event_loop_skips_when_ipython_owns_the_loop(monkeypatch, qtbot):
    """Under ``%gui qt`` the loop is already spinning; napari.run() would block."""
    import sys
    import types

    import napari

    calls = []
    monkeypatch.setattr(napari, "run", lambda: calls.append("run"))

    shell = types.SimpleNamespace(active_eventloop="qt")
    fake_ipython = types.ModuleType("IPython")
    fake_ipython.get_ipython = lambda: shell
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)

    _run_napari_event_loop()

    assert calls == []


def test_run_napari_event_loop_runs_when_ipython_has_no_qt_loop(monkeypatch, qtbot):
    import sys
    import types

    import napari

    calls = []
    monkeypatch.setattr(napari, "run", lambda: calls.append("run"))

    fake_ipython = types.ModuleType("IPython")
    fake_ipython.get_ipython = lambda: None
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)

    _run_napari_event_loop()

    assert calls == ["run"]


# ---------------------------------------------------------------------------
# load_labels_for_refinement
# ---------------------------------------------------------------------------

def test_load_labels_for_refinement_adds_editable_layer(demo_sdata, viewer):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 3

    layer = load_labels_for_refinement(viewer, mask, demo_sdata, "image", "refine")

    assert layer.name == "refine"
    assert layer.data.dtype == np.int32  # cast so painting new class IDs works
    np.testing.assert_array_equal(layer.data, mask)


def test_load_labels_for_refinement_rejects_mismatched_mask(demo_sdata, viewer):
    with pytest.raises(ValueError, match="does not match image shape"):
        load_labels_for_refinement(
            viewer, np.zeros((10, 10), dtype=np.int32), demo_sdata
        )


def test_load_labels_for_refinement_single_scale_image(single_scale_sdata, viewer):
    layer = load_labels_for_refinement(
        viewer, np.zeros((32, 48), dtype=np.int32), single_scale_sdata
    )
    assert layer.data.shape == (32, 48)


# ---------------------------------------------------------------------------
# Session orchestration
# ---------------------------------------------------------------------------

@pytest.fixture
def session(monkeypatch, patched_viewer_constructor):
    """
    Run a session without blocking, and hand back the viewer it opened.

    Only ``_run_napari_event_loop`` is stubbed; ``open_in_napari`` is wrapped
    rather than replaced so the real viewer construction still runs and the
    export still fires from the genuine ``destroyed`` signal.
    """
    state: dict = {}
    real_open = mod.open_in_napari

    def _recording_open(sdata, image_key="image"):
        opened = real_open(sdata, image_key=image_key)
        state["viewer"] = opened
        return opened

    monkeypatch.setattr(mod, "open_in_napari", _recording_open)
    monkeypatch.setattr(mod, "_run_napari_event_loop", lambda: None)

    return state


def _close_and_wait(session, qtbot, sdata):
    """
    Close the viewer and pump the Qt event queue until the on-close export has
    run. ``contrast_limits`` is the first thing every callback writes, so it is
    the signal that the callback fired regardless of which mode is under test.
    """
    session["viewer"].close()
    qtbot.waitUntil(lambda: "contrast_limits" in sdata.attrs, timeout=5000)


def test_run_annotation_session_labels_mode_exports_painted_mask(
    demo_sdata, session, qtbot
):
    run_annotation_session(demo_sdata, image_key="image", mode="labels")

    layer = session["viewer"].layers["semantic_labels"]
    layer.data[10:30, 10:30] = 1
    layer.data[40:60, 40:60] = 2

    _close_and_wait(session, qtbot, demo_sdata)

    stored = np.asarray(demo_sdata.labels["labels_semantic"].values)
    np.testing.assert_array_equal(stored, layer.data)
    assert sorted(np.unique(stored).tolist()) == [0, 1, 2]


def test_run_annotation_session_captures_contrast_limits_per_channel(
    demo_sdata, session, qtbot
):
    run_annotation_session(demo_sdata, image_key="image", mode="labels")

    session["viewer"].layers["DAPI"].contrast_limits = [12.0, 200.0]

    _close_and_wait(session, qtbot, demo_sdata)

    limits = demo_sdata.attrs["contrast_limits"]
    assert limits["DAPI"] == [12.0, 200.0]
    assert "GFP" in limits


def test_run_annotation_session_labels_mode_flushes_mask_and_zarr(
    demo_sdata, session, qtbot, tmp_path
):
    output_path = tmp_path / "session_mask.npy"

    run_annotation_session(
        demo_sdata, image_key="image", mode="labels", output_path=output_path
    )
    session["viewer"].layers["semantic_labels"].data[0:10, 0:10] = 1

    _close_and_wait(session, qtbot, demo_sdata)
    qtbot.waitUntil(output_path.exists, timeout=5000)

    np.testing.assert_array_equal(
        np.load(output_path), np.asarray(demo_sdata.labels["labels_semantic"].values)
    )
    assert output_path.with_suffix(".zarr").exists()


def test_run_annotation_session_shapes_mode_exports_and_rasterizes(
    demo_sdata, session, qtbot
):
    run_annotation_session(demo_sdata, image_key="image", mode="shapes")

    layer = session["viewer"].layers["semantic_shapes"]
    layer.add(_class_polygon(10, 10, 30, 30), shape_type="polygon")
    layer._annotator_class_ids.append(2)

    _close_and_wait(session, qtbot, demo_sdata)

    assert demo_sdata.shapes["shapes_semantic"]["class_id"].tolist() == [2]
    mask = np.asarray(demo_sdata.labels["labels_semantic"].values)
    assert mask[20, 20] == 2
    assert mask[0, 0] == 0


def test_run_annotation_session_shapes_mode_backfills_untagged_shapes(
    demo_sdata, session, qtbot
):
    """A polygon drawn but never tagged still gets exported, as class 1."""
    run_annotation_session(demo_sdata, image_key="image", mode="shapes")

    layer = session["viewer"].layers["semantic_shapes"]
    layer.add(_class_polygon(10, 10, 20, 20), shape_type="polygon")
    layer.add(_class_polygon(40, 40, 50, 50), shape_type="polygon")
    layer._annotator_class_ids.append(3)  # only the first was tagged

    _close_and_wait(session, qtbot, demo_sdata)

    assert demo_sdata.shapes["shapes_semantic"]["class_id"].tolist() == [3, 1]


def test_run_annotation_session_shapes_mode_flushes_rasterized_mask(
    demo_sdata, session, qtbot, tmp_path
):
    """In shapes mode the flushed raster comes from the rasterized labels."""
    output_path = tmp_path / "shapes_mask.tif"

    run_annotation_session(
        demo_sdata, image_key="image", mode="shapes", output_path=output_path
    )
    layer = session["viewer"].layers["semantic_shapes"]
    layer.add(_class_polygon(10, 10, 30, 30), shape_type="polygon")
    layer._annotator_class_ids.append(2)

    _close_and_wait(session, qtbot, demo_sdata)
    qtbot.waitUntil(output_path.exists, timeout=5000)

    import tifffile

    written = tifffile.imread(output_path)
    np.testing.assert_array_equal(
        written, np.asarray(demo_sdata.labels["labels_semantic"].values)
    )
    assert written[20, 20] == 2
    assert output_path.with_suffix(".zarr").exists()


def test_run_annotation_session_shapes_mode_with_nothing_drawn_stores_nothing(
    demo_sdata, session, qtbot, tmp_path, capsys
):
    output_path = tmp_path / "unused.tif"

    run_annotation_session(
        demo_sdata, image_key="image", mode="shapes", output_path=output_path
    )

    _close_and_wait(session, qtbot, demo_sdata)

    assert "shapes_semantic" not in demo_sdata.shapes
    assert "labels_semantic" not in demo_sdata.labels
    assert not output_path.exists()
    assert "skipping flattened mask export" in capsys.readouterr().out


def test_run_annotation_session_rejects_unknown_mode(demo_sdata, session):
    with pytest.raises(ValueError, match="mode must be 'labels' or 'shapes'"):
        run_annotation_session(demo_sdata, image_key="image", mode="polygons")


# ---------------------------------------------------------------------------
# run_refinement_session
# ---------------------------------------------------------------------------

def test_run_refinement_session_from_numpy_array(demo_sdata, session, qtbot):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:20, 10:20] = 1

    run_refinement_session(demo_sdata, image_key="image", mask_source=mask)

    layer = session["viewer"].layers["labels_refinement"]
    np.testing.assert_array_equal(layer.data, mask)

    layer.data[30:40, 30:40] = 2  # the correction the session exists for
    session["viewer"].close()
    qtbot.waitUntil(lambda: "labels_refined" in demo_sdata.labels, timeout=5000)

    stored = np.asarray(demo_sdata.labels["labels_refined"].values)
    assert sorted(np.unique(stored).tolist()) == [0, 1, 2]


def test_run_refinement_session_from_npy_path_and_saves_result(
    demo_sdata, session, qtbot, tmp_path
):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[5:15, 5:15] = 1
    mask_path = tmp_path / "existing.npy"
    np.save(mask_path, mask)
    save_path = tmp_path / "refined.npy"

    run_refinement_session(
        demo_sdata,
        image_key="image",
        mask_source=str(mask_path),
        save_npy_path=str(save_path),
    )

    np.testing.assert_array_equal(
        session["viewer"].layers["labels_refinement"].data, mask
    )

    session["viewer"].close()
    qtbot.waitUntil(save_path.exists, timeout=5000)
    np.testing.assert_array_equal(np.load(save_path), mask)


def test_run_refinement_session_from_sdata_labels_key(demo_sdata, session, qtbot):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[20:40, 20:40] = 5
    demo_sdata.labels["previous"] = wrap_to_multiscale_labels(
        mask, demo_sdata, scale_factors=[2]
    )

    run_refinement_session(demo_sdata, image_key="image", mask_source="previous")

    np.testing.assert_array_equal(
        session["viewer"].layers["labels_refinement"].data, mask
    )

    session["viewer"].close()
    qtbot.waitUntil(lambda: "labels_refined" in demo_sdata.labels, timeout=5000)


def test_run_refinement_session_from_single_scale_sdata_labels_key(
    demo_sdata, session, qtbot
):
    """A non-pyramidal Labels element resolves through the fallback branch."""
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[0:10, 0:10] = 4
    demo_sdata.labels["previous"] = wrap_to_multiscale_labels(mask, demo_sdata)

    run_refinement_session(demo_sdata, image_key="image", mask_source="previous")

    np.testing.assert_array_equal(
        session["viewer"].layers["labels_refinement"].data, mask
    )

    session["viewer"].close()
    qtbot.waitUntil(lambda: "labels_refined" in demo_sdata.labels, timeout=5000)


def test_run_refinement_session_rejects_unresolvable_mask_source(demo_sdata, session):
    with pytest.raises(ValueError, match="mask_source must be"):
        run_refinement_session(demo_sdata, image_key="image", mask_source="nope")
