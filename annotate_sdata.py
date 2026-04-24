"""
Multi-class semantic annotation workflow for SpatialData using napari.

Workflow
--------
1. Open a multiscale SpatialData image in napari via napari-spatialdata.
2. Add an editable Labels layer for multi-class painting (or a Shapes layer
   for polygon-based annotation).
3. After annotation, convert the result back into a SpatialData element
   (Labels2DModel or ShapesModel) with the correct coordinate transform.
4. Store the new element in the original SpatialData object.

Usage
-----
    python annotate_sdata.py

Requirements
------------
    pip install spatialdata napari napari-spatialdata geopandas shapely
"""

from __future__ import annotations

import re
import numpy as np
import napari
import geopandas as gpd
from shapely.geometry import Polygon

import spatialdata
from spatialdata.models import Labels2DModel, ShapesModel, TableModel
from spatialdata.transformations import get_transformation, set_transformation


# ---------------------------------------------------------------------------
# 1.  Open the multiscale image in napari
# ---------------------------------------------------------------------------

def open_in_napari(sdata: spatialdata.SpatialData, image_key: str = "image") -> napari.Viewer:
    """
    Open *sdata* in a napari viewer with the chosen multiscale image loaded.

    Channel names are read from the SpatialData element's ``c`` coordinate
    when available, falling back to "ch-0", "ch-1", … otherwise.

    Parameters
    ----------
    sdata:
        A loaded SpatialData object.
    image_key:
        Key of the image element inside ``sdata.images``.

    Returns
    -------
    napari.Viewer
        A running viewer with the image already visible.
    """
    element = sdata.images[image_key]

    # --- resolve channel names -----------------------------------------
    # MultiscaleSpatialImage: grab scale0 to inspect coords
    try:
        scale0 = element["scale0"].ds  # xarray.Dataset at full resolution
        xarr = next(iter(scale0.data_vars.values()))  # single DataArray
    except (TypeError, AttributeError):
        xarr = element  # already a plain DataArray

    if "c" in xarr.coords:
        channel_names = [str(c) for c in xarr.coords["c"].values]
    elif "c" in xarr.dims:
        channel_names = [f"ch-{i}" for i in range(xarr.sizes["c"])]
    else:
        channel_names = None

    # --- add multiscale image ------------------------------------------
    # Always add the image manually so all channels are immediately visible
    # in the napari canvas. napari-spatialdata's Interactive requires the user
    # to manually select elements in its side panel, which hides the image on
    # launch and is confusing for annotation workflows.
    viewer = napari.Viewer(title="SpatialData annotator")
    viewer = _add_image_manually(sdata, image_key, channel_names, viewer)

    return viewer


def _add_image_manually(
    sdata: spatialdata.SpatialData,
    image_key: str,
    channel_names: list[str] | None,
    viewer: napari.Viewer,
) -> napari.Viewer:
    """
    Add the SpatialData image to napari as a multiscale layer.

    Pyramid levels are passed as dask arrays (lazy) so that large images
    (e.g. 13 k × 24 k pixels) do not trigger an eager full load into RAM.
    Each channel is added as a separate Image layer so it can be toggled,
    coloured, and contrast-adjusted independently — essential for multiplexed
    / multichannel MSI data.
    """
    import dask.array as da

    element = sdata.images[image_key]

    try:
        # MultiscaleSpatialImage → collect pyramid levels in order, keeping
        # data as dask arrays (no .values call) to avoid eager loading.
        scale_keys = sorted(element.keys())  # "scale0", "scale1", …
        pyramid = []
        for sk in scale_keys:
            ds = element[sk].ds
            arr = next(iter(ds.data_vars.values()))
            # Convert to dask if not already (keeps loading lazy)
            pyramid.append(da.from_array(arr) if not hasattr(arr.data, "dask") else arr.data)
        n_channels = pyramid[0].shape[0]
        data_is_multiscale = True
    except (TypeError, AttributeError):
        # Plain DataArray — wrap in a single-element list so the loop below
        # works uniformly.
        arr = element
        dask_arr = da.from_array(arr) if not hasattr(arr.data, "dask") else arr.data
        pyramid = [dask_arr]
        n_channels = pyramid[0].shape[0]
        data_is_multiscale = False

    # Colormaps to cycle through for each channel
    _CHANNEL_COLORMAPS = [
        "blue", "green", "red", "cyan", "magenta", "yellow",
        "gray", "bop orange", "bop purple", "bop blue",
    ]

    for ch_idx in range(n_channels):
        # Slice each pyramid level to a single channel → (y, x) per level
        ch_pyramid = [level[ch_idx] for level in pyramid]
        ch_data = ch_pyramid if data_is_multiscale else ch_pyramid[0]

        ch_name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"ch-{ch_idx}"
        colormap = _CHANNEL_COLORMAPS[ch_idx % len(_CHANNEL_COLORMAPS)]

        viewer.add_image(
            ch_data,
            name=ch_name,
            colormap=colormap,
            blending="additive",
            visible=True,
        )

    return viewer


# ---------------------------------------------------------------------------
# 2a.  Add a Labels layer for pixel-level multi-class painting
# ---------------------------------------------------------------------------

def add_labels_layer(
    viewer: napari.Viewer,
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
    layer_name: str = "semantic_labels",
) -> napari.layers.Labels:
    """
    Add an empty integer Labels layer on top of the image.

    The layer canvas matches the *full-resolution* (scale0) spatial extent of
    the image so that painted pixels map 1-to-1 with image pixels at that
    scale.

    Paint with different integer values (1, 2, 3 …) to represent semantic
    classes.  Use the napari Labels layer controls to choose the active label
    value before painting.

    Parameters
    ----------
    viewer:
        An open napari Viewer (image already loaded).
    sdata:
        The SpatialData object (used to infer image shape).
    image_key:
        Key of the image inside ``sdata.images``.
    layer_name:
        Name given to the new layer.

    Returns
    -------
    napari.layers.Labels
    """
    element = sdata.images[image_key]

    # Get the (y, x) shape from scale0
    try:
        scale0 = element["scale0"].ds
        xarr = next(iter(scale0.data_vars.values()))
        h = xarr.sizes["y"]
        w = xarr.sizes["x"]
    except (TypeError, AttributeError):
        # Plain DataArray
        h = element.sizes["y"]
        w = element.sizes["x"]

    mask = np.zeros((h, w), dtype=np.int32)
    labels_layer = viewer.add_labels(mask, name=layer_name)
    return labels_layer


# ---------------------------------------------------------------------------
# 2b.  Add a Shapes layer for polygon-based annotation
# ---------------------------------------------------------------------------

def add_shapes_layer(
    viewer: napari.Viewer,
    layer_name: str = "semantic_shapes",
    class_labels: dict[int, str] | None = None,
) -> napari.layers.Shapes:
    """
    Add an empty Shapes layer for polygon annotation with a live class-tagging
    widget docked in the napari viewer.

    A spinbox in the viewer controls the *current class ID*.  Every polygon
    finished while that class is active is immediately tagged in
    ``layer.features["class_id"]`` and coloured by class so you can see which
    region belongs to which class without closing the viewer.

    Parameters
    ----------
    viewer:
        An open napari Viewer.
    layer_name:
        Name given to the Shapes layer.
    class_labels:
        Optional mapping of class_id → human-readable name, e.g.
        ``{1: "artefact", 2: "epidermis", ...}``.  Used for display only.

    Returns
    -------
    napari.layers.Shapes
    """
    from qtpy.QtWidgets import (  # type: ignore
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton,
    )
    from qtpy.QtCore import Qt  # type: ignore

    # Per-class edge colours so polygons are visually distinct while drawing
    _CLASS_COLORS = [
        "#ffffff",  # 0 – background (unused)
        "#e6194b",  # 1
        "#3cb44b",  # 2
        "#4363d8",  # 3
        "#f58231",  # 4
        "#911eb4",  # 5
        "#42d4f4",  # 6
        "#f032e6",  # 7
        "#bfef45",  # 8
        "#fabed4",  # 9
        "#469990",  # 10
        "#dcbeff",  # 11
        "#9a6324",  # 12
        "#aaffc3",  # 13
    ]

    def _color_for(cls_id: int) -> str:
        return _CLASS_COLORS[cls_id % len(_CLASS_COLORS)]

    shapes_layer = viewer.add_shapes(
        name=layer_name,
        face_color="transparent",
        edge_color=_color_for(1),
        edge_width=2,
    )

    # Internal list that grows as polygons are tagged; index = shape index.
    _class_ids: list[int] = []

    # ------------------------------------------------------------------ #
    # Qt widget: class selector docked in the viewer                      #
    # ------------------------------------------------------------------ #
    container = QWidget()
    layout = QVBoxLayout()
    container.setLayout(layout)

    # Title
    title = QLabel("<b>Semantic class selector</b>")
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)

    # Class spinbox row
    row = QHBoxLayout()
    row.addWidget(QLabel("Class ID:"))
    spinbox = QSpinBox()
    spinbox.setMinimum(1)
    spinbox.setMaximum(99)
    spinbox.setValue(1)
    spinbox.setFixedWidth(60)
    row.addWidget(spinbox)
    layout.addLayout(row)

    # Label showing the class name (if provided)
    name_label = QLabel()
    name_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(name_label)

    def _update_name_label(cls_id: int) -> None:
        if class_labels and cls_id in class_labels:
            name_label.setText(f"<i>{class_labels[cls_id]}</i>")
        else:
            name_label.setText("")

    _update_name_label(1)

    # Colour swatch (visual feedback only)
    swatch = QPushButton()
    swatch.setFixedHeight(20)
    swatch.setEnabled(False)
    swatch.setStyleSheet(f"background-color: {_color_for(1)};")
    layout.addWidget(swatch)

    def _on_class_changed(cls_id: int) -> None:
        color = _color_for(cls_id)
        swatch.setStyleSheet(f"background-color: {color};")
        _update_name_label(cls_id)

    spinbox.valueChanged.connect(_on_class_changed)

    layout.addSpacing(8)

    # "Tag last polygon" button — pressed after finishing each polygon
    tag_btn = QPushButton("Tag last polygon  [T]")
    tag_btn.setToolTip(
        "Press after finishing a polygon to assign the current class ID to it."
    )
    layout.addWidget(tag_btn)

    status_label = QLabel("Polygons tagged: 0 / drawn: 0")
    status_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(status_label)

    log_label = QLabel("")
    log_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(log_label)

    layout.addStretch()

    def _tag_last() -> None:
        """Assign the current class ID to the most-recently drawn polygon."""
        n_drawn = len(shapes_layer.data)
        if n_drawn == 0:
            log_label.setText("No polygons drawn yet.")
            return

        # Sync _class_ids length with however many shapes exist (handles
        # polygons drawn before the button was first pressed).
        while len(_class_ids) < n_drawn - 1:
            _class_ids.append(1)  # back-fill untagged shapes with class 1

        cls_id = spinbox.value()

        if len(_class_ids) < n_drawn:
            # New shape — append
            _class_ids.append(cls_id)
        else:
            # Re-tag the last shape (user pressed T twice to correct)
            _class_ids[-1] = cls_id

        # Write features and recolour — safe to do here because we are NOT
        # inside a shapes data event callback.
        import pandas as pd
        shapes_layer.features = pd.DataFrame({"class_id": _class_ids})
        shapes_layer.edge_color = [_color_for(c) for c in _class_ids]

        n_tagged = len(_class_ids)
        cls_name = class_labels.get(cls_id, "") if class_labels else ""
        log_label.setText(f"Tagged #{n_tagged} → class {cls_id} {cls_name}")
        status_label.setText(f"Polygons tagged: {n_tagged} / drawn: {n_drawn}")

    tag_btn.clicked.connect(_tag_last)

    # Keyboard shortcut: T
    from qtpy.QtWidgets import QShortcut  # type: ignore
    from qtpy.QtGui import QKeySequence   # type: ignore
    shortcut = QShortcut(QKeySequence("T"), viewer.window._qt_window)
    shortcut.activated.connect(_tag_last)

    viewer.window.add_dock_widget(container, name="Class selector", area="right")

    # Expose _class_ids so run_annotation_session can read it on close
    shapes_layer._annotator_class_ids = _class_ids

    return shapes_layer


# ---------------------------------------------------------------------------
# 3a.  Export a Labels layer → SpatialData Labels2DModel element
# ---------------------------------------------------------------------------

def labels_layer_to_sdata(
    labels_layer: napari.layers.Labels,
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
    element_name: str = "labels_semantic",
    scale_factors: list[int] | None = None,
) -> None:
    """
    Convert a napari Labels layer into a SpatialData Labels element and store
    it in *sdata*.

    The coordinate transform is copied from the reference image so that the
    new element lives in the same coordinate system.

    Parameters
    ----------
    labels_layer:
        The annotated napari Labels layer.
    sdata:
        Target SpatialData object.
    image_key:
        Key of the reference image (used to copy the coordinate transform).
    element_name:
        Key under which the Labels element is stored in ``sdata.labels``.
    scale_factors:
        If provided (e.g. ``[2, 2]``), a multiscale Labels element is created.
        Omit for a single-scale element.
    """
    mask: np.ndarray = labels_layer.data  # (y, x) int array

    # Retrieve the transform from the reference image
    ref_transform = get_transformation(
        sdata.images[image_key], to_coordinate_system="global"
    )

    parse_kwargs: dict = dict(
        data=mask,
        dims=["y", "x"],
        transformations={"global": ref_transform},
    )
    if scale_factors:
        parse_kwargs["scale_factors"] = scale_factors

    labels_element = Labels2DModel.parse(**parse_kwargs)
    sdata.labels[element_name] = labels_element

    print(
        f"Stored '{element_name}' in sdata.labels  "
        f"(shape={mask.shape}, unique classes={np.unique(mask).tolist()})"
    )


# ---------------------------------------------------------------------------
# 3b.  Export a Shapes layer → SpatialData ShapesModel element
# ---------------------------------------------------------------------------

def shapes_layer_to_sdata(
    shapes_layer: napari.layers.Shapes,
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
    element_name: str = "shapes_semantic",
    class_ids: list[int] | None = None,
) -> None:
    """
    Convert a napari Shapes layer into a SpatialData Shapes element and store
    it in *sdata*.

    Polygon vertices from napari are in (row, col) == (y, x) order.
    ``class_id`` is stored as a column in the resulting GeoDataFrame.

    Parameters
    ----------
    shapes_layer:
        The annotated napari Shapes layer.
    sdata:
        Target SpatialData object.
    image_key:
        Key of the reference image (used to copy the coordinate transform).
    element_name:
        Key under which the Shapes element is stored in ``sdata.shapes``.
    class_ids:
        Integer class label for each shape (same length as
        ``shapes_layer.data``).  Defaults to all-ones (single class).
    """
    polygons = []
    for shape_data, shape_type in zip(shapes_layer.data, shapes_layer.shape_type):
        if shape_type in ("polygon", "rectangle"):
            # napari returns (row, col) == (y, x); shapely expects (x, y)
            xy = shape_data[:, ::-1]  # swap columns
            polygons.append(Polygon(xy))
        else:
            raise ValueError(
                f"Shape type '{shape_type}' is not supported. "
                "Use polygons or rectangles."
            )

    if class_ids is None:
        class_ids = [1] * len(polygons)

    if len(class_ids) != len(polygons):
        raise ValueError(
            f"len(class_ids)={len(class_ids)} != len(polygons)={len(polygons)}"
        )

    if not polygons:
        print("No shapes were drawn — skipping storage.")
        return

    gdf = gpd.GeoDataFrame(
        {"class_id": class_ids, "geometry": polygons},
        geometry="geometry",
    )

    ref_transform = get_transformation(
        sdata.images[image_key], to_coordinate_system="global"
    )

    shapes_element = ShapesModel.parse(
        gdf,
        transformations={"global": ref_transform},
    )
    sdata.shapes[element_name] = shapes_element

    print(
        f"Stored '{element_name}' in sdata.shapes  "
        f"(n_shapes={len(gdf)}, classes={sorted(set(class_ids))})"
    )


# ---------------------------------------------------------------------------
# 4.  Rasterize Shapes → Labels (optional post-processing step)
# ---------------------------------------------------------------------------

def rasterize_shapes_to_labels(
    sdata: spatialdata.SpatialData,
    shapes_key: str = "shapes_semantic",
    image_key: str = "image",
    element_name: str = "labels_from_shapes",
    class_column: str = "class_id",
    scale_factors: list[int] | None = None,
) -> None:
    """
    Burn a Shapes element into a raster Labels element at the same resolution
    as the reference image, preserving the ``class_id`` values.

    Uses ``spatialdata.transform`` + manual rasterization via rasterio/numpy
    because ``spatialdata.rasterize`` burns *instance* IDs by default.

    Parameters
    ----------
    sdata:
        SpatialData object containing both the image and shapes element.
    shapes_key:
        Key of the Shapes element in ``sdata.shapes``.
    image_key:
        Key of the reference image (determines output raster size).
    element_name:
        Key for the new Labels element stored in ``sdata.labels``.
    class_column:
        Column in the Shapes GeoDataFrame that holds integer class IDs.
    scale_factors:
        Optional multiscale factors for the output Labels element.
    """
    from rasterio.features import rasterize as rio_rasterize  # type: ignore
    from affine import Affine  # type: ignore

    gdf: gpd.GeoDataFrame = sdata.shapes[shapes_key]

    # Determine image size at scale0
    element = sdata.images[image_key]
    try:
        scale0 = element["scale0"].ds
        xarr = next(iter(scale0.data_vars.values()))
        h, w = xarr.sizes["y"], xarr.sizes["x"]
    except (TypeError, AttributeError):
        h, w = element.sizes["y"], element.sizes["x"]

    # Polygon vertices are stored in napari pixel-index space:
    # napari uses (row, col) = (y, x) with origin at top-left and step = 1.
    # shapes_layer_to_sdata swaps napari's (row, col) → shapely (x=col, y=row).
    # rasterio burns shapes using their (x, y) coords against this affine.
    #
    # Affine.translation(0, 0) * Affine.scale(1, 1) is the identity:
    #   col  = x * 1 + 0  → maps shapely x directly to pixel column
    #   row  = y * 1 + 0  → maps shapely y directly to pixel row
    # This keeps y increasing downward (row 0 = top), matching array layout.
    affine = Affine.translation(0, 0) * Affine.scale(1, 1)

    # Sort largest polygons first so smaller, finer regions are burned last
    # and therefore win when polygons overlap (last-write wins in rasterio).
    gdf_sorted = gdf.copy()
    gdf_sorted["_area"] = gdf_sorted.geometry.area
    gdf_sorted = gdf_sorted.sort_values("_area", ascending=False)

    shapes_iter = [
        (geom, int(cls_id))
        for geom, cls_id in zip(gdf_sorted.geometry, gdf_sorted[class_column])
    ]

    mask = rio_rasterize(
        shapes_iter,
        out_shape=(h, w),
        transform=affine,
        fill=0,
        dtype=np.int32,
    )

    ref_transform = get_transformation(
        sdata.images[image_key], to_coordinate_system="global"
    )

    parse_kwargs: dict = dict(
        data=mask,
        dims=["y", "x"],
        transformations={"global": ref_transform},
    )
    if scale_factors:
        parse_kwargs["scale_factors"] = scale_factors

    labels_element = Labels2DModel.parse(**parse_kwargs)
    sdata.labels[element_name] = labels_element

    print(
        f"Rasterized '{shapes_key}' → '{element_name}'  "
        f"(shape={mask.shape}, unique classes={np.unique(mask).tolist()})"
    )



def labels_to_shapes(
    sdata: spatialdata.SpatialData,
    labels_key: str = "labels_from_shapes",
    image_key: str = "image",
    element_name: str = "shapes_from_labels",
) -> None:
    """
    Vectorize a Labels element back into a Shapes element.

    Each connected region in the label mask becomes one polygon.  The original
    semantic ``class_id`` is preserved as a column in the resulting GeoDataFrame.
    Background pixels (value 0) are ignored.

    Parameters
    ----------
    sdata:
        SpatialData object containing the labels element.
    labels_key:
        Key of the Labels element in ``sdata.labels``.
    image_key:
        Reference image whose coordinate transform is copied to the new shapes.
    element_name:
        Key for the new Shapes element stored in ``sdata.shapes``.
    """
    from rasterio.features import shapes as rio_shapes  # type: ignore
    from affine import Affine  # type: ignore
    from shapely.geometry import shape as shapely_shape
    from skimage.measure import label as skimage_label  # type: ignore

    # --- resolve label mask (multiscale or single-scale) --------------------
    element = sdata.labels[labels_key]
    try:
        scale0_node = element["scale0"]
        ds = scale0_node.ds
        xarr = next(iter(ds.data_vars.values()))
        mask = np.asarray(xarr.values)
    except (TypeError, AttributeError, KeyError):
        mask = np.asarray(element.values)

    # --- assign unique instance IDs per connected region --------------------
    # Label each class independently so that adjacent regions of different
    # classes are never merged into the same instance.
    instance_mask = np.zeros_like(mask, dtype=np.int32)
    instance_to_class: dict[int, int] = {}
    next_id = 1
    for class_val in np.unique(mask):
        if class_val == 0:
            continue
        class_instances = skimage_label(mask == class_val, connectivity=2)
        for local_id in np.unique(class_instances):
            if local_id == 0:
                continue
            instance_mask[class_instances == local_id] = next_id
            instance_to_class[next_id] = int(class_val)
            next_id += 1

    # --- vectorize with rasterio --------------------------------------------
    affine = Affine.translation(0, 0) * Affine.scale(1, 1)
    records = []
    for geom_dict, value in rio_shapes(instance_mask, mask=(instance_mask > 0), transform=affine):
        iid = int(value)
        records.append({
            "geometry": shapely_shape(geom_dict),
            "instance_id": iid,
            "class_id": instance_to_class.get(iid, 0),
        })

    gdf = gpd.GeoDataFrame(records, crs=None)

    ref_transform = get_transformation(
        sdata.images[image_key], to_coordinate_system="global"
    )
    shapes_element = ShapesModel.parse(gdf, transformations={"global": ref_transform})
    sdata.shapes[element_name] = shapes_element

    print(
        f"Vectorized '{labels_key}' → '{element_name}'  "
        f"(regions={len(gdf)}, classes={sorted(gdf['class_id'].unique().tolist())})"
    )


def measure_label_morphology(
    sdata: spatialdata.SpatialData,
    labels_key: str = "labels_from_shapes",
    shapes_key: str = "shapes_from_labels",
    image_key: str = "image",
    element_name: str = "morphology",
) -> None:
    """
    Measure scikit-image regionprops features for every connected region in a
    Labels element and store the result as a spatialdata Table.

    Shape properties
    ~~~~~~~~~~~~~~~~
    area, perimeter, eccentricity, solidity, extent, orientation,
    axis_major_length, axis_minor_length, centroid-0/1, bbox-0/1/2/3

    Intensity properties (per image channel)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    <channel>_mean_intensity, <channel>_max_intensity, <channel>_min_intensity

    The table is linked to *shapes_key* so that it can be joined spatially
    using spatialdata's native query API.

    Parameters
    ----------
    sdata:
        SpatialData object.
    labels_key:
        Key of the Labels element used to derive instance masks.
    shapes_key:
        Key of the Shapes element the table will be linked to (must already
        exist, e.g. created by :func:`labels_to_shapes`).
    image_key:
        Key of the reference image for per-channel intensity measurements.
    element_name:
        Key for the new Table element stored in ``sdata.tables``.
    """
    import anndata as ad  # type: ignore
    import pandas as pd
    from skimage.measure import label as skimage_label, regionprops_table  # type: ignore
    from sdata_utils import get_scale0_xarray, get_channel_names

    # --- resolve label mask -------------------------------------------------
    element = sdata.labels[labels_key]
    try:
        scale0_node = element["scale0"]
        ds = scale0_node.ds
        xarr_labels = next(iter(ds.data_vars.values()))
        mask = np.asarray(xarr_labels.values)
    except (TypeError, AttributeError, KeyError):
        mask = np.asarray(element.values)

    instance_mask = np.zeros_like(mask, dtype=np.int32)
    instance_to_class: dict[int, int] = {}
    next_id = 1
    for class_val in np.unique(mask):
        if class_val == 0:
            continue
        class_instances = skimage_label(mask == class_val, connectivity=2)
        for local_id in np.unique(class_instances):
            if local_id == 0:
                continue
            instance_mask[class_instances == local_id] = next_id
            instance_to_class[next_id] = int(class_val)
            next_id += 1

    # --- shape morphology props ---------------------------------------------
    shape_props = [
        "label", "area", "perimeter", "eccentricity", "solidity",
        "extent", "orientation", "axis_major_length", "axis_minor_length",
        "centroid", "bbox",
    ]
    df = pd.DataFrame(regionprops_table(instance_mask, properties=shape_props))
    df.rename(columns={"label": "instance_id"}, inplace=True)

    # --- per-channel intensity props ----------------------------------------
    xarr_image = get_scale0_xarray(sdata, image_key)
    channel_names = get_channel_names(sdata, image_key)
    intensity_props = ["mean_intensity", "max_intensity", "min_intensity"]

    for i, ch_name in enumerate(channel_names):
        safe_ch_name = re.sub(r"[^A-Za-z0-9_.\-]", "_", ch_name)
        ch_array = np.asarray(xarr_image.values[i])  # (y, x)
        ch_df = pd.DataFrame(
            regionprops_table(instance_mask, intensity_image=ch_array, properties=["label"] + intensity_props)
        )
        ch_df.rename(
            columns={p: f"{safe_ch_name}_{p}" for p in intensity_props},
            inplace=True,
        )
        ch_df.drop(columns=["label"], inplace=True)
        df = pd.concat([df, ch_df], axis=1)

    # --- add class_id and region columns ------------------------------------
    df["class_id"] = df["instance_id"].map(instance_to_class)
    df["region"] = shapes_key

    # --- build AnnData and store as spatialdata Table -----------------------
    obs = df.set_index("instance_id").copy()
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(obs=obs)
    adata.obs["instance_id"] = df["instance_id"].values
    adata.obs["region"] = shapes_key

    table = TableModel.parse(
        adata,
        region=shapes_key,
        region_key="region",
        instance_key="instance_id",
    )
    sdata.tables[element_name] = table

    print(
        f"Measured morphology for '{labels_key}' → table '{element_name}'  "
        f"(n_instances={len(df)}, n_features={len(df.columns)})"
    )


# ---------------------------------------------------------------------------
# 5.  High-level entry point: full interactive annotation session
# ---------------------------------------------------------------------------

def _run_napari_event_loop() -> None:
    """
    Start the napari/Qt event loop only when one is not already running.

    When a notebook cell contains ``%gui qt``, IPython installs a Qt event
    loop integration that keeps the loop spinning continuously.  In that
    context ``napari.run()`` would block forever (or raise) because the loop
    is already active.  We detect this and skip the call so the viewer
    stays open non-blocking while the kernel remains responsive.
    """
    try:
        from qtpy.QtWidgets import QApplication  # type: ignore
        app = QApplication.instance()
        if app is not None and app.thread() is not None:
            # Check whether an IPython Qt integration is already running
            try:
                import IPython
                ip = IPython.get_ipython()
                if ip is not None and getattr(ip, "active_eventloop", None) == "qt":
                    # Event loop already managed by %gui qt — do not block
                    return
            except Exception:
                pass
    except Exception:
        pass

    napari.run()


def run_annotation_session(
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
    mode: str = "labels",
    labels_element_name: str = "labels_semantic",
    shapes_element_name: str = "shapes_semantic",
    scale_factors: list[int] | None = None,
    class_labels: dict[int, str] | None = None,
) -> None:
    """
    Launch an interactive napari annotation session and export the result back
    into *sdata*.

    After the napari window is closed the annotation is automatically exported.
    The contrast limits chosen in napari are captured on close and stored as
    ``sdata.attrs["contrast_limits"]`` — a dict keyed by channel name with
    ``[low, high]`` values in the displayed image's native units (uint16 here).

    Parameters
    ----------
    sdata:
        SpatialData object with at least one image element.
    image_key:
        Image to display.
    mode:
        ``"labels"`` – pixel-level painting (integer mask).
        ``"shapes"`` – polygon drawing.
    labels_element_name:
        Output key for the Labels element (used when ``mode="labels"`` or
        after rasterizing shapes).
    shapes_element_name:
        Output key for the Shapes element (used when ``mode="shapes"``).
    scale_factors:
        Multiscale factors for the output element (e.g. ``[2, 2]``).
    class_labels:
        Optional mapping of class_id → human-readable name shown in the
        class selector widget (shapes mode only), e.g.
        ``{1: "artefact", 2: "epidermis", ...}``.
    """
    viewer = open_in_napari(sdata, image_key=image_key)

    def _capture_contrast_limits() -> dict[str, list[float]]:
        """Read contrast_limits from every Image layer in the viewer."""
        limits = {}
        for lyr in viewer.layers:
            if hasattr(lyr, "contrast_limits"):
                limits[lyr.name] = list(lyr.contrast_limits)
        return limits

    if mode == "labels":
        layer = add_labels_layer(viewer, sdata, image_key=image_key)
        print(
            "\n[napari] Paint with integer labels:\n"
            "  background = 0  (unpainted)\n"
            "  class 1    = 1\n"
            "  class 2    = 2  … etc.\n"
            "Close the viewer when done.\n"
        )

        def _on_viewer_close_labels(event=None):
            sdata.attrs["contrast_limits"] = _capture_contrast_limits()
            labels_layer_to_sdata(
                layer,
                sdata,
                image_key=image_key,
                element_name=labels_element_name,
                scale_factors=scale_factors,
            )

        viewer.window._qt_window.destroyed.connect(_on_viewer_close_labels)
        _run_napari_event_loop()

    elif mode == "shapes":
        layer = add_shapes_layer(viewer, class_labels=class_labels)
        print(
            "\n[napari] Draw polygons/rectangles on the image.\n"
            "After each polygon: press T (or click 'Tag last polygon') to\n"
            "assign the current class ID, then draw the next polygon.\n"
            "Close the viewer when done.\n"
        )

        def _on_viewer_close(event=None):
            sdata.attrs["contrast_limits"] = _capture_contrast_limits()
            # Read class IDs from the annotator list (set via Tag button / T key)
            class_ids = getattr(layer, "_annotator_class_ids", None) or None
            # Back-fill any untagged trailing shapes with class 1
            if class_ids is not None:
                n_drawn = len(layer.data)
                while len(class_ids) < n_drawn:
                    class_ids.append(1)
            shapes_layer_to_sdata(
                layer,
                sdata,
                image_key=image_key,
                element_name=shapes_element_name,
                class_ids=class_ids,
            )
            # Optionally also rasterize to a Labels element
            if shapes_element_name in sdata.shapes:
                rasterize_shapes_to_labels(
                    sdata,
                    shapes_key=shapes_element_name,
                    image_key=image_key,
                    element_name=labels_element_name,
                    scale_factors=scale_factors,
                )

        viewer.window._qt_window.destroyed.connect(_on_viewer_close)
        _run_napari_event_loop()

    else:
        raise ValueError(f"mode must be 'labels' or 'shapes', got '{mode}'")


# ---------------------------------------------------------------------------
# 4.  Load an existing mask for fine-tuning / correction
# ---------------------------------------------------------------------------

def load_labels_for_refinement(
    viewer: napari.Viewer,
    mask: np.ndarray,
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
    layer_name: str = "labels_refinement",
) -> napari.layers.Labels:
    """
    Add an existing integer mask as an editable Labels layer for correction.

    Parameters
    ----------
    viewer:
        An open napari Viewer (image already loaded).
    mask:
        2-D integer numpy array (the mask to correct).
    sdata:
        SpatialData object used to validate the mask shape against the image.
    image_key:
        Key of the reference image inside ``sdata.images``.
    layer_name:
        Name given to the layer.

    Returns
    -------
    napari.layers.Labels
    """
    element = sdata.images[image_key]

    try:
        scale0 = element["scale0"].ds
        xarr = next(iter(scale0.data_vars.values()))
        h = xarr.sizes["y"]
        w = xarr.sizes["x"]
    except (TypeError, AttributeError):
        h = element.sizes["y"]
        w = element.sizes["x"]

    if mask.shape != (h, w):
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape ({h}, {w})"
        )

    labels_layer = viewer.add_labels(mask.astype(np.int32), name=layer_name)
    return labels_layer


def run_refinement_session(
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
    mask_source: "str | np.ndarray | None" = None,
    labels_element_name: str = "labels_refined",
    scale_factors: "list[int] | None" = None,
    layer_name: str = "labels_refinement",
    save_npy_path: "str | None" = None,
) -> None:
    """
    Launch an interactive napari session to correct an existing mask.

    Loads ``mask_source`` as an editable Labels layer on top of the image.
    On viewer close, saves the corrected mask back into ``sdata.labels`` and
    optionally to a ``.npy`` file.

    Parameters
    ----------
    sdata:
        SpatialData object with at least one image element.
    image_key:
        Image to display.
    mask_source:
        The mask to refine.  Accepts three forms:

        * a 2-D ``np.ndarray`` (already loaded)
        * a ``str`` path ending in ``.npy`` (loaded with ``np.load``)
        * a ``str`` key present in ``sdata.labels`` (scale0 extracted)
    labels_element_name:
        Key under which the refined mask is stored in ``sdata.labels``.
    scale_factors:
        Multiscale factors for the output element (e.g. ``[2, 2, 2]``).
    layer_name:
        Name of the napari Labels layer.
    save_npy_path:
        If provided, also saves the corrected mask as a ``.npy`` file on close.
    """
    # 1. Resolve mask_source → numpy array
    if isinstance(mask_source, np.ndarray):
        mask = mask_source
        source_desc = "numpy array"
    elif isinstance(mask_source, str) and mask_source.endswith(".npy"):
        mask = np.load(mask_source)
        source_desc = mask_source
    elif isinstance(mask_source, str) and mask_source in sdata.labels:
        element = sdata.labels[mask_source]
        try:
            mask = np.asarray(element["scale0"].ds["image"].values)
        except (TypeError, AttributeError, KeyError):
            mask = np.asarray(element.values)
        source_desc = f"sdata.labels['{mask_source}']"
    else:
        raise ValueError(
            "mask_source must be a numpy array, a .npy file path, or a key in "
            f"sdata.labels. Got: {mask_source!r}"
        )

    # 2. Open viewer and pre-fill the labels layer
    viewer = open_in_napari(sdata, image_key=image_key)
    layer = load_labels_for_refinement(viewer, mask, sdata, image_key, layer_name)

    print(
        f"\n[napari] Refining existing mask loaded from: {source_desc}\n"
        "Paint corrections: use label 0 to erase, paint new class IDs to fill.\n"
        "Close the viewer when done.\n"
    )

    # 3. On-close callback: save back to sdata (and optionally .npy)
    def _on_close(event=None):
        labels_layer_to_sdata(
            layer,
            sdata,
            image_key=image_key,
            element_name=labels_element_name,
            scale_factors=scale_factors,
        )
        if save_npy_path is not None:
            np.save(save_npy_path, layer.data)
            print(f"Saved refined mask to {save_npy_path!r}")

    viewer.window._qt_window.destroyed.connect(_on_close)
    _run_napari_event_loop()
