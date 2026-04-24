"""
napari-spatial-data
===================

Interactive semantic annotation of SpatialData images using napari.

Public API
----------
Annotation sessions:
    run_annotation_session
    run_refinement_session

Viewer helpers:
    open_in_napari
    add_labels_layer
    add_shapes_layer

Export / conversion:
    labels_layer_to_sdata
    shapes_layer_to_sdata
    rasterize_shapes_to_labels
    labels_to_shapes
    load_labels_for_refinement

Measurement:
    measure_label_morphology

Utilities:
    get_scale0_xarray
    get_scale0_shape
    get_channel_names
    copy_transform
    wrap_to_multiscale_labels
    class_id_colormap
    print_sdata_summary
"""

from __future__ import annotations

__version__ = "v0.2.0"

from napari_spatial_data.annotate_sdata import (
    open_in_napari,
    add_labels_layer,
    add_shapes_layer,
    labels_layer_to_sdata,
    shapes_layer_to_sdata,
    rasterize_shapes_to_labels,
    labels_to_shapes,
    measure_label_morphology,
    load_labels_for_refinement,
    run_annotation_session,
    run_refinement_session,
)
from napari_spatial_data.sdata_utils import (
    get_scale0_xarray,
    get_scale0_shape,
    get_channel_names,
    copy_transform,
    wrap_to_multiscale_labels,
    class_id_colormap,
    print_sdata_summary,
)

__all__ = [
    "__version__",
    # Annotation sessions
    "run_annotation_session",
    "run_refinement_session",
    # Viewer helpers
    "open_in_napari",
    "add_labels_layer",
    "add_shapes_layer",
    # Export / conversion
    "labels_layer_to_sdata",
    "shapes_layer_to_sdata",
    "rasterize_shapes_to_labels",
    "labels_to_shapes",
    "load_labels_for_refinement",
    # Measurement
    "measure_label_morphology",
    # Utilities
    "get_scale0_xarray",
    "get_scale0_shape",
    "get_channel_names",
    "copy_transform",
    "wrap_to_multiscale_labels",
    "class_id_colormap",
    "print_sdata_summary",
]
