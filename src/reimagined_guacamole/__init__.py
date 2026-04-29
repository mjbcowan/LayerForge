"""
reimagined_guacamole
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

from reimagined_guacamole._version import __version__

try:
    from reimagined_guacamole.annotate_sdata import (
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
    from reimagined_guacamole.sample_data import (
        load_skin_sample,
        SKIN_CLASS_LABELS,
    )
    from reimagined_guacamole.sdata_utils import (
        get_scale0_xarray,
        get_scale0_shape,
        get_channel_names,
        copy_transform,
        wrap_to_multiscale_labels,
        class_id_colormap,
        print_sdata_summary,
    )
except ModuleNotFoundError as e:
    if "pkg_resources" in str(e):
        raise RuntimeError(
            "spatialdata requires 'setuptools' on Python 3.12+. "
            "Fix: pip install setuptools"
        ) from e
    raise

__all__ = [
    "__version__",
    # Sample data
    "load_skin_sample",
    "SKIN_CLASS_LABELS",
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
