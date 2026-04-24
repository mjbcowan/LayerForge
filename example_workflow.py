"""
End-to-end example: build a toy SpatialData object, annotate it, and verify
the results.

Run this script in an environment with:
    pip install spatialdata napari napari-spatialdata geopandas shapely rasterio

The napari window will open twice (once in labels mode, once in shapes mode).
Close each window to continue.  The final sdata state is printed at the end.
"""

from __future__ import annotations

import numpy as np
import spatialdata
from spatialdata.models import Image2DModel

from reimagined_guacamole import (
    open_in_napari,
    add_labels_layer,
    add_shapes_layer,
    labels_layer_to_sdata,
    shapes_layer_to_sdata,
    rasterize_shapes_to_labels,
    run_annotation_session,
)


# ---------------------------------------------------------------------------
# Step 0: Build a minimal SpatialData object with a multichannel, multiscale
#         image to use as a demo.
# ---------------------------------------------------------------------------

def make_demo_sdata(height: int = 512, width: int = 512, n_channels: int = 4) -> spatialdata.SpatialData:
    """
    Create a toy SpatialData object with a multiscale multichannel image.

    Channel layout (mimics a DAPI / marker panel):
        0: DAPI
        1: Marker-A
        2: Marker-B
        3: Marker-C
    """
    rng = np.random.default_rng(42)
    data = rng.integers(0, 2**16, size=(n_channels, height, width), dtype=np.uint16)

    channel_names = ["DAPI", "Marker-A", "Marker-B", "Marker-C"][:n_channels]

    # Parse with two downsampling levels: full, ½, ¼
    image = Image2DModel.parse(
        data,
        dims=["c", "y", "x"],
        c_coords=channel_names,
        scale_factors=[2, 2],
    )

    sdata = spatialdata.SpatialData(images={"image": image})
    print("Created demo SpatialData:")
    print(sdata)
    return sdata


# ---------------------------------------------------------------------------
# Step 1 (option A): Labels-layer annotation session
# ---------------------------------------------------------------------------

def demo_labels_annotation(sdata: spatialdata.SpatialData) -> None:
    """
    Interactive pixel-painting demo.

    1. Opens the image in napari.
    2. Adds an empty Labels layer.
    3. You paint with integer values (1, 2, 3 …) for each semantic class.
    4. On close, the mask is stored in sdata.labels["labels_semantic"].
    """
    run_annotation_session(
        sdata,
        image_key="image",
        mode="labels",
        labels_element_name="labels_semantic",
        scale_factors=[2, 2],   # produce a multiscale labels element
    )

    print("\nsdata after labels annotation:")
    print(sdata)
    print("Labels element:")
    print(sdata.labels["labels_semantic"])


# ---------------------------------------------------------------------------
# Step 1 (option B): Shapes-layer annotation + rasterization
# ---------------------------------------------------------------------------

def demo_shapes_annotation(sdata: spatialdata.SpatialData) -> None:
    """
    Interactive polygon-drawing demo.

    1. Opens the image in napari.
    2. Adds an empty Shapes layer.
    3. You draw polygons; optionally add a 'class_id' column to layer.features.
    4. On close:
       - Polygons are stored in sdata.shapes["shapes_semantic"].
       - The polygons are rasterized to sdata.labels["labels_from_shapes"].
    """
    run_annotation_session(
        sdata,
        image_key="image",
        mode="shapes",
        labels_element_name="labels_from_shapes",
        shapes_element_name="shapes_semantic",
        scale_factors=[2, 2],
    )

    print("\nsdata after shapes annotation:")
    print(sdata)
    print("Shapes element:")
    print(sdata.shapes["shapes_semantic"])
    if "labels_from_shapes" in sdata.labels:
        print("Rasterized labels element:")
        print(sdata.labels["labels_from_shapes"])


# ---------------------------------------------------------------------------
# Step 2: Programmatic (non-interactive) annotation for unit testing /
#         scripted workflows (no GUI needed)
# ---------------------------------------------------------------------------

def demo_programmatic_annotation(sdata: spatialdata.SpatialData) -> None:
    """
    Create a synthetic multi-class mask without opening napari, store it in
    sdata, and print the result.  Useful for testing the export helpers in CI.
    """
    import napari

    element = sdata.images["image"]
    try:
        scale0 = element["scale0"].ds
        xarr = next(iter(scale0.data_vars.values()))
        h, w = xarr.sizes["y"], xarr.sizes["x"]
    except (TypeError, AttributeError):
        h, w = element.sizes["y"], element.sizes["x"]

    # Synthetic 3-class mask
    mask = np.zeros((h, w), dtype=np.int32)
    mask[:h // 3, :] = 1          # class 1 in top third
    mask[h // 3: 2 * h // 3, :] = 2  # class 2 in middle third
    mask[2 * h // 3:, :] = 3     # class 3 in bottom third

    # Wrap in a headless Labels layer
    viewer = napari.Viewer(show=False)
    labels_layer = viewer.add_labels(mask, name="semantic_labels")

    labels_layer_to_sdata(
        labels_layer,
        sdata,
        image_key="image",
        element_name="labels_programmatic",
        scale_factors=[2, 2],
    )
    viewer.close()

    print("\nsdata after programmatic annotation:")
    print(sdata)
    print("Programmatic labels element:")
    print(sdata.labels["labels_programmatic"])


# ---------------------------------------------------------------------------
# Step 3: Save / reload the annotated SpatialData
# ---------------------------------------------------------------------------

def save_and_reload(sdata: spatialdata.SpatialData, path: str = "annotated.zarr") -> spatialdata.SpatialData:
    """
    Persist the annotated sdata to a Zarr store and reload it.

    Parameters
    ----------
    sdata:
        Annotated SpatialData object.
    path:
        Zarr store path (directory).  Will be created if absent.

    Returns
    -------
    spatialdata.SpatialData
        Reloaded object.
    """
    sdata.write(path, overwrite=True)
    print(f"Saved to {path!r}")

    reloaded = spatialdata.read_zarr(path)
    print(f"Reloaded from {path!r}")
    print(reloaded)
    return reloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sdata = make_demo_sdata()

    # Choose one of:
    # A) Interactive pixel painting
    # demo_labels_annotation(sdata)

    # B) Interactive polygon drawing
    # demo_shapes_annotation(sdata)

    # C) Programmatic (headless, for testing)
    demo_programmatic_annotation(sdata)

    # Optionally save
    # save_and_reload(sdata, "annotated.zarr")
