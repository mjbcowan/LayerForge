"""
Utility helpers for SpatialData annotation workflows.

Functions
---------
get_scale0_shape          – Return (height, width) of the full-resolution image.
get_scale0_xarray         – Return the scale0 DataArray of a SpatialData image.
get_channel_names         – Extract channel names from a SpatialData image element.
copy_transform            – Copy the coordinate transform from one element to another.
wrap_to_multiscale_labels – Wrap a single numpy mask into a multiscale Labels element.
sdata_image_to_layer_data – Convert a SpatialData image element into napari LayerDataTuples.
class_id_colormap         – Build a napari colormap dict for N semantic classes.
print_sdata_summary       – Pretty-print a SpatialData object's element inventory.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import spatialdata
from spatialdata.models import Labels2DModel
from spatialdata.transformations import get_transformation, set_transformation


# ---------------------------------------------------------------------------
# Image introspection helpers
# ---------------------------------------------------------------------------

def get_scale0_xarray(
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
) -> xr.DataArray:
    """
    Return the full-resolution DataArray for a SpatialData image element.

    Works for both plain ``xr.DataArray`` (single-scale) and
    ``MultiscaleSpatialImage`` (multiscale) elements.
    """
    element = sdata.images[image_key]
    try:
        # MultiscaleSpatialImage has a mapping interface: element["scale0"]
        scale0_node = element["scale0"]
        ds = scale0_node.ds  # xarray.Dataset
        return next(iter(ds.data_vars.values()))
    except (TypeError, AttributeError, KeyError):
        # Already a plain DataArray
        return element


def get_scale0_shape(
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
) -> tuple[int, int]:
    """Return ``(height, width)`` at full resolution (scale0)."""
    xarr = get_scale0_xarray(sdata, image_key)
    return xarr.sizes["y"], xarr.sizes["x"]


def get_channel_names(
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
) -> list[str]:
    """
    Return the channel names of a SpatialData image element.

    Falls back to ``["ch-0", "ch-1", …]`` if no ``c`` coordinate is present.
    """
    xarr = get_scale0_xarray(sdata, image_key)
    if "c" in xarr.coords:
        return [str(c) for c in xarr.coords["c"].values]
    if "c" in xarr.dims:
        return [f"ch-{i}" for i in range(xarr.sizes["c"])]
    return []


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def copy_transform(
    source_element,
    target_element,
    coordinate_system: str = "global",
) -> None:
    """
    Copy the coordinate transform from *source_element* to *target_element*.

    Both elements must be valid SpatialData spatial elements (DataArray,
    DataTree, or GeoDataFrame).

    Parameters
    ----------
    source_element:
        Element whose transform is read.
    target_element:
        Element that receives the copied transform.
    coordinate_system:
        The coordinate system to copy (default ``"global"``).
    """
    transform = get_transformation(source_element, to_coordinate_system=coordinate_system)
    set_transformation(target_element, transform, to_coordinate_system=coordinate_system)


# ---------------------------------------------------------------------------
# napari layer conversion
# ---------------------------------------------------------------------------

# Colormaps to cycle through for each channel when adding an image to napari.
_CHANNEL_COLORMAPS = [
    "blue", "green", "red", "cyan", "magenta", "yellow",
    "gray", "bop orange", "bop purple", "bop blue",
]


def sdata_image_to_layer_data(
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
) -> list[tuple]:
    """
    Convert a SpatialData image element into napari ``LayerDataTuple``s.

    Each channel becomes its own ``"image"`` layer tuple ``(data, kwargs,
    "image")`` so channels can be toggled, coloured, and contrast-adjusted
    independently — the shared logic behind both
    :func:`LayerForge.annotate_sdata.open_in_napari` and the npe2 reader
    contribution, so channel handling lives in exactly one place. Pyramid
    levels are kept as dask arrays (lazy) so large images are not eagerly
    loaded into RAM.

    Every returned layer's ``metadata`` carries ``"sdata"`` and
    ``"image_key"`` — the same convention napari-spatialdata uses to attach
    a SpatialData object to a layer — so viewer widgets can recover *sdata*
    without a bespoke file dialog.

    Parameters
    ----------
    sdata:
        A loaded SpatialData object.
    image_key:
        Key of the image element inside ``sdata.images``.

    Returns
    -------
    list[tuple]
        A list of ``(data, add_kwargs, "image")`` tuples, one per channel.
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

    channel_names = get_channel_names(sdata, image_key) or None

    layer_data: list[tuple] = []
    for ch_idx in range(n_channels):
        ch_pyramid = [level[ch_idx] for level in pyramid]
        ch_data = ch_pyramid if data_is_multiscale else ch_pyramid[0]

        ch_name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"ch-{ch_idx}"
        colormap = _CHANNEL_COLORMAPS[ch_idx % len(_CHANNEL_COLORMAPS)]

        kwargs = dict(
            name=ch_name,
            colormap=colormap,
            blending="additive",
            visible=True,
            metadata={"sdata": sdata, "image_key": image_key, "channel_index": ch_idx},
        )
        layer_data.append((ch_data, kwargs, "image"))

    return layer_data


# ---------------------------------------------------------------------------
# Multiscale wrapping
# ---------------------------------------------------------------------------

def wrap_to_multiscale_labels(
    mask: np.ndarray,
    sdata: spatialdata.SpatialData,
    image_key: str = "image",
    scale_factors: list[int] | None = None,
    coordinate_system: str = "global",
) -> object:
    """
    Wrap a 2-D integer numpy *mask* into a SpatialData Labels element.

    If *scale_factors* is provided (e.g. ``[2, 2]``), a multiscale element
    is returned (as a ``datatree.DataTree`` / ``MultiscaleSpatialImage``).
    Otherwise a single-scale ``xr.DataArray`` is returned.

    The coordinate transform is copied from the reference image so that the
    label element is aligned in the same coordinate system.

    Parameters
    ----------
    mask:
        2-D integer numpy array of shape ``(height, width)``.
    sdata:
        SpatialData object containing the reference image.
    image_key:
        Key of the reference image.
    scale_factors:
        Relative downsampling factors between consecutive pyramid levels.
        ``[2, 2]`` → 3 total levels (original, ½, ¼).
    coordinate_system:
        Target coordinate system name.

    Returns
    -------
    xr.DataArray or datatree.DataTree
        A validated SpatialData Labels element ready for insertion into
        ``sdata.labels``.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")

    ref_transform = get_transformation(
        sdata.images[image_key],
        to_coordinate_system=coordinate_system,
    )

    parse_kwargs: dict = dict(
        data=mask,
        dims=["y", "x"],
        transformations={coordinate_system: ref_transform},
    )
    if scale_factors:
        parse_kwargs["scale_factors"] = scale_factors

    return Labels2DModel.parse(**parse_kwargs)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def class_id_colormap(n_classes: int, background_id: int = 0) -> dict[int, tuple]:
    """
    Build a napari-compatible label colormap for *n_classes* semantic classes.

    The *background_id* is mapped to transparent (alpha=0).  All other class
    IDs are assigned visually distinct RGBA colours.

    Parameters
    ----------
    n_classes:
        Total number of classes *excluding* background.
    background_id:
        Integer value used for the background (typically 0).

    Returns
    -------
    dict[int, tuple]
        Mapping from integer class ID → RGBA tuple in [0, 1] range.
        Pass to ``napari.layers.Labels`` via ``color=colormap``.
    """
    import colorsys

    colormap: dict[int, tuple] = {background_id: (0.0, 0.0, 0.0, 0.0)}
    for i in range(1, n_classes + 1):
        hue = (i - 1) / max(n_classes, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.90)
        colormap[i] = (r, g, b, 0.7)
    return colormap


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def print_sdata_summary(sdata: spatialdata.SpatialData) -> None:
    """Print a human-readable inventory of all elements in *sdata*."""
    print("=" * 60)
    print("SpatialData summary")
    print("=" * 60)

    for category in ("images", "labels", "shapes", "points", "tables"):
        store = getattr(sdata, category, {})
        if not store:
            continue
        print(f"\n[{category}]")
        for key, element in store.items():
            try:
                shape = dict(element["scale0"].ds.dims)  # multiscale
            except Exception:
                try:
                    shape = dict(element.sizes)
                except Exception:
                    shape = str(type(element))
            print(f"  {key!r:30s} → {shape}")

    print("=" * 60)
