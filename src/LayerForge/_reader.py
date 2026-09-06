"""
npe2 reader contribution for LayerForge.

Loads .tif/.tiff/.png/.jpg/.jpeg files via :func:`spatialdata_io.generic` —
no bespoke image-dispatch loader — so that napari's native ``File > Open``
and drag-and-drop work out of the box. See the napari reader-plugin guide:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers

Public API
----------
default_scale_factors  – Pick sensible multiscale pyramid factors from image size.
load_image_as_sdata     – spatialdata_io.generic()-based loader wrapper.
napari_get_reader       – npe2 reader-contribution entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import spatialdata as sd
import spatialdata_io
from spatialdata.models import Image2DModel
from spatialdata.transformations import get_transformation

if TYPE_CHECKING:
    from npe2.types import LayerData, PathOrPaths, ReaderFunction

# Kept in sync with spatialdata_io.readers.generic.VALID_IMAGE_TYPES.
SUPPORTED_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# spatialdata_io.generic() requires data_axes for image files; these are
# sensible per-format defaults for the common case (microscopy TIFFs are
# channel-first, everyday photos loaded via dask-image are channel-last).
_DEFAULT_DATA_AXES = {
    ".tif": ("c", "y", "x"),
    ".tiff": ("c", "y", "x"),
    ".png": ("y", "x", "c"),
    ".jpg": ("y", "x", "c"),
    ".jpeg": ("y", "x", "c"),
}


def default_scale_factors(height: int, width: int, min_size: int = 256) -> list[int] | None:
    """
    Pick sensible multiscale ``scale_factors`` by halving image size.

    Reuses the halving pattern from the skin tutorial
    (docs/tutorials/annotate_RGB.md), where a 900x900 image is given
    ``scale_factors=[2, 2]`` (900 -> 450 -> 225): each extra ``2`` in the
    returned list halves the previous pyramid level. Halving continues while
    the current level's longest side is still above *min_size*.

    Returns ``None`` (single-scale) for images already at or below
    *min_size*.
    """
    n_levels = 0
    size = max(height, width)
    while size > min_size:
        size //= 2
        n_levels += 1
    return [2] * n_levels if n_levels else None


def load_image_as_sdata(
    path: "str | Path",
    data_axes: "tuple[str, ...] | list[str] | None" = None,
    image_key: "str | None" = None,
    scale_factors: "list[int] | None | str" = "auto",
    coordinate_system: "str | None" = None,
) -> sd.SpatialData:
    """
    Load a single .tif/.tiff/.png/.jpg/.jpeg file into a SpatialData object.

    All format dispatch, axis handling, and coordinate-system setup is
    delegated to :func:`spatialdata_io.generic`; this wrapper only adds a
    sensible default multiscale pyramid chosen from the image size, since
    ``spatialdata_io.generic`` itself does not pick one.

    Parameters
    ----------
    path:
        Path to the input file.
    data_axes:
        Axis order of the raw file, forwarded to ``spatialdata_io.generic``.
        Defaults to ``("c", "y", "x")`` for TIFFs and ``("y", "x", "c")``
        for PNG/JPEG.
    image_key:
        Key under which the image is stored in ``sdata.images``. Defaults
        to the file stem.
    scale_factors:
        ``"auto"`` (default) picks halving factors from image size via
        :func:`default_scale_factors`; pass an explicit list (e.g.
        ``[2, 2]``) or ``None`` for a single-scale image.
    coordinate_system:
        Forwarded to ``spatialdata_io.generic`` (defaults to ``"global"``
        there).

    Returns
    -------
    spatialdata.SpatialData
        A SpatialData object with the loaded image under *image_key*.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {path.suffix!r}; expected one of {SUPPORTED_SUFFIXES}"
        )

    image_key = image_key or path.stem
    if data_axes is None:
        data_axes = _DEFAULT_DATA_AXES[suffix]

    element = spatialdata_io.generic(
        path, data_axes=list(data_axes), coordinate_system=coordinate_system
    )

    cs = coordinate_system or "global"
    if scale_factors == "auto":
        scale_factors = default_scale_factors(element.sizes["y"], element.sizes["x"])

    if scale_factors:
        transform = get_transformation(element, to_coordinate_system=cs)
        element = Image2DModel.parse(
            element.data,
            dims=element.dims,
            transformations={cs: transform},
            scale_factors=scale_factors,
        )

    return sd.SpatialData(images={image_key: element})


def napari_get_reader(path: "PathOrPaths") -> "ReaderFunction | None":
    """npe2 reader contribution: recognise LayerForge-supported image files."""
    if isinstance(path, list):
        # Multi-file selection is not supported by this reader.
        return None
    if Path(path).suffix.lower() not in SUPPORTED_SUFFIXES:
        return None
    return _reader_function


def _reader_function(path: str) -> "list[LayerData]":
    from LayerForge.sdata_utils import sdata_image_to_layer_data

    sdata = load_image_as_sdata(path)
    image_key = next(iter(sdata.images))
    return sdata_image_to_layer_data(sdata, image_key)
