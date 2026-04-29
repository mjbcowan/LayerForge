"""
Sample data helpers for LayerForge tutorials.

Provides a pre-annotated skin histology SpatialData object so that tutorial
users can skip the interactive annotation step and jump straight to exploring
labels, measuring morphology, or refining an existing mask.

Usage
-----
    from LayerForge.sample_data import load_skin_sample

    sdata = load_skin_sample()
    # sdata.images["skin"]       — multiscale RGB image (900×900)
    # sdata.labels["skin_labels"] — pre-made semantic labels (5 classes)
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import spatialdata as sd
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity

from skimage import data

# Class ID → tissue name mapping bundled with the sample dataset.
SKIN_CLASS_LABELS: dict[int, str] = {
    1: "stratum corneum",
    2: "epidermis",
    3: "papillary dermis",
    4: "reticular dermis",
    5: "vascular structure",
}


def load_skin_sample(scale_factors: list[int] | None = None) -> sd.SpatialData:
    """
    Return a SpatialData object containing the scikit-image skin image and
    the bundled pre-annotated semantic labels.

    Parameters
    ----------
    scale_factors:
        Multiscale pyramid factors for both the image and labels elements.
        Defaults to ``[2, 2]`` (three resolution levels: 900, 450, 225).

    Returns
    -------
    spatialdata.SpatialData
        Keys:
        - ``images["skin"]``        — multiscale RGB image
        - ``labels["skin_labels"]`` — multiscale semantic labels (int32)
    """
    from skimage import data as skdata  # optional at import time

    if scale_factors is None:
        scale_factors = [2, 2]

    # --- image ----------------------------------------------------------
    full_img = data.skin()

    img = full_img[:900, :900]
         # (900, 900, 3) uint8
    img_cyx = np.moveaxis(img, -1, 0)        # (3, 900, 900)
    image = Image2DModel.parse(
        img_cyx,
        dims=("c", "y", "x"),
        scale_factors=scale_factors,
        transformations={"global": Identity()},
    )

    # --- labels ---------------------------------------------------------
    data_file = files("LayerForge.data").joinpath("skin_labels.npy")
    mask = np.load(str(data_file)).astype(np.int32)  # (900, 900)
    labels = Labels2DModel.parse(
        mask,
        dims=["y", "x"],
        scale_factors=scale_factors,
        transformations={"global": Identity()},
    )

    return sd.SpatialData(
        images={"skin": image},
        labels={"skin_labels": labels},
    )
