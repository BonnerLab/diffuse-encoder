"""The Natural Scenes Dataset (Allen 2021)."""
# This script is written by Raj from Bonner Lab
# Thank you for his help
__all__ = (
    "N_SUBJECTS",
    "StimulusSet",
    "compute_shared_stimuli",
)

import typing
from collections.abc import Mapping, Sequence
#This process helps to load images
from collections.abc import Collection
import hashlib
from pathlib import Path
import numpy as np
import xarray as xr
from PIL import Image
import PIL
import requests
from io import BytesIO
import more_itertools
import numpy as np
import xarray as xr
from bonner.caching import cache
from bonner.datasets.allen2021_natural_scenes import (
    IDENTIFIER,
    N_SUBJECTS,
    StimulusSet,
    compute_shared_stimuli,
    convert_ndarray_to_nifti1image,
    create_roi_selector,
    load_betas,
    load_rois,
)
from bonner.datasets.allen2021_natural_scenes._visualization import (
    load_transformation,
    reshape_dataarray_to_brain,
)
# from bonner.plotting import plot_brain_map
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nibabel.nifti1 import Nifti1Image
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_roi
from nilearn.surface import vol_to_surf
from scipy.ndimage import map_coordinates
from tqdm.auto import tqdm

MNI_SHAPE = (182, 218, 182)
ROIS: Mapping[str, Sequence[dict[str, str]]] = {
    "general": ({"source": "nsdgeneral", "label": "nsdgeneral"},),
    "V1-4": ({"source": "prf-visualrois"},),
    "V1": (
        {"source": "prf-visualrois", "label": "V1v"},
        {"source": "prf-visualrois", "label": "V1d"},
    ),
    "V2": (
        {"source": "prf-visualrois", "label": "V2v"},
        {"source": "prf-visualrois", "label": "V2d"},
    ),
    "V3": (
        {"source": "prf-visualrois", "label": "V3v"},
        {"source": "prf-visualrois", "label": "V3d"},
    ),
    "V4": ({"source": "prf-visualrois", "label": "hV4"},),
    "frontal": (
        {"source": "corticalsulc", "label": "IFG"},
        {"source": "corticalsulc", "label": "IFRS"},
        {"source": "corticalsulc", "label": "MFG"},
        {"source": "corticalsulc", "label": "OS"},
        {"source": "corticalsulc", "label": "PrCS"},
        {"source": "corticalsulc", "label": "SFRS"},
        {"source": "corticalsulc", "label": "SRS"},
        {"source": "HCP_MMP1", "label": "1"},
        {"source": "HCP_MMP1", "label": "2"},
        {"source": "HCP_MMP1", "label": "3a"},
        {"source": "HCP_MMP1", "label": "3b"},
    ),
    "places": ({"source": "floc-places"},),
    "faces": ({"source": "floc-faces"},),
    "bodies": ({"source": "floc-bodies"},),
    "words": ({"source": "floc-words"},),
} | {
    f"{stream} visual stream": ({"source": "streams", "label": stream},)
    for stream in (
        "early",
        "lateral",
        "parietal",
        "ventral",
        "midlateral",
        "midparietal",
        "midventral",
    )
}


@cache(
    "data"
    f"/dataset={IDENTIFIER}"
    "/betas"
    "/resolution={resolution}"
    "/preprocessing={preprocessing}"
    "/z_score={z_score}"
    "/roi={roi}"
    "/subject={subject}.nc",
)
def _open_betas_by_roi(
    *,
    subject: int,
    resolution: str,
    preprocessing: str,
    z_score: bool,
    roi: str,
) -> xr.DataArray:
    if roi == "whole-brain":
        betas = load_betas(
            subject=subject,
            resolution=resolution,
            preprocessing=preprocessing,
            z_score=z_score,
        )
    else:
        betas = (
            _open_betas_by_roi(
                subject=subject,
                resolution=resolution,
                preprocessing=preprocessing,
                z_score=z_score,
                roi="whole-brain",
            )
            .load()
            .set_index({"neuroid": ("x", "y", "z")})
        )
        rois = load_rois(subject=subject, resolution=resolution).load()
        selector = create_roi_selector(rois=rois, selectors=ROIS[roi])
        selector = (
            rois.isel({"neuroid": selector})
            .set_index({"neuroid": ("x", "y", "z")})
            .indexes["neuroid"]
        )
        # remove invalid voxels present in `selector` but removed in `betas`
        betas = (
            betas
            .sel(neuroid=list(set(selector) & set(betas["neuroid"].data)))
            .reset_index("neuroid")
        )

    return betas


def load_dataset(
    *,
    subject: int,
    resolution: str = "1pt8mm",
    preprocessing: str = "fithrf",
    z_score: bool = True,
    roi: str = "general",
) -> xr.DataArray:
    betas = _open_betas_by_roi(
        resolution=resolution,
        preprocessing=preprocessing,
        z_score=z_score,
        roi=roi,
        subject=subject,
    ).assign_attrs({"roi": roi})

    identifier = ".".join([f"{key}={value}" for key, value in betas.attrs.items()])
    return (
        betas.rename(f"{IDENTIFIER}.{identifier}")
        .set_xindex(["stimulus", "repetition"])
        .set_xindex(["x", "y", "z"])
    )
    

#Load some 
def compute_shared_stimuli(
    assemblies: Collection[xr.DataArray], *, n_repetitions: int = 1
) -> set[int]:
    """Gets the IDs of the stimuli shared across all the provided assemblies.

    Args:
        assemblies: assemblies for different subjects
        n_repetitions: minimum number of repetitions for the shared stimuli in each subject

    Returns:
        shared stimulus ids
    """
    try:
        return set.intersection(
            *[
                set(
                    assembly["stimulus"].values[
                        (assembly["repetition"] == n_repetitions - 1).values
                    ]
                )
                for assembly in assemblies
            ]
        )
    except Exception:
        return set.intersection(
            *[set(assembly["stimulus"].values) for assembly in assemblies]
        )


def filter_by_stimulus(data: xr.DataArray, *, stimuli: set[int]) -> xr.DataArray:
    stimuli = np.array(sorted(list(stimuli)))
    hash_ = hashlib.sha1(stimuli).hexdigest()

    hashmap: dict[str, list[int]] = {}
    for index, value in enumerate(data["stimulus"].values):
        if value in hashmap:
            hashmap[value].append(index)
        else:
            hashmap[value] = [index]

    indices = []
    for stimulus in stimuli:
        if stimulus in hashmap:
            indices.extend(hashmap[stimulus])

    return (
        data.load()
        .isel({"presentation": indices})
        .assign_attrs({"stimuli": hash_})
        .rename(f"{data.name}.stimuli={hash_}")
    )


def split_by_repetition(
    data: xr.DataArray, *, n_repetitions: int
) -> dict[int, xr.DataArray]:
    return {
        repetition: (
            data.load()
            .isel({"presentation": data["repetition"] == repetition})
            .rename(f"{data.name}.repetition={repetition}")
        )
        for repetition in range(n_repetitions)
    }