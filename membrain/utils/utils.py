import os
from pathlib import Path
from typing import List, Tuple

import mrcfile
import numpy as np
import yaml

from cets_data_model.models.models import (
    AnnotatedObject,
    CoordinateSystem,
    Axis,
    AxisType,
    AxisUnit,
    Scale,
    SegmentationMask3D,
    ProbabilityMap3D,
)
from membrain.constants import MEMBRANE_FEATURE_TYPE


def write_segmentation_yaml(
    segmentation: SegmentationMask3D | ProbabilityMap3D, output_directory: Path
) -> None:
    # The annotation id already contains the tomogram id and the kind of segmentation
    _write_obj_yaml(
        segmentation, output_directory / f"{segmentation.id}_membrain_to_cets.yaml"
    )


def _write_obj_yaml(
    cets_obj: SegmentationMask3D | ProbabilityMap3D, yaml_file: Path
) -> None:
    metadata_dict = cets_obj.model_dump(mode="json")
    with open(yaml_file, "w") as f:
        yaml.dump(metadata_dict, f, sort_keys=False, explicit_start=True)
    print(f"yaml file successfully written! -> {yaml_file}")


def gen_coordinate_systems(
    name: str, ndim: int = 2
) -> Tuple[CoordinateSystem, CoordinateSystem]:
    """Builds the (array, physical) coordinate-system pair for an ``ndim`` image/frame.

    Names follow the CETS proposal convention ``{name}_array`` / ``{name}_physical``. The array
    system is pixel/array coords (unit ``pixel``); the physical system is in Å. ``ndim`` is 2 for
    2-D images (x, y) and 3 for volumes (x, y, z).
    """
    axes = ("x", "y", "z")[:ndim]
    array_cs = CoordinateSystem(
        name=f"{name}_array",
        axes=[
            Axis(name=a, axis_type=AxisType.array, axis_unit=AxisUnit.pixel)
            for a in axes
        ],
    )
    physical_cs = CoordinateSystem(
        name=f"{name}_physical",
        axes=[
            Axis(name=a, axis_type=AxisType.space, axis_unit=AxisUnit.angstrom)
            for a in axes
        ],
    )
    return array_cs, physical_cs


def gen_array_to_physical(
    pixel_size: float, array_cs_name: str, physical_cs_name: str, ndim: int = 2
) -> Scale:
    """The single canonical ``array_to_physical`` transformation for an image: a Scale mapping
    pixel/array coordinates to physical (Å) coordinates by the (isotropic) pixel/voxel size.
    The spec requires exactly one such transformation per image."""
    return Scale(
        scale=[pixel_size] * ndim,
        name="array_to_physical",
        input=array_cs_name,
        output=physical_cs_name,
    )


def gen_membrane_object() -> AnnotatedObject:
    """The CETS description of the object segmented by MemBrain-seg."""
    return AnnotatedObject(feature_type=MEMBRANE_FEATURE_TYPE)


def gen_tomogram_frame(
    tomo_id: str, pixel_size: float
) -> Tuple[List[CoordinateSystem], List[Scale]]:
    """Coordinate systems and the array_to_physical scale of a volume sampled on the voxel
    grid of the tomogram ``tomo_id``. The names match the ones of the CETS tomogram, so they
    refer to the very same frame."""
    array_cs, physical_cs = gen_coordinate_systems(tomo_id, ndim=3)
    array_to_physical = gen_array_to_physical(
        pixel_size, array_cs.name, physical_cs.name, ndim=3
    )
    return [array_cs, physical_cs], [array_to_physical]


def validate_file(filename: os.PathLike, expected_ext: str) -> Path:
    if filename is None:
        raise ValueError("The introduced file cannot be None")
    p = Path(filename).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} does not exist: {p}") from None

    if not p.is_file():
        raise IsADirectoryError(f"{filename} must be a file, not a directory: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for {filename}: {p}")

    ext = p.suffix
    if ext != expected_ext:
        raise ValueError(f"Invalid file extension '{ext}'. Expected: {expected_ext}")
    return p


def get_max_label(mrc_file: os.PathLike) -> int:
    """Returns the highest voxel value of a segmentation mask. MemBrain-seg binary masks
    contain only 0 and 1, while connected-component masks contain the labels 0..N."""
    with mrcfile.mmap(mrc_file, mode="r", permissive=True) as mrc:
        return int(np.max(mrc.data))
