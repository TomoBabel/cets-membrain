from pathlib import Path

import mrcfile
import numpy as np

from membrain.constants import MRC_EXT, SCORES_SUFFIX, SEGMENTED_SUFFIX, SKELETON_SUFFIX

# Real MemBrain-seg data, used by tests_cli_fixtures.py
DATA_DIR = Path(__file__).parent / "data"

TOMO_ID = "TS_01"
PIXEL_SIZE = 5.4
# Volume dimensions in x, y, z (the mrc data is stored as z, y, x)
VOL_DIMS = (12, 10, 8)


def _write_mrc(mrc_file: Path, data: np.ndarray, voxel_size: float) -> Path:
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size
    return mrc_file


def _empty_volume(dtype: type) -> np.ndarray:
    x, y, z = VOL_DIMS
    return np.zeros((z, y, x), dtype=dtype)


def write_binary_mrc(directory: Path, voxel_size: float = PIXEL_SIZE) -> Path:
    data = _empty_volume(np.int8)
    data[2:4, 3:6, 4:9] = 1
    return _write_mrc(
        directory / f"{TOMO_ID}{SEGMENTED_SUFFIX}{MRC_EXT}", data, voxel_size
    )


def write_connected_components_mrc(directory: Path) -> Path:
    data = _empty_volume(np.int8)
    data[1:3, 1:3, 1:3] = 1
    data[5:7, 6:8, 8:10] = 2
    return _write_mrc(
        directory / f"{TOMO_ID}{SEGMENTED_SUFFIX}{MRC_EXT}", data, PIXEL_SIZE
    )


def write_skeleton_mrc(directory: Path) -> Path:
    data = _empty_volume(np.int8)
    data[3, 4, 4:9] = 1
    return _write_mrc(
        directory / f"{TOMO_ID}{SEGMENTED_SUFFIX}{SKELETON_SUFFIX}{MRC_EXT}",
        data,
        PIXEL_SIZE,
    )


def write_scores_mrc(directory: Path) -> Path:
    data = _empty_volume(np.float32) - 10.0
    data[2:4, 3:6, 4:9] = 5.0
    return _write_mrc(
        directory / f"{TOMO_ID}{SCORES_SUFFIX}{MRC_EXT}", data, PIXEL_SIZE
    )
