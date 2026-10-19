from enum import Enum

from cets_data_model.models.models import SegmentedFeatureType

MRC_EXT = ".mrc"

# MemBrain-seg output files. Natively they are named <tomo>_<ckpt>_segmented.mrc,
# <tomo>_scores.mrc and <segmentation>_skel.mrc (scipion-em-membrain renames the
# segmentation to <tsId>_segmented.mrc, keeping the other suffixes).
SEGMENTED_SUFFIX = "_segmented"
SCORES_SUFFIX = "_scores"
SKELETON_SUFFIX = "_skel"

# CETS SegmentedFeatureType of the object segmented by MemBrain-seg
MEMBRANE_FEATURE_TYPE = SegmentedFeatureType.membrane
# Value of the membrane voxels in binary segmentations and skeletons (the background is 0)
MEMBRANE_LABEL_VALUE = 1
# Default of membrain segment --segmentation-threshold (also used by scipion-em-membrain): the
# voxels with a score above it are segmented as membrane.
SEGMENTATION_THRESHOLD_DEFAULT = 0.0


class MembrainMaskType(str, Enum):
    """Kinds of MemBrain-seg segmentation masks."""

    binary = "binary"
    """Thresholded segmentation: 1 = membrane, 0 = background."""
    connected_components = "connected_components"
    """Segmentation stored as labelled connected components (--store-connected-components):
    0 = background, 1..N = membrane instance."""
    skeleton = "skeleton"
    """Skeletonized (thinned) binary segmentation (membrain skeletonize)."""
