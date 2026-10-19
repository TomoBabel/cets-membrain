"""Converts the MemBrain-seg outputs found in a directory into CETS metadata."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from cets_data_model.models.models import ProbabilityMap3D, SegmentationMask3D
from membrain.converters.scores import MembrainScores
from membrain.converters.segmentation import MembrainSegmentation
from membrain.utils.discovery import (
    MembrainOutputFile,
    MembrainOutputKind,
    SkeletonMatch,
    find_membrain_outputs,
    find_skeleton_sources,
)
from membrain.utils.utils import write_segmentation_yaml

### MEMBRAIN TO CETS ###############################################################
scratch_dir = Path("/home/jjimenez/CZII/cets_scratch_dir")
# Any directory containing MemBrain-seg outputs (searched recursively). The files are found
# from their native names: <tomo>_<ckpt>_segmented.mrc (or <tomo>_segmented.mrc),
# <tomo>_scores.mrc and <segmentation>_skel.mrc.
membrain_dir = Path("/home/jjimenez/ScipionUserData/projects/czii_membrain")
# Ids of the segmented tomograms, needed only to split the tomogram id from the checkpoint
# token in segmentation names when there are no score maps (empty: taken from the scores).
tomo_ids: List[str] = []
# Skeletons are matched to their source segmentation by file name. Only when several
# segmentations share the name and the inferred one (the latest written before the skeleton)
# is not right, give the --label-path used in membrain skeletonize: {skeleton: segmentation}.
skeleton_label_paths: Dict[Path, Path] = {}


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return number


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--sampling-rate",
    type=positive_float,
    default=None,
    help="Sampling rate (Å/px) of the segmented tomograms. If given, it is used for all the "
    "outputs; otherwise, it is read from the header of each output (MemBrain-seg copies it from the input tomogram).",
)
sampling_rate: Optional[float] = parser.parse_args().sampling_rate
# --segmentation-threshold used (not recorded in the output files)
segmentation_threshold = 0.0

outputs = find_membrain_outputs(membrain_dir, tomo_ids)
if not outputs:
    raise SystemExit(f"No MemBrain-seg outputs were found in {membrain_dir}")


# Kind of each mask (binary, connected components or skeleton), detected once from its labels
mask_types = {
    o.path: MembrainSegmentation.get_mask_type(o.path)
    for o in outputs
    if o.kind != MembrainOutputKind.scores
}


def get_annotation_ids(outputs: List[MembrainOutputFile]) -> Dict[Path, str]:
    """The converters' default ids depend on the tomogram and the kind of output, so the
    outputs of several runs on the same tomogram would collide. In that case, the directory
    (relative to membrain_dir) is appended to keep the ids unique within the region."""
    default_ids = {
        o.path: MembrainScores.get_annotation_id(o.tomo_id)
        if o.kind == MembrainOutputKind.scores
        else MembrainSegmentation.get_annotation_id(o.tomo_id, mask_types[o.path])
        for o in outputs
    }
    counts = Counter(default_ids.values())
    annotation_ids = {}
    for path, default_id in default_ids.items():
        if counts[default_id] > 1:
            rel_dir = path.parent.relative_to(membrain_dir).as_posix()
            annotation_ids[path] = f"{default_id}__{rel_dir.replace('/', '_')}"
        else:
            annotation_ids[path] = default_id
    return annotation_ids


annotation_ids = get_annotation_ids(outputs)
converted: Dict[Path, SegmentationMask3D | ProbabilityMap3D] = {}

# Scores (--store-probabilities) -> ProbabilityMap3D (Region.annotations)
for output in outputs:
    if output.kind == MembrainOutputKind.scores:
        converted[output.path] = MembrainScores().membrain_to_cets(
            scores_mrc=output.path,
            tomo_id=output.tomo_id,
            pixel_size=sampling_rate,
            threshold=segmentation_threshold,
            annotation_id=annotation_ids[output.path],
        )

# Segmentations -> SegmentationMask3D (Region.annotations), semantic (binary) or instance
# (connected components) depending on their labels. membrain segment writes the scores in the
# same folder, so a segmentation is derived from the scores of its tomogram next to it.
segmentation = MembrainSegmentation()
for output in outputs:
    if output.kind == MembrainOutputKind.segmentation:
        scores_ids = [
            converted[o.path].id
            for o in outputs
            if o.kind == MembrainOutputKind.scores
            and o.tomo_id == output.tomo_id
            and o.path.parent == output.path.parent
        ]
        converted[output.path] = segmentation.membrain_to_cets(
            segmentation_mrc=output.path,
            tomo_id=output.tomo_id,
            pixel_size=sampling_rate,
            mask_type=mask_types[output.path],
            derived_from_annotation_id=scores_ids[0] if scores_ids else None,
            annotation_id=annotation_ids[output.path],
        )

# Skeletons -> SegmentationMask3D (Region.annotations), derived from the segmentation they
# were computed from, matched by file name (<segmentation>_skel.mrc <-> <segmentation>.mrc).
skeleton_sources = find_skeleton_sources(outputs, skeleton_label_paths)
for output in outputs:
    if output.kind == MembrainOutputKind.skeleton:
        source = skeleton_sources[output.path]
        source_annotation = converted.get(source.segmentation) if source else None
        if source is None or source_annotation is None:
            print(f"WARNING: no source segmentation found for {output.path}")
        elif source.match == SkeletonMatch.latest_before_skeleton:
            print(
                f"NOTE: {source.n_candidates} segmentations named "
                f"{source.segmentation.name}; {output.path} linked to the latest one "
                f"written before it ({source.segmentation}). Add it to "
                f"skeleton_label_paths if that is not the --label-path used."
            )
        converted[output.path] = segmentation.membrain_to_cets(
            segmentation_mrc=output.path,
            tomo_id=output.tomo_id,
            pixel_size=sampling_rate,
            mask_type=mask_types[output.path],
            derived_from_annotation_id=source_annotation.id
            if source_annotation
            else None,
            annotation_id=annotation_ids[output.path],
        )

for annotation in converted.values():
    write_segmentation_yaml(annotation, scratch_dir)
