import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

from membrain.constants import MRC_EXT, SCORES_SUFFIX, SEGMENTED_SUFFIX, SKELETON_SUFFIX

# Checkpoint token that membrain segment inserts in the native segmentation file names:
# <tomo>_<ckpt>_segmented.mrc, being <ckpt> the basename of --ckpt-path (e.g.
# MemBrain_seg_v10_beta.ckpt). Used only when the tomogram id is not known.
_CKPT_TOKEN_RE = re.compile(r"^(?P<tomo>.+?)_(?P<ckpt>[^_].*\.ckpt)$")


class MembrainOutputKind(str, Enum):
    """Kinds of files written by MemBrain-seg."""

    segmentation = "segmentation"
    """membrain segment: <tomo>_<ckpt>_segmented.mrc (binary or connected components)."""
    scores = "scores"
    """membrain segment --store-probabilities: <tomo>_scores.mrc."""
    skeleton = "skeleton"
    """membrain skeletonize: <segmentation>_skel.mrc."""


@dataclass(frozen=True)
class MembrainOutputFile:
    """A MemBrain-seg output file found on disk."""

    path: Path
    kind: MembrainOutputKind
    tomo_id: str


def find_membrain_outputs(
    root: os.PathLike, tomo_ids: Optional[Iterable[str]] = None
) -> List[MembrainOutputFile]:
    """Recursively finds the MemBrain-seg output files under ``root`` from their native
    names (also accepting segmentations renamed to <tomo>_segmented.mrc). Any other mrc file
    (e.g. the input tomograms) is ignored.

    The tomogram id of a segmentation cannot be split from the checkpoint token in its name
    when either of them contains underscores, so it is matched against the known tomogram ids:
    the given ``tomo_ids`` plus the ones of the score maps found (<tomo>_scores.mrc).

    :returns: the output files, sorted by path.
    """
    mrc_files = sorted(Path(root).rglob(f"*{MRC_EXT}"))
    known_ids: Set[str] = set(tomo_ids or [])
    scores_end = f"{SCORES_SUFFIX}{MRC_EXT}"
    known_ids.update(
        f.name[: -len(scores_end)] for f in mrc_files if f.name.endswith(scores_end)
    )
    outputs = []
    for mrc_file in mrc_files:
        output = parse_membrain_filename(mrc_file, known_ids)
        if output is not None:
            outputs.append(output)
    return outputs


def parse_membrain_filename(
    path: os.PathLike, known_tomo_ids: Iterable[str] = ()
) -> Optional[MembrainOutputFile]:
    """Identifies a MemBrain-seg output file and its tomogram id from its name, or returns
    None if it is not named as a MemBrain-seg output."""
    path = Path(path)
    name = path.name
    if not name.endswith(MRC_EXT):
        return None
    stem = name[: -len(MRC_EXT)]
    if stem.endswith(SKELETON_SUFFIX):
        # <label stem>_skel: the label is usually a MemBrain-seg segmentation
        label_stem = stem[: -len(SKELETON_SUFFIX)]
        tomo_id = _tomo_id_from_segmented_stem(label_stem, known_tomo_ids)
        return MembrainOutputFile(
            path, MembrainOutputKind.skeleton, tomo_id or label_stem
        )
    if stem.endswith(SCORES_SUFFIX):
        return MembrainOutputFile(
            path, MembrainOutputKind.scores, stem[: -len(SCORES_SUFFIX)]
        )
    if stem.endswith(SEGMENTED_SUFFIX):
        tomo_id = _tomo_id_from_segmented_stem(stem, known_tomo_ids)
        if tomo_id:
            return MembrainOutputFile(path, MembrainOutputKind.segmentation, tomo_id)
    return None


def _tomo_id_from_segmented_stem(
    stem: str, known_tomo_ids: Iterable[str]
) -> Optional[str]:
    """Tomogram id of a <tomo>_<ckpt>_segmented or <tomo>_segmented file stem."""
    if not stem.endswith(SEGMENTED_SUFFIX):
        return None
    prefix = stem[: -len(SEGMENTED_SUFFIX)]
    # The longest known id that the name starts with (TS_1 must not match TS_10_...)
    matches = [
        tomo_id
        for tomo_id in known_tomo_ids
        if prefix == tomo_id or prefix.startswith(f"{tomo_id}_")
    ]
    if matches:
        return max(matches, key=len)
    match = _CKPT_TOKEN_RE.match(prefix)
    return match.group("tomo") if match else prefix or None


class SkeletonMatch(str, Enum):
    """How the source segmentation of a skeleton was identified."""

    given = "given"
    """Given by the caller (the --label-path of membrain skeletonize)."""
    same_directory = "same_directory"
    """The segmentation with the matching name in the skeleton directory."""
    unique_name = "unique_name"
    """The only segmentation with the matching name."""
    latest_before_skeleton = "latest_before_skeleton"
    """Several segmentations with the matching name: the latest one written before the
    skeleton (inferred)."""


@dataclass(frozen=True)
class SkeletonSource:
    """The segmentation a skeleton was computed from."""

    segmentation: Path
    match: SkeletonMatch
    n_candidates: int


def find_skeleton_sources(
    outputs: List[MembrainOutputFile],
    label_paths: Optional[Mapping[Path, Path]] = None,
) -> Dict[Path, Optional[SkeletonSource]]:
    """Finds the segmentation each skeleton was computed from, matching them by file name.

    MemBrain-seg does not record the --label-path of membrain skeletonize, which writes
    <label stem>_skel.mrc in its --out-folder, so the source is the segmentation named
    <label stem>.mrc. When several segmentations have that name (e.g. several runs on the same
    tomogram), it is resolved, in this order, by:

    1. ``label_paths`` ({skeleton path: --label-path}), if given for the skeleton;
    2. the segmentation in the same directory as the skeleton (the default --out-folder);
    3. the only segmentation with that name;
    4. the latest segmentation with that name written before the skeleton (a skeleton cannot
       be older than its source), or the latest one if all of them are newer.

    :returns: {skeleton path: its source, or None if no segmentation has the matching name}.
    """
    given = {
        Path(skel).resolve(): Path(label).resolve()
        for skel, label in (label_paths or {}).items()
    }
    segmentations = [
        o.path for o in outputs if o.kind == MembrainOutputKind.segmentation
    ]
    sources: Dict[Path, Optional[SkeletonSource]] = {}
    for output in outputs:
        if output.kind != MembrainOutputKind.skeleton:
            continue
        skeleton = output.path
        label_name = skeleton.name.replace(f"{SKELETON_SUFFIX}{MRC_EXT}", MRC_EXT)
        candidates = [s for s in segmentations if s.name == label_name]
        n_candidates = len(candidates)
        same_dir = [s for s in candidates if s.parent == skeleton.parent]
        if skeleton.resolve() in given:
            source = SkeletonSource(
                given[skeleton.resolve()], SkeletonMatch.given, n_candidates
            )
        elif same_dir:
            source = SkeletonSource(
                same_dir[0], SkeletonMatch.same_directory, n_candidates
            )
        elif n_candidates == 1:
            source = SkeletonSource(
                candidates[0], SkeletonMatch.unique_name, n_candidates
            )
        elif candidates:
            skeleton_mtime = skeleton.stat().st_mtime
            older = [s for s in candidates if s.stat().st_mtime <= skeleton_mtime]
            latest = max(older or candidates, key=lambda s: s.stat().st_mtime)
            source = SkeletonSource(
                latest, SkeletonMatch.latest_before_skeleton, n_candidates
            )
        else:
            source = None
        sources[skeleton] = source
    return sources
