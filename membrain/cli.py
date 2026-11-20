"""Command line of the MemBrain-seg <-> CETS converters (``cets-membrain``).

Examples::

    cets-membrain scores to-cets --scores TS_01_scores.mrc --tomo-id TS_01 \\
        -o TS_01_scores_cets.yaml
    cets-membrain segmentation to-cets --segmentation TS_01_ckpt_segmented.mrc \\
        --tomo-id TS_01 --derived-from membrain_scores_TS_01 -o TS_01_seg_cets.yaml
    cets-membrain segmentation from-cets -i TS_01_seg_cets.yaml --output-dir out \\
        --tomo-name TS_01
    cets-membrain scores from-cets -i TS_01_scores_cets.yaml --output-dir out

``to-cets`` writes a CETS yaml file with the annotation under ``segmentation`` (a
SegmentationMask3D: binary or connected-component segmentation, or skeleton) or ``scores``
(a ProbabilityMap3D), which is what ``from-cets`` reads.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from cets_data_model.models.models import ProbabilityMap3D, SegmentationMask3D
from cets_data_model.utils.cli import (
    FROM_CETS,
    TO_CETS,
    add_data_type,
    add_input_yaml,
    add_output_yaml,
    run_cli,
)
from cets_data_model.utils.yaml_io import load_cets_object, write_cets_yaml
from membrain.constants import SEGMENTATION_THRESHOLD_DEFAULT, MembrainMaskType
from membrain.converters.scores import MembrainScores
from membrain.converters.segmentation import MembrainSegmentation

SEGMENTATION_KEY = "segmentation"
SCORES_KEY = "scores"


def _tomo_name(args: argparse.Namespace, source_tomogram_id: Optional[str]) -> str:
    """--tomo-name, or the id of the tomogram the annotation was obtained from."""
    tomo_name = args.tomo_name or source_tomogram_id
    if not tomo_name:
        raise ValueError(
            f"{args.input} has no source tomogram id to name the files with: provide "
            f"--tomo-name."
        )
    return str(tomo_name)


def _add_from_cets_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the file in.",
    )
    parser.add_argument(
        "--tomo-name",
        help="Name of the tomogram in the MemBrain-seg file names (the segmented "
        "tomogram file name without extension). Default: the source tomogram id of the "
        "annotation.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the file instead of linking it.",
    )


def _add_to_cets_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tomo-id",
        required=True,
        help="Id of the CETS tomogram it was obtained from (source_tomogram_id; it also "
        "names the coordinate systems).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        help="Sampling rate (Å/px). Default: the one of the mrc header.",
    )
    parser.add_argument(
        "--annotation-id",
        help="Id of the annotation, unique within the region (needed when several "
        "MemBrain-seg runs on the same tomogram are converted). Default: generated from "
        "the tomogram id.",
    )


# ------------------------------------------------------------------------- segmentation
def segmentation_to_cets(args: argparse.Namespace) -> None:
    mask = MembrainSegmentation().membrain_to_cets(
        args.segmentation,
        args.tomo_id,
        pixel_size=args.pixel_size,
        mask_type=MembrainMaskType(args.mask_type) if args.mask_type else None,
        derived_from_annotation_id=args.derived_from,
        annotation_id=args.annotation_id,
    )
    out = write_cets_yaml(args.output, {SEGMENTATION_KEY: mask})
    print(f"CETS segmentation written -> {out}")


def segmentation_from_cets(args: argparse.Namespace) -> None:
    mask = load_cets_object(args.input, SEGMENTATION_KEY, SegmentationMask3D)
    out = MembrainSegmentation().cets_to_membrain(
        mask,
        args.output_dir,
        _tomo_name(args, mask.source_tomogram_id),
        checkpoint=args.checkpoint,
        skeleton=args.skeleton,
        copy=args.copy,
    )
    print(f"MemBrain-seg segmentation written -> {out}")


def _add_segmentation(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types,
        "segmentation",
        "Membrane segmentation (*_segmented.mrc) or its skeleton (*_skel.mrc).",
    )

    to_cets = directions.add_parser(
        TO_CETS, help="MemBrain-seg segmentation or skeleton -> CETS yaml."
    )
    to_cets.add_argument(
        "--segmentation",
        required=True,
        type=Path,
        help="Segmentation (*_segmented.mrc) or skeleton (*_skel.mrc).",
    )
    _add_to_cets_common(to_cets)
    to_cets.add_argument(
        "--mask-type",
        choices=[t.value for t in MembrainMaskType],
        help="Kind of mask. Default: skeleton if the file ends in _skel.mrc, else binary "
        "or connected components depending on its highest label.",
    )
    to_cets.add_argument(
        "--derived-from",
        metavar="ANNOTATION_ID",
        help="Id of the annotation it was computed from (derived_from_annotation_id): "
        "the score map of a segmentation, the segmentation of a skeleton.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=segmentation_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS,
        help="CETS yaml -> MemBrain-seg segmentation "
        "(<tomo>[_<checkpoint>]_segmented[_skel].mrc).",
    )
    add_input_yaml(from_cets, "CETS yaml with the segmentation.")
    _add_from_cets_common(from_cets)
    from_cets.add_argument(
        "--checkpoint",
        help="Name of the MemBrain-seg model checkpoint, added to the file name.",
    )
    from_cets.add_argument(
        "--skeleton",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether the mask is a skeleton (named *_skel.mrc). Default: whether its "
        "file ends in _skel.mrc.",
    )
    from_cets.set_defaults(func=segmentation_from_cets)


# ------------------------------------------------------------------------- scores
def scores_to_cets(args: argparse.Namespace) -> None:
    scores = MembrainScores().membrain_to_cets(
        args.scores,
        args.tomo_id,
        pixel_size=args.pixel_size,
        threshold=None if args.unknown_threshold else args.threshold,
        annotation_id=args.annotation_id,
    )
    out = write_cets_yaml(args.output, {SCORES_KEY: scores})
    print(f"CETS scores written -> {out}")


def scores_from_cets(args: argparse.Namespace) -> None:
    scores = load_cets_object(args.input, SCORES_KEY, ProbabilityMap3D)
    out = MembrainScores().cets_to_membrain(
        scores,
        args.output_dir,
        _tomo_name(args, scores.source_tomogram_id),
        copy=args.copy,
    )
    print(f"MemBrain-seg scores written -> {out}")


def _add_scores(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types, "scores", "Membrane score map (*_scores.mrc, logits)."
    )

    to_cets = directions.add_parser(
        TO_CETS, help="MemBrain-seg score map -> CETS yaml."
    )
    to_cets.add_argument(
        "--scores", required=True, type=Path, help="Score map (*_scores.mrc)."
    )
    _add_to_cets_common(to_cets)
    threshold = to_cets.add_mutually_exclusive_group()
    threshold.add_argument(
        "--threshold",
        type=float,
        default=SEGMENTATION_THRESHOLD_DEFAULT,
        help="--segmentation-threshold used to segment these scores (default: "
        "%(default)s, the MemBrain-seg default).",
    )
    threshold.add_argument(
        "--unknown-threshold",
        action="store_true",
        help="The segmentation threshold is unknown: do not record it.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=scores_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS, help="CETS yaml -> MemBrain-seg score map (<tomo>_scores.mrc)."
    )
    add_input_yaml(from_cets, "CETS yaml with the score map.")
    _add_from_cets_common(from_cets)
    from_cets.set_defaults(func=scores_from_cets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cets-membrain",
        description="Converts MemBrain-seg files to and from CETS metadata (yaml).",
    )
    data_types = parser.add_subparsers(
        dest="data_type", metavar="{segmentation,scores}", required=True
    )
    _add_segmentation(data_types)
    _add_scores(data_types)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
