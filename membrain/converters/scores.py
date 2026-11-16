import os
from pathlib import Path

from cets_data_model.models.models import (
    ProbabilityMap3D,
    ProbabilityMapValueType,
    AnnotationType,
)
from cets_data_model.utils.image_utils import get_mrc_info
from membrain.constants import (
    MRC_EXT,
    SCORES_SUFFIX,
    SEGMENTATION_THRESHOLD_DEFAULT,
)
from membrain.utils.utils import (
    export_mrc,
    validate_file,
    gen_membrane_object,
    gen_tomogram_frame,
    get_voxel_size,
)


class MembrainScores:
    def membrain_to_cets(
        self,
        scores_mrc: os.PathLike,
        tomo_id: str,
        pixel_size: float | None = None,
        threshold: float | None = SEGMENTATION_THRESHOLD_DEFAULT,
        annotation_id: str | None = None,
    ) -> ProbabilityMap3D:
        """Converts a MemBrain-seg membrane score map (``*_scores.mrc``, written with
        --store-probabilities) into CETS metadata.

        The score map is a real-valued volume sampled on the same voxel grid as the segmented
        tomogram, so it is represented as a ``ProbabilityMap3D`` annotation (stored under
        ``Region.annotations``) that links back to the tomogram through
        ``source_tomogram_id``. Note that MemBrain-seg stores the raw network outputs (logits,
        unbounded, the segmentation being score > --segmentation-threshold, 0.0 by default),
        not probabilities in [0, 1].

        :param scores_mrc: path to the MemBrain-seg score map.
        :type scores_mrc: os.PathLike.

        :param tomo_id: identifier of the tomogram the score map was obtained from. It is used
        as ``source_tomogram_id`` and to name the coordinate systems, so it must match the id of
        the CETS tomogram ({tomo_id}_array / {tomo_id}_physical refer to the same frame).
        :type tomo_id: str.

        :param pixel_size: sampling rate (Å/px) of the score map. If given, it is used;
        otherwise it is read from the mrc header, which MemBrain-seg copies from the input
        tomogram.
        :type pixel_size: float, optional. Defaults to None.

        :param threshold: --segmentation-threshold used by MemBrain-seg to obtain the
        segmentation from these scores (voxels with a score above it are membrane). It is not
        recorded in the output files, so pass the one used if it was not the default; None if
        unknown.
        :type threshold: float, optional. Defaults to 0.0 (the MemBrain-seg default).

        :param annotation_id: id of the annotation. It must be unique within its
        Region.annotations, so it has to be provided when several MemBrain-seg runs on the
        same tomogram are converted. Defaults to ``get_annotation_id``.
        :type annotation_id: str, optional. Defaults to None.
        """
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"The pixel size must be positive. Got: {pixel_size}")
        scores_path = validate_file(scores_mrc, expected_ext=MRC_EXT)
        mrc_info = get_mrc_info(scores_path)
        voxel_size = pixel_size or mrc_info.apix_x
        if not voxel_size:
            raise ValueError(
                f"No pixel size was provided and {scores_path} does not contain it in its "
                f"header."
            )
        coordinate_systems, coordinate_transformations = gen_tomogram_frame(
            tomo_id, voxel_size
        )
        return ProbabilityMap3D(
            id=annotation_id or self.get_annotation_id(tomo_id),
            name=f"MemBrain-seg membrane scores (logits) for {tomo_id}",
            annotation_type=AnnotationType.probability_map_3D,
            source_tomogram_id=tomo_id,
            annotated_object=gen_membrane_object(),
            value_type=ProbabilityMapValueType.logit,
            threshold=threshold,
            path=str(scores_path),
            width=mrc_info.size_x,
            height=mrc_info.size_y,
            depth=mrc_info.size_z,
            coordinate_systems=coordinate_systems,
            coordinate_transformations=coordinate_transformations,
        )

    def cets_to_membrain(
        self,
        scores: ProbabilityMap3D,
        output_directory: os.PathLike,
        tomo_name: str,
        copy: bool = False,
    ) -> Path:
        """Makes the score map of a CETS ``ProbabilityMap3D`` annotation available with the
        name MemBrain-seg gives it (``<tomo_name>_scores.mrc``), e.g. to threshold it with
        membrain thresholds.

        The map itself is the mrc file the annotation points to, so nothing is converted: it
        is linked, or copied if requested or if the sampling rate of its header is not the
        one of the annotation (the header of the copy is then updated). MemBrain-seg score
        maps hold logits, so annotations with other kinds of values are rejected.

        :param scores: the score map annotation.
        :type scores: ProbabilityMap3D.

        :param output_directory: directory in which the score map will be written.
        :type output_directory: os.PathLike.

        :param tomo_name: name of the tomogram in the MemBrain-seg file names: the name of
        the segmented tomogram file without extension.
        :type tomo_name: str.

        :param copy: copy the map instead of linking it.
        :type copy: bool, optional. Defaults to False.

        :returns: the path of the score map.
        """
        if scores.value_type not in (None, ProbabilityMapValueType.logit.value):
            raise ValueError(
                f"The annotation [{scores.id}] holds values of type "
                f"'{scores.value_type}', not MemBrain-seg scores (logits)."
            )
        if not scores.path:
            raise ValueError(f"The annotation [{scores.id}] has no file.")
        return export_mrc(
            scores.path,
            Path(output_directory) / f"{tomo_name}{SCORES_SUFFIX}{MRC_EXT}",
            voxel_size=get_voxel_size(scores),
            copy=copy,
        )

    @staticmethod
    def get_annotation_id(tomo_id: str) -> str:
        """Id of the score map annotation, so that the segmentation thresholded from it can
        refer to it (derived_from_annotation_id)."""
        return f"membrain_scores_{tomo_id}"
