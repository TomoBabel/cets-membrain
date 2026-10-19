import os

from cets_data_model.models.models import (
    SegmentationMask3D,
    SegmentationLabel,
    SegmentationType,
    AnnotationType,
)
from cets_data_model.utils.image_utils import get_mrc_info
from membrain.constants import (
    MEMBRANE_LABEL_VALUE,
    MRC_EXT,
    SKELETON_SUFFIX,
    MembrainMaskType,
)
from membrain.utils.utils import (
    validate_file,
    gen_membrane_object,
    gen_tomogram_frame,
    get_max_label,
)


class MembrainSegmentation:
    def membrain_to_cets(
        self,
        segmentation_mrc: os.PathLike,
        tomo_id: str,
        pixel_size: float | None = None,
        mask_type: MembrainMaskType | None = None,
        derived_from_annotation_id: str | None = None,
        annotation_id: str | None = None,
    ) -> SegmentationMask3D:
        """Converts a MemBrain-seg membrane segmentation (``*_segmented.mrc``) or its
        skeleton (``*_skel.mrc``) into CETS metadata.

        All of them are categorical volumes sampled on the same voxel grid as the segmented
        tomogram, so they are represented as a ``SegmentationMask3D`` annotation (stored
        under ``Region.annotations``) that links back to the tomogram through
        ``source_tomogram_id``.

        :param segmentation_mrc: path to the MemBrain-seg segmentation or skeleton.
        :type segmentation_mrc: os.PathLike.

        :param tomo_id: identifier of the tomogram the segmentation was obtained from. It is
        used as ``source_tomogram_id`` and to name the coordinate systems, so it must match the
        id of the CETS tomogram ({tomo_id}_array / {tomo_id}_physical refer to the same frame).
        MemBrain-seg file names do not contain a reliable tomogram id, so it must be supplied.
        :type tomo_id: str.

        :param pixel_size: sampling rate (Å/px) of the segmentation. If given, it is used;
        otherwise it is read from the mrc header, which MemBrain-seg copies from the input
        tomogram.
        :type pixel_size: float, optional. Defaults to None.

        :param mask_type: kind of mask. When omitted, files ending in ``_skel.mrc`` are
        considered skeletons, and the rest binary or connected-component masks depending on
        their highest label (MemBrain-seg does not record it anywhere else).
        :type mask_type: MembrainMaskType, optional. Defaults to None.

        :param derived_from_annotation_id: id of the annotation this mask was computed from:
        for a segmentation, the score map it was thresholded from (see
        ``MembrainScores.get_annotation_id``); for a skeleton, the segmentation it was
        skeletonized from (see ``MembrainSegmentation.get_annotation_id``).
        :type derived_from_annotation_id: str, optional. Defaults to None.

        :param annotation_id: id of the annotation. It must be unique within its
        Region.annotations, so it has to be provided when several MemBrain-seg runs on the
        same tomogram are converted. Defaults to ``get_annotation_id``.
        :type annotation_id: str, optional. Defaults to None.
        """
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"The pixel size must be positive. Got: {pixel_size}")
        mask_path = validate_file(segmentation_mrc, expected_ext=MRC_EXT)
        mrc_info = get_mrc_info(mask_path)
        voxel_size = pixel_size or mrc_info.apix_x
        if not voxel_size:
            raise ValueError(
                f"No pixel size was provided and {mask_path} does not contain it in its "
                f"header."
            )
        if mask_type is None:
            mask_type = self.get_mask_type(mask_path)
        coordinate_systems, coordinate_transformations = gen_tomogram_frame(
            tomo_id, voxel_size
        )
        membrane = gen_membrane_object()
        if mask_type == MembrainMaskType.connected_components:
            # Each non-background value (1..N) is a membrane instance
            segmentation_type = SegmentationType.instance
            labels = []
        else:
            # Binary segmentations and skeletons: membrane voxels are 1, background 0
            segmentation_type = SegmentationType.semantic
            labels = [
                SegmentationLabel(
                    label_value=MEMBRANE_LABEL_VALUE, annotated_object=membrane
                )
            ]
        return SegmentationMask3D(
            id=annotation_id or self.get_annotation_id(tomo_id, mask_type),
            name=f"MemBrain-seg membrane segmentation ({mask_type.value}) for {tomo_id}",
            annotation_type=AnnotationType.segmentation_mask_3D,
            source_tomogram_id=tomo_id,
            annotated_object=membrane,
            derived_from_annotation_id=derived_from_annotation_id,
            segmentation_type=segmentation_type,
            labels=labels,
            path=str(mask_path),
            width=mrc_info.size_x,
            height=mrc_info.size_y,
            depth=mrc_info.size_z,
            coordinate_systems=coordinate_systems,
            coordinate_transformations=coordinate_transformations,
        )

    @staticmethod
    def get_annotation_id(tomo_id: str, mask_type: MembrainMaskType) -> str:
        """Id of the mask annotation, so that the masks derived from it (e.g. its skeleton)
        can refer to it (derived_from_annotation_id)."""
        return f"membrain_{mask_type.value}_{tomo_id}"

    @staticmethod
    def get_mask_type(mask_path: os.PathLike) -> MembrainMaskType:
        """Kind of MemBrain-seg mask, from its name (skeletons) and its highest label."""
        if str(mask_path).endswith(f"{SKELETON_SUFFIX}{MRC_EXT}"):
            return MembrainMaskType.skeleton
        # A connected-component mask with a single component cannot be distinguished from a
        # binary one, but in that case both have the same meaning.
        if get_max_label(mask_path) > 1:
            return MembrainMaskType.connected_components
        return MembrainMaskType.binary
