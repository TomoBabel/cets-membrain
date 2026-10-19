# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import yaml

from cets_data_model.models.models import (
    AnnotationType,
    ProbabilityMap3D,
    ProbabilityMapValueType,
    SegmentationMask3D,
    SegmentationType,
    SegmentedFeatureType,
)
from membrain.constants import MembrainMaskType
from membrain.converters.scores import MembrainScores
from membrain.converters.segmentation import MembrainSegmentation
from membrain.tests import (
    PIXEL_SIZE,
    TOMO_ID,
    VOL_DIMS,
    write_binary_mrc,
    write_connected_components_mrc,
    write_scores_mrc,
    write_skeleton_mrc,
)
from membrain.utils.utils import write_segmentation_yaml

CETS_MEMBRAIN = "cets_membrain"


class CetsMembrainBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix=CETS_MEMBRAIN))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _check_volume(
        self, annotation: SegmentationMask3D | ProbabilityMap3D, mrc_file: Path
    ) -> None:
        self.assertEqual(annotation.source_tomogram_id, TOMO_ID)
        self.assertEqual(
            annotation.annotated_object.feature_type, SegmentedFeatureType.membrane
        )
        self.assertEqual(annotation.path, str(mrc_file.resolve()))
        self.assertEqual(
            (annotation.width, annotation.height, annotation.depth), VOL_DIMS
        )
        cs_names = [cs.name for cs in annotation.coordinate_systems or []]
        self.assertEqual(cs_names, [f"{TOMO_ID}_array", f"{TOMO_ID}_physical"])
        transformations = annotation.coordinate_transformations or []
        self.assertEqual(len(transformations), 1)
        self.assertEqual(transformations[0].name, "array_to_physical")
        # The mrc header stores the pixel size as float32
        for value in transformations[0].scale:
            self.assertAlmostEqual(value, PIXEL_SIZE, places=5)


class CetsMembrainSegmentationTest(CetsMembrainBaseTest):
    def _convert(
        self, mrc_file: Path, derived_from_annotation_id: str | None = None
    ) -> SegmentationMask3D:
        mask = MembrainSegmentation().membrain_to_cets(
            mrc_file, TOMO_ID, derived_from_annotation_id=derived_from_annotation_id
        )
        self.assertEqual(mask.annotation_type, AnnotationType.segmentation_mask_3D)
        self.assertEqual(mask.derived_from_annotation_id, derived_from_annotation_id)
        self._check_volume(mask, mrc_file)
        return mask

    def _check_semantic_membrane_labels(self, mask: SegmentationMask3D) -> None:
        self.assertEqual(mask.segmentation_type, SegmentationType.semantic)
        self.assertEqual(len(mask.labels), 1)
        self.assertEqual(mask.labels[0].label_value, 1)
        self.assertEqual(mask.labels[0].annotated_object, mask.annotated_object)

    def test_binary_segmentation(self) -> None:
        scores_id = MembrainScores.get_annotation_id(TOMO_ID)
        mask = self._convert(write_binary_mrc(self.test_dir), scores_id)
        self.assertEqual(mask.id, f"membrain_binary_{TOMO_ID}")
        self._check_semantic_membrane_labels(mask)

    def test_connected_components_segmentation(self) -> None:
        mask = self._convert(write_connected_components_mrc(self.test_dir))
        self.assertEqual(mask.id, f"membrain_connected_components_{TOMO_ID}")
        # Instance mask: every non-background value is a membrane instance
        self.assertEqual(mask.segmentation_type, SegmentationType.instance)
        self.assertEqual(mask.labels, [])

    def test_skeleton(self) -> None:
        segmentation_id = MembrainSegmentation.get_annotation_id(
            TOMO_ID, MembrainMaskType.binary
        )
        mask = self._convert(write_skeleton_mrc(self.test_dir), segmentation_id)
        self.assertEqual(mask.id, f"membrain_skeleton_{TOMO_ID}")
        self._check_semantic_membrane_labels(mask)

    def test_explicit_mask_type_and_pixel_size(self) -> None:
        # Header pixel size 1.0 (unset), so the given one must be used
        mrc_file = write_binary_mrc(self.test_dir, voxel_size=1.0)
        mask = MembrainSegmentation().membrain_to_cets(
            mrc_file,
            TOMO_ID,
            pixel_size=PIXEL_SIZE,
            mask_type=MembrainMaskType.connected_components,
        )
        self.assertEqual(mask.id, f"membrain_connected_components_{TOMO_ID}")
        self._check_volume(mask, mrc_file)

    def test_no_pixel_size(self) -> None:
        mrc_file = write_binary_mrc(self.test_dir, voxel_size=0.0)
        with self.assertRaises(ValueError):
            MembrainSegmentation().membrain_to_cets(mrc_file, TOMO_ID)

    def test_segmentation_yaml(self) -> None:
        mask = self._convert(write_binary_mrc(self.test_dir))
        write_segmentation_yaml(mask, self.test_dir)
        yaml_file = self.test_dir / f"{mask.id}_membrain_to_cets.yaml"
        with open(yaml_file) as f:
            reloaded = SegmentationMask3D.model_validate(yaml.safe_load(f))
        self.assertEqual(reloaded, mask)


class CetsMembrainScoresTest(CetsMembrainBaseTest):
    def test_scores(self) -> None:
        mrc_file = write_scores_mrc(self.test_dir)
        scores = MembrainScores().membrain_to_cets(mrc_file, TOMO_ID)
        self.assertEqual(scores.annotation_type, AnnotationType.probability_map_3D)
        self.assertEqual(scores.id, f"membrain_scores_{TOMO_ID}")
        self.assertEqual(scores.value_type, ProbabilityMapValueType.logit)
        self.assertEqual(scores.threshold, 0.0)
        self._check_volume(scores, mrc_file)

    def test_scores_custom_and_unknown_threshold(self) -> None:
        mrc_file = write_scores_mrc(self.test_dir)
        scores = MembrainScores().membrain_to_cets(mrc_file, TOMO_ID, threshold=-1.5)
        self.assertEqual(scores.threshold, -1.5)
        scores = MembrainScores().membrain_to_cets(mrc_file, TOMO_ID, threshold=None)
        self.assertIsNone(scores.threshold)

    def test_annotation_id_override(self) -> None:
        mrc_file = write_scores_mrc(self.test_dir)
        scores = MembrainScores().membrain_to_cets(
            mrc_file, TOMO_ID, annotation_id="scores_run_2"
        )
        self.assertEqual(scores.id, "scores_run_2")
        mask = MembrainSegmentation().membrain_to_cets(
            write_binary_mrc(self.test_dir),
            TOMO_ID,
            derived_from_annotation_id=scores.id,
            annotation_id="binary_run_2",
        )
        self.assertEqual(mask.id, "binary_run_2")
        self.assertEqual(mask.derived_from_annotation_id, "scores_run_2")

    def test_invalid_pixel_size(self) -> None:
        mrc_file = write_scores_mrc(self.test_dir)
        for pixel_size in (0.0, -5.4):
            with self.assertRaises(ValueError):
                MembrainScores().membrain_to_cets(
                    mrc_file, TOMO_ID, pixel_size=pixel_size
                )
            with self.assertRaises(ValueError):
                MembrainSegmentation().membrain_to_cets(
                    write_binary_mrc(self.test_dir), TOMO_ID, pixel_size=pixel_size
                )

    def test_given_pixel_size_overrides_header(self) -> None:
        """If given, the sampling rate is used instead of the one in the header"""
        mrc_file = write_scores_mrc(self.test_dir)  # header: PIXEL_SIZE
        scores = MembrainScores().membrain_to_cets(mrc_file, TOMO_ID, pixel_size=10.0)
        self.assertEqual(scores.coordinate_transformations[0].scale, [10.0] * 3)
