# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import mrcfile

from cets_data_model.models.models import ProbabilityMapValueType
from membrain.converters.scores import MembrainScores
from membrain.converters.segmentation import MembrainSegmentation
from membrain.tests import (
    PIXEL_SIZE,
    TOMO_ID,
    write_binary_mrc,
    write_connected_components_mrc,
    write_scores_mrc,
    write_skeleton_mrc,
)


class CetsToMembrainBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_to_membrain"))
        self.out_dir = self.test_dir / "out"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class CetsToMembrainSegmentationTest(CetsToMembrainBaseTest):
    def test_segmentation_names(self) -> None:
        """Binary and connected-component masks are named <tomo>[_<ckpt>]_segmented.mrc"""
        converter = MembrainSegmentation()
        binary_dir, components_dir = self.test_dir / "binary", self.test_dir / "cc"
        binary_dir.mkdir()
        components_dir.mkdir()
        mask_files = (
            write_binary_mrc(binary_dir),
            write_connected_components_mrc(components_dir),
        )
        for i, mask_file in enumerate(mask_files):
            mask = converter.membrain_to_cets(mask_file, TOMO_ID)
            out = converter.cets_to_membrain(mask, self.out_dir / str(i), "tomo_a")
            self.assertEqual(out.name, "tomo_a_segmented.mrc")
            self.assertTrue(out.is_symlink())
            self.assertEqual(out.resolve(), mask_file.resolve())
            out = converter.cets_to_membrain(
                mask, self.out_dir / str(i), "tomo_a", checkpoint="MemBrain_seg_v10"
            )
            self.assertEqual(out.name, "tomo_a_MemBrain_seg_v10_segmented.mrc")

    def test_skeleton_name(self) -> None:
        """Skeletons are recognized by their file name, or stated explicitly"""
        converter = MembrainSegmentation()
        skeleton = converter.membrain_to_cets(
            write_skeleton_mrc(self.test_dir), TOMO_ID
        )
        out = converter.cets_to_membrain(skeleton, self.out_dir, "tomo_a")
        self.assertEqual(out.name, "tomo_a_segmented_skel.mrc")
        binary = converter.membrain_to_cets(write_binary_mrc(self.test_dir), TOMO_ID)
        out = converter.cets_to_membrain(
            binary, self.out_dir / "explicit", "tomo_a", skeleton=True
        )
        self.assertEqual(out.name, "tomo_a_segmented_skel.mrc")
        out = converter.cets_to_membrain(
            skeleton, self.out_dir / "explicit", "tomo_b", skeleton=False
        )
        self.assertEqual(out.name, "tomo_b_segmented.mrc")

    def test_exported_mask_is_read_back(self) -> None:
        converter = MembrainSegmentation()
        mask = converter.membrain_to_cets(write_binary_mrc(self.test_dir), TOMO_ID)
        out = converter.cets_to_membrain(mask, self.out_dir, TOMO_ID, copy=True)
        self.assertFalse(out.is_symlink())
        back = converter.membrain_to_cets(out, TOMO_ID)
        self.assertEqual(back.model_copy(update={"path": mask.path}), mask)

    def test_copied_if_the_sampling_rate_differs(self) -> None:
        converter = MembrainSegmentation()
        mask_file = write_binary_mrc(self.test_dir, voxel_size=1.0)
        mask = converter.membrain_to_cets(mask_file, TOMO_ID, pixel_size=PIXEL_SIZE)
        out = converter.cets_to_membrain(mask, self.out_dir, "tomo_a")
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), PIXEL_SIZE, places=5)
        with mrcfile.open(mask_file, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 1.0, places=5)

    def test_existing_file_is_not_overwritten(self) -> None:
        converter = MembrainSegmentation()
        mask = converter.membrain_to_cets(write_binary_mrc(self.test_dir), TOMO_ID)
        other_dir = self.test_dir / "other"
        other_dir.mkdir()
        other = converter.membrain_to_cets(write_binary_mrc(other_dir), TOMO_ID)
        converter.cets_to_membrain(mask, self.out_dir, "tomo_a")
        # The same mask again is harmless
        converter.cets_to_membrain(mask, self.out_dir, "tomo_a")
        with self.assertRaises(FileExistsError):
            converter.cets_to_membrain(other, self.out_dir, "tomo_a")


class CetsToMembrainScoresTest(CetsToMembrainBaseTest):
    def test_linked_with_the_membrain_name(self) -> None:
        converter = MembrainScores()
        scores_file = write_scores_mrc(self.test_dir)
        scores = converter.membrain_to_cets(scores_file, TOMO_ID)
        out = converter.cets_to_membrain(scores, self.out_dir, "tomo_a")
        self.assertEqual(out.name, "tomo_a_scores.mrc")
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), scores_file.resolve())

    def test_not_logits(self) -> None:
        converter = MembrainScores()
        scores = converter.membrain_to_cets(write_scores_mrc(self.test_dir), TOMO_ID)
        probabilities = scores.model_copy(
            update={"value_type": ProbabilityMapValueType.probability.value}
        )
        with self.assertRaises(ValueError):
            converter.cets_to_membrain(probabilities, self.out_dir, "tomo_a")
