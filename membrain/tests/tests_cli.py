# -*- coding: utf-8 -*-
import contextlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest import TestCase

import mrcfile

from cets_data_model.models.models import (
    ProbabilityMap3D,
    SegmentationMask3D,
    SegmentationType,
)
from cets_data_model.utils.yaml_io import load_cets_object
from membrain.cli import main
from membrain.tests import (
    PIXEL_SIZE,
    TOMO_ID,
    write_binary_mrc,
    write_connected_components_mrc,
    write_scores_mrc,
    write_skeleton_mrc,
)


def run(args: List[str]) -> int:
    """Runs cets-membrain with the given arguments, hiding its messages."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return main(args)


class CetsMembrainCliTest(TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_membrain_cli"))
        self.yaml = self.dir / "cets.yaml"
        self.out_dir = self.dir / "out" / "sub"  # created by the command

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _segmentation_to_cets(self, mrc_file: Path, *options: str) -> None:
        code = run(
            [
                "segmentation",
                "to-cets",
                "--segmentation",
                str(mrc_file),
                "--tomo-id",
                TOMO_ID,
                *options,
                "-o",
                str(self.yaml),
            ]
        )
        self.assertEqual(code, 0)

    def test_segmentation_to_cets(self) -> None:
        self._segmentation_to_cets(
            write_binary_mrc(self.dir),
            "--derived-from",
            "membrain_scores_TS_01",
            "--annotation-id",
            "seg_1",
            "--pixel-size",
            "7.5",
        )
        mask = load_cets_object(self.yaml, "segmentation", SegmentationMask3D)
        self.assertEqual(mask.id, "seg_1")
        self.assertEqual(mask.source_tomogram_id, TOMO_ID)
        self.assertEqual(mask.derived_from_annotation_id, "membrain_scores_TS_01")
        self.assertEqual(mask.segmentation_type, SegmentationType.semantic)
        self.assertEqual(mask.coordinate_transformations[0].scale, [7.5] * 3)

    def test_segmentation_mask_type(self) -> None:
        """Connected components are detected, unless the mask type is given"""
        components = write_connected_components_mrc(self.dir)
        self._segmentation_to_cets(components)
        mask = load_cets_object(self.yaml, "segmentation", SegmentationMask3D)
        self.assertEqual(mask.segmentation_type, SegmentationType.instance)
        self.assertEqual(mask.id, f"membrain_connected_components_{TOMO_ID}")
        self._segmentation_to_cets(components, "--mask-type", "binary")
        mask = load_cets_object(self.yaml, "segmentation", SegmentationMask3D)
        self.assertEqual(mask.segmentation_type, SegmentationType.semantic)

    def test_segmentation_round_trip(self) -> None:
        mask_file = write_binary_mrc(self.dir)
        self._segmentation_to_cets(mask_file)
        code = run(
            [
                "segmentation",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.out_dir),
                "--checkpoint",
                "MemBrain_seg_v10",
            ]
        )
        self.assertEqual(code, 0)
        # Named after the source tomogram id by default; linked
        out = self.out_dir / f"{TOMO_ID}_MemBrain_seg_v10_segmented.mrc"
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), mask_file.resolve())
        # Copied, as a skeleton, with another tomogram name
        code = run(
            [
                "segmentation",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.out_dir),
                "--tomo-name",
                "tomo_a",
                "--skeleton",
                "--copy",
            ]
        )
        self.assertEqual(code, 0)
        out = self.out_dir / "tomo_a_segmented_skel.mrc"
        self.assertTrue(out.is_file())
        self.assertFalse(out.is_symlink())

    def test_skeleton_round_trip(self) -> None:
        self._segmentation_to_cets(write_skeleton_mrc(self.dir))
        mask = load_cets_object(self.yaml, "segmentation", SegmentationMask3D)
        self.assertEqual(mask.id, f"membrain_skeleton_{TOMO_ID}")
        code = run(
            [
                "segmentation",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.out_dir),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue((self.out_dir / f"{TOMO_ID}_segmented_skel.mrc").exists())

    def test_scores_round_trip(self) -> None:
        scores_file = write_scores_mrc(self.dir)
        to_cets = [
            "scores",
            "to-cets",
            "--scores",
            str(scores_file),
            "--tomo-id",
            TOMO_ID,
            "-o",
            str(self.yaml),
        ]
        self.assertEqual(run(to_cets), 0)
        scores = load_cets_object(self.yaml, "scores", ProbabilityMap3D)
        self.assertEqual(scores.threshold, 0.0)
        self.assertEqual(scores.id, f"membrain_scores_{TOMO_ID}")
        self.assertEqual(run(to_cets + ["--threshold", "1.5"]), 0)
        self.assertEqual(
            load_cets_object(self.yaml, "scores", ProbabilityMap3D).threshold, 1.5
        )
        self.assertEqual(run(to_cets + ["--unknown-threshold"]), 0)
        self.assertIsNone(
            load_cets_object(self.yaml, "scores", ProbabilityMap3D).threshold
        )
        # The header pixel size differs from the annotation one: copied and fixed
        self.assertEqual(run(to_cets + ["--pixel-size", "10"]), 0)
        code = run(
            [
                "scores",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.out_dir),
            ]
        )
        self.assertEqual(code, 0)
        out = self.out_dir / f"{TOMO_ID}_scores.mrc"
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 10.0, places=4)
            self.assertNotAlmostEqual(float(mrc.voxel_size.x), PIXEL_SIZE, places=4)

    def test_errors_give_exit_code_1(self) -> None:
        missing = [
            "segmentation",
            "to-cets",
            "--segmentation",
            str(self.dir / "missing.mrc"),
            "--tomo-id",
            TOMO_ID,
            "-o",
            str(self.yaml),
        ]
        self.assertEqual(run(missing), 1)
        # A segmentation is not a score map
        self._segmentation_to_cets(write_binary_mrc(self.dir))
        wrong_type = [
            "scores",
            "from-cets",
            "-i",
            str(self.yaml),
            "--output-dir",
            str(self.out_dir),
        ]
        self.assertEqual(run(wrong_type), 1)
        # The output file already exists and is another file
        self.out_dir.mkdir(parents=True)
        (self.out_dir / f"{TOMO_ID}_segmented.mrc").write_text("other")
        exists = [
            "segmentation",
            "from-cets",
            "-i",
            str(self.yaml),
            "--output-dir",
            str(self.out_dir),
        ]
        self.assertEqual(run(exists), 1)
