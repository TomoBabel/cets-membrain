# -*- coding: utf-8 -*-
"""Tests of the ``cets-membrain`` command line with the data of two real MemBrain-seg runs
on the tomogram vTomo1 (``membrain/tests/data``), in both directions.

Fixtures in ``membrain/tests/data``:

- ``vTomo1.mrc``: the segmented tomogram (141 x 281 x 91 voxels).
- ``sampling_rate.txt``: its sampling rate (Å/voxel), the one of all the mrc headers.
- ``segmentation_binary/``: a run with a binary segmentation: the score map
  (``vTomo1_scores.mrc``), the segmentation (``vTomo1_segmented.mrc``) and their reference
  CETS files (``membrain_scores_vTomo1_membrain_to_cets.yaml``,
  ``membrain_binary_vTomo1_membrain_to_cets.yaml``).
- ``segmentation_connected_components_and_skel/``: a run with the segmentation stored as
  connected components and its skeleton (``vTomo1_segmented_skel.mrc``), with the score map
  and the reference CETS files of the three.

The expected values are read from the mrc files (mrcfile), and the outputs are also
compared with the reference CETS files.
"""

import contextlib
import hashlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import (
    ProbabilityMap3D,
    ProbabilityMapValueType,
    Region,
    Scale,
    SegmentationMask3D,
    SegmentationType,
)
from cets_data_model.utils.yaml_io import load_cets_object
from membrain.cli import main
from membrain.constants import (
    MEMBRANE_FEATURE_TYPE,
    MEMBRANE_LABEL_VALUE,
    SEGMENTATION_THRESHOLD_DEFAULT,
)
from membrain.tests import DATA_DIR

TESTS_DIR = DATA_DIR.parent

TOMOGRAM = DATA_DIR / "vTomo1.mrc"
SAMPLING_RATE = DATA_DIR / "sampling_rate.txt"
BINARY_DIR = DATA_DIR / "segmentation_binary"
CC_DIR = DATA_DIR / "segmentation_connected_components_and_skel"

BINARY_SCORES = BINARY_DIR / "vTomo1_scores.mrc"
BINARY = BINARY_DIR / "vTomo1_segmented.mrc"
CC_SCORES = CC_DIR / "vTomo1_scores.mrc"
CC = CC_DIR / "vTomo1_segmented.mrc"
SKELETON = CC_DIR / "vTomo1_segmented_skel.mrc"

REF_BINARY_SCORES = BINARY_DIR / "membrain_scores_vTomo1_membrain_to_cets.yaml"
REF_BINARY = BINARY_DIR / "membrain_binary_vTomo1_membrain_to_cets.yaml"
REF_CC_SCORES = CC_DIR / "membrain_scores_vTomo1_membrain_to_cets.yaml"
REF_CC = CC_DIR / "membrain_connected_components_vTomo1_membrain_to_cets.yaml"
REF_SKELETON = CC_DIR / "membrain_skeleton_vTomo1_membrain_to_cets.yaml"
# Reference CETS file (written by cets-membrain to-cets, with paths relative to
# membrain/tests): (yaml entry, the file it was converted from, --derived-from)
REFERENCES: Dict[Path, Tuple[str, Path, Optional[str]]] = {
    REF_BINARY_SCORES: ("scores", BINARY_SCORES, None),
    REF_BINARY: ("segmentation", BINARY, "membrain_scores_vTomo1"),
    REF_CC_SCORES: ("scores", CC_SCORES, None),
    REF_CC: ("segmentation", CC, "membrain_scores_vTomo1"),
    REF_SKELETON: ("segmentation", SKELETON, "membrain_connected_components_vTomo1"),
}

TOMO_ID = "vTomo1"
# Default ids of the annotations
SCORES_ID = f"membrain_scores_{TOMO_ID}"
BINARY_ID = f"membrain_binary_{TOMO_ID}"
CC_ID = f"membrain_connected_components_{TOMO_ID}"
SKELETON_ID = f"membrain_skeleton_{TOMO_ID}"
OTHER_PIXEL_SIZE = 2.5


def read_sampling_rate() -> float:
    return float(SAMPLING_RATE.read_text().strip())


def grid(path: Path) -> Tuple[Tuple[int, int, int], float]:
    """Dimensions (x, y, z) and voxel size of an mrc file."""
    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        header = mrc.header
        return (int(header.nx), int(header.ny), int(header.nz)), float(mrc.voxel_size.x)


def array_to_physical_scale(obj: SegmentationMask3D | ProbabilityMap3D) -> List[float]:
    scales = [
        t
        for t in obj.coordinate_transformations or []
        if isinstance(t, Scale) and t.name == "array_to_physical"
    ]
    if len(scales) != 1:
        raise AssertionError(f"Expected one array_to_physical scale, found {scales}")
    return list(scales[0].scale)


def without_path(obj: SegmentationMask3D | ProbabilityMap3D) -> Dict:
    dump = obj.model_dump(mode="json")
    dump.pop("path")
    return dump


def tree_digest(directory: Path) -> Dict[str, str]:
    """sha256 of every file under a directory, to check that it is not modified."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def run(args: List[str]) -> Tuple[int, str, str]:
    """Runs cets-membrain with the given arguments; returns the exit code and what it
    printed to stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class CetsMembrainCliFixturesTest(TestCase):
    pixel_size: float

    @classmethod
    def setUpClass(cls) -> None:
        cls.pixel_size = read_sampling_rate()

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_membrain_cli_fixtures"))
        self.yaml = self.dir / "vTomo1_cets.yaml"
        self.out = self.dir / "out"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def assert_run(self, args: List[str], code: int = 0) -> str:
        """Runs cets-membrain, checks its exit code and returns what it printed to
        stderr."""
        result, out, err = run(args)
        self.assertEqual(result, code, msg=f"stdout:\n{out}\nstderr:\n{err}")
        return err

    def _segmentation_args(self, mrc_file: Path, *extra: str) -> List[str]:
        return [
            "segmentation",
            "to-cets",
            "--segmentation",
            str(mrc_file),
            "--tomo-id",
            TOMO_ID,
            "-o",
            str(self.yaml),
            *extra,
        ]

    def _segmentation_to_cets(
        self, mrc_file: Path, *extra: str, yaml_file: Optional[Path] = None
    ) -> SegmentationMask3D:
        args = self._segmentation_args(mrc_file, *extra)
        if yaml_file is not None:
            args[args.index("-o") + 1] = str(yaml_file)
        self.assert_run(args)
        return load_cets_object(
            yaml_file or self.yaml, "segmentation", SegmentationMask3D
        )

    def _scores_args(self, mrc_file: Path, *extra: str) -> List[str]:
        return [
            "scores",
            "to-cets",
            "--scores",
            str(mrc_file),
            "--tomo-id",
            TOMO_ID,
            "-o",
            str(self.yaml),
            *extra,
        ]

    def _scores_to_cets(self, mrc_file: Path, *extra: str) -> ProbabilityMap3D:
        self.assert_run(self._scores_args(mrc_file, *extra))
        return load_cets_object(self.yaml, "scores", ProbabilityMap3D)

    def _from_cets(self, data_type: str, yaml_file: Path, *extra: str) -> None:
        self.assert_run(
            [
                data_type,
                "from-cets",
                "-i",
                str(yaml_file),
                "--output-dir",
                str(self.out),
                *extra,
            ]
        )

    # ------------------------------------------------------------------ fixtures
    def test_fixture_files(self) -> None:
        """All the files are on the voxel grid of the tomogram, with the sampling rate of
        sampling_rate.txt, and are consistent with each other."""
        tomo_dims, tomo_voxel = grid(TOMOGRAM)
        self.assertAlmostEqual(tomo_voxel, self.pixel_size, places=4)
        for mrc_file in (BINARY_SCORES, BINARY, CC_SCORES, CC, SKELETON):
            with self.subTest(mrc_file=mrc_file.relative_to(DATA_DIR)):
                self.assertEqual(grid(mrc_file), (tomo_dims, tomo_voxel))
        scores = mrcfile.read(BINARY_SCORES)
        binary = mrcfile.read(BINARY)
        components = mrcfile.read(CC)
        skeleton = mrcfile.read(SKELETON)
        # Both runs gave the same scores
        np.testing.assert_array_equal(mrcfile.read(CC_SCORES), scores)
        # The binary segmentation is the scores above the default threshold
        np.testing.assert_array_equal(np.unique(binary), [0, 1])
        np.testing.assert_array_equal(binary, scores > SEGMENTATION_THRESHOLD_DEFAULT)
        # The connected components label the membrane voxels of the binary segmentation
        self.assertGreater(components.max(), 1)
        np.testing.assert_array_equal(
            np.unique(components), np.arange(components.max() + 1)
        )
        np.testing.assert_array_equal(components > 0, binary > 0)
        # The skeleton is a thinning of the segmentation
        np.testing.assert_array_equal(np.unique(skeleton), [0, 1])
        self.assertTrue(np.all(binary[skeleton > 0] == 1))
        self.assertLess(skeleton.sum(), binary.sum())

    # ------------------------------------------------------------------ references
    def test_outputs_match_references(self) -> None:
        """to-cets, with the default options and the links of each run (scores ->
        segmentation -> skeleton), gives the reference (the kind of mask is detected from
        the file)."""
        for ref_yaml, (key, mrc_file, derived_from) in REFERENCES.items():
            with self.subTest(reference=ref_yaml.relative_to(DATA_DIR)):
                if key == "scores":
                    ref: SegmentationMask3D | ProbabilityMap3D = load_cets_object(
                        ref_yaml, key, ProbabilityMap3D
                    )
                    out: SegmentationMask3D | ProbabilityMap3D = self._scores_to_cets(
                        mrc_file
                    )
                else:
                    ref = load_cets_object(ref_yaml, key, SegmentationMask3D)
                    extra = ["--derived-from", derived_from] if derived_from else []
                    out = self._segmentation_to_cets(mrc_file, *extra)
                self.assertEqual(without_path(out), without_path(ref))
                # The output path is absolute; the reference one, relative to
                # membrain/tests
                assert out.path is not None and ref.path is not None
                self.assertEqual(Path(out.path), mrc_file.resolve())
                self.assertEqual((TESTS_DIR / ref.path).resolve(), mrc_file.resolve())

    def test_from_cets_of_references(self) -> None:
        """from-cets reads the reference files, run from membrain/tests (their paths are
        relative to it, and relative paths are resolved from the working directory)."""
        cases = [
            (REF_BINARY, "segmentation", BINARY, "binary_segmented.mrc"),
            (REF_CC, "segmentation", CC, "cc_segmented.mrc"),
            (REF_SKELETON, "segmentation", SKELETON, "cc_segmented_skel.mrc"),
            (REF_BINARY_SCORES, "scores", BINARY_SCORES, "binary_scores.mrc"),
            (REF_CC_SCORES, "scores", CC_SCORES, "cc_scores.mrc"),
        ]
        with contextlib.chdir(TESTS_DIR):
            for ref_yaml, data_type, src, name in cases:
                with self.subTest(reference=ref_yaml.relative_to(DATA_DIR)):
                    tomo_name = name.split("_")[0]
                    self._from_cets(data_type, ref_yaml, "--tomo-name", tomo_name)
                    out = self.out / name
                    self.assertTrue(out.is_symlink())
                    self.assertEqual(out.resolve(), src.resolve())

    # ------------------------------------------------------------------ to-cets
    def test_to_cets_values(self) -> None:
        """The defaults: the ids, the kind of mask detected, its labels and the frame of
        the tomogram, with the sampling rate of the headers."""
        dims, _ = grid(TOMOGRAM)
        scores = self._scores_to_cets(CC_SCORES)
        binary = self._segmentation_to_cets(BINARY)
        components = self._segmentation_to_cets(CC)
        skeleton = self._segmentation_to_cets(SKELETON)
        expected = {
            SCORES_ID: (scores, None),
            BINARY_ID: (binary, SegmentationType.semantic),
            CC_ID: (components, SegmentationType.instance),
            SKELETON_ID: (skeleton, SegmentationType.semantic),
        }
        for annotation_id, (obj, segmentation_type) in expected.items():
            with self.subTest(annotation_id=annotation_id):
                self.assertEqual(obj.id, annotation_id)
                self.assertEqual(obj.source_tomogram_id, TOMO_ID)
                self.assertIsNone(obj.derived_from_annotation_id)
                self.assertEqual((obj.width, obj.height, obj.depth), dims)
                self.assertEqual(
                    [cs.name for cs in obj.coordinate_systems or []],
                    [f"{TOMO_ID}_array", f"{TOMO_ID}_physical"],
                )
                self.assertEqual(array_to_physical_scale(obj), [self.pixel_size] * 3)
                assert obj.annotated_object is not None
                self.assertEqual(
                    obj.annotated_object.feature_type, MEMBRANE_FEATURE_TYPE.value
                )
                if isinstance(obj, SegmentationMask3D):
                    self.assertEqual(obj.segmentation_type, segmentation_type)
        self.assertEqual(scores.value_type, ProbabilityMapValueType.logit)
        self.assertEqual(scores.threshold, SEGMENTATION_THRESHOLD_DEFAULT)
        for mask in (binary, skeleton):
            self.assertEqual(
                [label.label_value for label in mask.labels or []],
                [MEMBRANE_LABEL_VALUE],
            )
        # Each value 1..N of the connected components is a membrane instance
        self.assertEqual(components.labels, [])

    def test_to_cets_derived_from(self) -> None:
        """scores -> connected components -> skeleton: the links resolve in a region."""
        scores = self._scores_to_cets(CC_SCORES)
        components = self._segmentation_to_cets(
            CC, "--derived-from", scores.id, yaml_file=self.dir / "cc.yaml"
        )
        skeleton = self._segmentation_to_cets(
            SKELETON, "--derived-from", components.id, yaml_file=self.dir / "skel.yaml"
        )
        self.assertEqual(components.derived_from_annotation_id, SCORES_ID)
        self.assertEqual(skeleton.derived_from_annotation_id, CC_ID)
        region = Region(
            id=TOMO_ID,
            annotations=[
                obj.model_dump(mode="json") for obj in (scores, components, skeleton)
            ],
        )
        by_id = {a.id: a for a in region.annotations or []}
        self.assertIsInstance(by_id[SCORES_ID], ProbabilityMap3D)
        self.assertIsInstance(by_id[CC_ID], SegmentationMask3D)
        self.assertEqual(by_id[by_id[SKELETON_ID].derived_from_annotation_id].id, CC_ID)

    def test_to_cets_options(self) -> None:
        # The kind of mask, instead of the detected one
        mask = self._segmentation_to_cets(CC, "--mask-type", "binary")
        self.assertEqual(mask.id, BINARY_ID)
        self.assertEqual(mask.segmentation_type, SegmentationType.semantic)
        mask = self._segmentation_to_cets(BINARY, "--mask-type", "skeleton")
        self.assertEqual(mask.id, SKELETON_ID)
        # Sampling rate and annotation id
        mask = self._segmentation_to_cets(
            SKELETON, "--pixel-size", str(OTHER_PIXEL_SIZE), "--annotation-id", "skel_2"
        )
        self.assertEqual(mask.id, "skel_2")
        self.assertEqual(array_to_physical_scale(mask), [OTHER_PIXEL_SIZE] * 3)
        # Threshold of the scores
        scores = self._scores_to_cets(BINARY_SCORES, "--threshold", "1.5")
        self.assertEqual(scores.threshold, 1.5)
        scores = self._scores_to_cets(BINARY_SCORES, "--unknown-threshold")
        self.assertIsNone(scores.threshold)
        scores = self._scores_to_cets(
            BINARY_SCORES, "--pixel-size", str(OTHER_PIXEL_SIZE)
        )
        self.assertEqual(array_to_physical_scale(scores), [OTHER_PIXEL_SIZE] * 3)

    # ------------------------------------------------------------------ from-cets
    def test_segmentation_from_cets(self) -> None:
        """The masks are linked with the MemBrain-seg names."""
        self._segmentation_to_cets(BINARY)
        self._from_cets("segmentation", self.yaml)
        # Named after the source tomogram id
        out = self.out / f"{TOMO_ID}_segmented.mrc"
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), BINARY.resolve())
        self._from_cets(
            "segmentation",
            self.yaml,
            "--checkpoint",
            "MemBrain_seg_v10_alpha.ckpt",
            "--tomo-name",
            "tomo_a",
        )
        out = self.out / "tomo_a_MemBrain_seg_v10_alpha.ckpt_segmented.mrc"
        self.assertEqual(out.resolve(), BINARY.resolve())
        # Skeletons are detected from the name, unless told otherwise
        self._segmentation_to_cets(SKELETON)
        self._from_cets("segmentation", self.yaml)
        out = self.out / f"{TOMO_ID}_segmented_skel.mrc"
        self.assertEqual(out.resolve(), SKELETON.resolve())
        self._from_cets("segmentation", self.yaml, "--no-skeleton", "--tomo-name", "b")
        self.assertEqual((self.out / "b_segmented.mrc").resolve(), SKELETON.resolve())
        self._segmentation_to_cets(CC)
        self._from_cets("segmentation", self.yaml, "--skeleton", "--tomo-name", "c")
        self.assertEqual((self.out / "c_segmented_skel.mrc").resolve(), CC.resolve())

    def test_from_cets_copy(self) -> None:
        """--copy writes the whole file, identical to the input."""
        self._segmentation_to_cets(CC)
        self._from_cets("segmentation", self.yaml, "--copy")
        self._scores_to_cets(CC_SCORES)
        self._from_cets("scores", self.yaml, "--copy")
        for out, src in (
            (self.out / f"{TOMO_ID}_segmented.mrc", CC),
            (self.out / f"{TOMO_ID}_scores.mrc", CC_SCORES),
        ):
            with self.subTest(src=src.name):
                self.assertTrue(out.is_file())
                self.assertFalse(out.is_symlink())
                self.assertEqual(out.read_bytes(), src.read_bytes())

    def test_scores_from_cets(self) -> None:
        self._scores_to_cets(BINARY_SCORES)
        self._from_cets("scores", self.yaml)
        out = self.out / f"{TOMO_ID}_scores.mrc"
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), BINARY_SCORES.resolve())
        self._from_cets("scores", self.yaml, "--tomo-name", "tomo_a")
        self.assertEqual(
            (self.out / "tomo_a_scores.mrc").resolve(), BINARY_SCORES.resolve()
        )
        # Writing it again is not an error
        self._from_cets("scores", self.yaml)

    def test_from_cets_other_voxel_size(self) -> None:
        """If the CETS sampling rate is not the one of the header, the file is copied
        (whole) and the header of the copy updated."""
        cases = (
            ("segmentation", SKELETON, f"{TOMO_ID}_segmented_skel.mrc"),
            ("scores", CC_SCORES, f"{TOMO_ID}_scores.mrc"),
        )
        for data_type, src, name in cases:
            with self.subTest(data_type=data_type):
                if data_type == "scores":
                    self._scores_to_cets(src, "--pixel-size", str(OTHER_PIXEL_SIZE))
                else:
                    self._segmentation_to_cets(
                        src, "--pixel-size", str(OTHER_PIXEL_SIZE)
                    )
                self._from_cets(data_type, self.yaml)
                out = self.out / name
                self.assertFalse(out.is_symlink())
                self.assertEqual(grid(out), (grid(src)[0], OTHER_PIXEL_SIZE))
                self.assertEqual(out.stat().st_size, src.stat().st_size)
                np.testing.assert_array_equal(mrcfile.read(out), mrcfile.read(src))

    # ------------------------------------------------------------------ errors
    def test_errors(self) -> None:
        """Conversion errors exit with code 1 and a one-line message."""
        missing = self._segmentation_args(BINARY_DIR / "missing.mrc")
        not_mrc = self._scores_args(SAMPLING_RATE)
        zero_pixel_size = self._scores_args(BINARY_SCORES, "--pixel-size", "0")
        for args in (missing, not_mrc, zero_pixel_size):
            with self.subTest(args=args):
                err = self.assert_run(args, 1)
                self.assertEqual(len(err.strip().splitlines()), 1)
                self.assertFalse(self.yaml.exists())
        # A segmentation is not a score map, nor the other way round
        self._segmentation_to_cets(BINARY)
        self.assert_run(
            [
                "scores",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.out),
            ],
            1,
        )
        self.assert_run(
            [
                "segmentation",
                "from-cets",
                "-i",
                str(REF_BINARY_SCORES),
                "--output-dir",
                str(self.out),
            ],
            1,
        )
        # The output exists and is another file
        self._from_cets("segmentation", self.yaml)
        self._segmentation_to_cets(CC)
        self.assert_run(
            [
                "segmentation",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.out),
            ],
            1,
        )
        self.assertEqual(
            (self.out / f"{TOMO_ID}_segmented.mrc").resolve(), BINARY.resolve()
        )

    def test_input_files_are_not_modified(self) -> None:
        before = tree_digest(DATA_DIR)
        for mrc_file in (BINARY, CC, SKELETON):
            # Copied, with the header updated, and linked
            self._segmentation_to_cets(mrc_file, "--pixel-size", str(OTHER_PIXEL_SIZE))
            run_name = (
                mrc_file.parent.name
            )  # binary and connected components: same name
            self._from_cets("segmentation", self.yaml, "--tomo-name", run_name)
            self._segmentation_to_cets(mrc_file)
            self._from_cets("segmentation", self.yaml, "--tomo-name", f"{run_name}_l")
        for mrc_file in (BINARY_SCORES, CC_SCORES):
            self._scores_to_cets(mrc_file)
            self._from_cets("scores", self.yaml, "--tomo-name", mrc_file.parent.name)
        self.assertEqual(tree_digest(DATA_DIR), before)

    def test_invalid_arguments(self) -> None:
        """Usage errors exit with code 2 (argparse) before running any conversion."""
        invalid: List[List[str]] = [
            [],
            ["segmentation"],
            ["tomogram", "to-cets"],
            # --tomo-id is required
            [
                "segmentation",
                "to-cets",
                "--segmentation",
                str(BINARY),
                "-o",
                str(self.yaml),
            ],
            self._segmentation_args(CC, "--mask-type", "instance"),
            self._segmentation_args(CC, "--pixel-size", "one"),
            # --threshold and --unknown-threshold exclude each other
            self._scores_args(CC_SCORES, "--threshold", "0.5", "--unknown-threshold"),
            ["scores", "from-cets", "-i", str(REF_CC_SCORES)],  # --output-dir
        ]
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as ctx:
                    run(args)
                self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(self.yaml.exists())

    def test_help(self) -> None:
        for args in (
            ["--help"],
            ["segmentation", "to-cets", "--help"],
            ["scores", "from-cets", "--help"],
        ):
            with self.subTest(args=args):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    with self.assertRaises(SystemExit) as ctx:
                        main(args)
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("cets-membrain", out.getvalue())
