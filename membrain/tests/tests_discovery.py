# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

from membrain.utils.discovery import (
    MembrainOutputKind,
    SkeletonMatch,
    find_membrain_outputs,
    find_skeleton_sources,
    parse_membrain_filename,
)

CKPT = "MemBrain_seg_v10_beta.ckpt"


class MembrainDiscoveryTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_membrain_discovery"))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _touch(self, rel_path: str) -> Path:
        path = self.test_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_native_names_with_underscores(self) -> None:
        """Tomogram ids and checkpoint names may both contain underscores: the id is matched
        against the known ids (longest match, so TS_1 does not take TS_10 files)."""
        known = ["TS_1", "TS_10"]
        seg = parse_membrain_filename(Path(f"TS_10_{CKPT}_segmented.mrc"), known)
        assert seg is not None
        self.assertEqual(seg.kind, MembrainOutputKind.segmentation)
        self.assertEqual(seg.tomo_id, "TS_10")
        skel = parse_membrain_filename(Path(f"TS_1_{CKPT}_segmented_skel.mrc"), known)
        assert skel is not None
        self.assertEqual(skel.kind, MembrainOutputKind.skeleton)
        self.assertEqual(skel.tomo_id, "TS_1")
        scores = parse_membrain_filename(Path("TS_10_scores.mrc"))
        assert scores is not None
        self.assertEqual(scores.kind, MembrainOutputKind.scores)
        self.assertEqual(scores.tomo_id, "TS_10")

    def test_renamed_and_unknown_ids(self) -> None:
        """Segmentations renamed to <tomo>_segmented.mrc, and the checkpoint token split
        without known ids (exact only for ids without underscores)."""
        renamed = parse_membrain_filename(Path("TS_01_segmented.mrc"))
        assert renamed is not None
        self.assertEqual(renamed.tomo_id, "TS_01")
        native = parse_membrain_filename(Path(f"vTomo1_{CKPT}_segmented.mrc"))
        assert native is not None
        self.assertEqual(native.tomo_id, "vTomo1")

    def test_find_outputs_ignores_other_files(self) -> None:
        self._touch("vTomo1.mrc")  # input tomogram
        self._touch("run_a/notes.txt")
        scores = self._touch("run_a/TS_01_scores.mrc")
        seg = self._touch(f"run_a/TS_01_{CKPT}_segmented.mrc")
        outputs = find_membrain_outputs(self.test_dir)
        self.assertEqual(
            [(o.path, o.kind, o.tomo_id) for o in outputs],
            [  # sorted by path
                (seg, MembrainOutputKind.segmentation, "TS_01"),
                (scores, MembrainOutputKind.scores, "TS_01"),
            ],
        )

    def test_skeleton_sources(self) -> None:
        """Skeletons are matched to their source segmentation by file name"""
        # Same directory: the segmentation next to the skeleton
        self._touch("a/T1_segmented.mrc")
        same_dir = self._touch("a/T1_segmented_skel.mrc")
        # Single segmentation with the matching name, elsewhere
        single = self._touch("b/skel/T2_segmented_skel.mrc")
        self._touch("b/seg/T2_segmented.mrc")
        # No segmentation with the matching name
        orphan = self._touch("d/T4_other_skel.mrc")
        outputs = find_membrain_outputs(self.test_dir)
        sources = find_skeleton_sources(outputs)
        source = sources[same_dir]
        assert source is not None
        self.assertEqual(source.segmentation, self.test_dir / "a/T1_segmented.mrc")
        self.assertEqual(source.match, SkeletonMatch.same_directory)
        source = sources[single]
        assert source is not None
        self.assertEqual(source.segmentation, self.test_dir / "b/seg/T2_segmented.mrc")
        self.assertEqual(source.match, SkeletonMatch.unique_name)
        self.assertIsNone(sources[orphan])

    def test_skeleton_source_among_several_runs(self) -> None:
        """Several segmentations with the same name (runs on the same tomogram): the latest
        one written before the skeleton, unless the --label-path is given"""
        run1 = self._touch("run1/T3_segmented.mrc")
        run2 = self._touch("run2/T3_segmented.mrc")
        run3 = self._touch("run3/T3_segmented.mrc")
        skeleton = self._touch("skel/T3_segmented_skel.mrc")
        for path, mtime in ((run1, 100), (run2, 200), (skeleton, 300), (run3, 400)):
            os.utime(path, (mtime, mtime))
        outputs = find_membrain_outputs(self.test_dir)
        source = find_skeleton_sources(outputs)[skeleton]
        assert source is not None
        self.assertEqual(source.segmentation, run2)  # run3 is newer than the skeleton
        self.assertEqual(source.match, SkeletonMatch.latest_before_skeleton)
        self.assertEqual(source.n_candidates, 3)
        source = find_skeleton_sources(outputs, {skeleton: run1})[skeleton]
        assert source is not None
        self.assertEqual(source.segmentation, run1.resolve())
        self.assertEqual(source.match, SkeletonMatch.given)
        # All the candidates newer than the skeleton (e.g. copied files): the latest one
        os.utime(skeleton, (50, 50))
        source = find_skeleton_sources(outputs)[skeleton]
        assert source is not None
        self.assertEqual(source.segmentation, run3)
